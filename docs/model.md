# How the model works

A reference for the DUET-Prob network behind the forecast: what the checkpoint actually
contains, what happens to a tensor on its way through, and which parts of the vendored
code are inert. Written against checkpoint `1c7a531768d8`
(SHA256 `1c7a531768d883af0c70aea1d7fe62fe59638000bf70097d61fb90f2bc4309b0`).

For *where the code came from* and what was stripped, see
[`eisbach/model/PROVENANCE.md`](../eisbach/model/PROVENANCE.md). This document is about
what it *does*. For what to change about it and why, see [`docs/PRD.md`](PRD.md) — the
measurements in "Traps" and "What is inert" below are the evidence behind its R6.

Everything below was verified by instrumenting a real forward pass, not read off the
source alone. Where a claim rests on measurement, the measurement is quoted.

## The checkpoint's real hyperparameters

`config_dict` carries 73 keys, most of them training leftovers. The ones with effect:

| | |
| --- | --- |
| `seq_len` 384 (16 days hourly) | `horizon` 96 |
| `enc_in` 3, `CI` False | `d_model` 256, `d_ff` 128, `n_heads` 2, `e_layers` 1 |
| `num_linear_experts` 2 | `num_univariate_esn_experts` 8 |
| `num_multivariate_esn_experts` 1 | `k` 8 of 11 experts |
| `reservoir_size_uni` 16 | `reservoir_size_multi` 128 |
| `leak_rate_uni` 0.877 | `leak_rate_multi` 0.128 |
| `input_scaling_uni` 0.439 (**ignored — see Traps**) | `input_scaling_multi` 0.0328 |
| `norm_mode` `subtract_median` | `distribution_family` `student_t` |
| `projection_head_layers` 3, dim factor 1, dropout 0.241 | `moving_avg` 49 |
| `use_channel_adjacency_prior` **False** | `channel_adjacency_prior` `[[1,1,1],[0,1,1],[0,0,1]]` — stored, never applied |

**137 tensors, 2 696 265 parameters**, loading clean under `strict=True`.

The config also records how the checkpoint was selected: `optimization_metric='cvar'`,
`optimization_target_channel='wassertemp'`, `cvar_alpha=0.05`,
`log_dir='results/optuna_heuristic/eisbach_96_studentt/trial_157'`. So this is Optuna
trial 157, chosen on the CVaR of the water channel. Inference ignores all four.

`W_res` and `W_in` are registered as **buffers**, so the "random" reservoirs are the
checkpoint's own matrices, not re-randomized on load. This matters: the reservoirs are
reproducible, and the spectral-radius/sparsity config knobs have no inference effect
because `load_state_dict` overwrites whatever `__init__` generated.

## Data flow

With B=1, L=384, N=3, D=256, H=96, Q=7. Measured cost: **~86 ms per forward+icdf on CPU**.

```
input                                                  [1, 384, 3]   api.py:166
│
├─ RevIN 'norm'                     duet_prob_model.py:183 → RevIN.py:56-61
│    location = per-channel median over the window      RevIN.py:85   (detached)
│    scale    = RMS about that median                   RevIN.py:88   (detached)
│    x_norm   = (x-loc)/scale * gamma + beta            RevIN.py:92-94
│    gamma = [0.135, 1.590, 0.146]   beta = [0.032, 0.090, 0.012]
│
├─ Linear_extractor_cluster(x_norm)                     duet_prob_model.py:195
│   ├─ router: mean over CHANNELS → [1,384]             distributional_router_encoder.py:17
│   │          Linear(384→32) ReLU Linear(32→11), no bias → logits [1,11]
│   ├─ softmax → topk(9) → keep 8 → renormalize         linear_extractor_cluster.py:148-155
│   ├─ expert 0,1   linear: series_decomp(49), 2×Linear(384→256), summed
│   ├─ expert 2..9  univariate ESN: 16-unit reservoir per channel, Linear(16→256)
│   ├─ expert 10    multivariate ESN: W_in (128,3), 128 units, Linear(128→768) split per channel
│   └─ gate-weighted sum                                → [1,256,3]
│
├─ Mahalanobis_mask(x_norm)                             duet_prob_model.py:217
│    |rfft| → [1,3,193]; pairwise diff through learned A (193,193)
│    → p_learned [1,3,3] → hard Gumbel-Bernoulli → mask [1,1,3,3]
│
├─ Channel_transformer (1 pre-LN layer, 2 heads)        → [1,3,256]
│
├─ per channel i: MLPProjectionHead (3 residual blocks, hidden 288)
│    → [1,288] → rearrange 'b (h p) -> b h p', h=96     → [1,3,96,3]
│
├─ StudentTOutput                    student_t_standalone.py:158-189
│    chunk(-1,3) → df = softplus+2.0, loc = raw, scale = softplus+1e-6
│
├─ DenormalizingDistribution                            duet_prob_model.py:18-30
└─ .icdf(QUANTILES) → [1,3,96,7] → [96,21]              api.py:175,184
```

## The mixture of experts

**The router sees the channel mean, not the channels.**
`distributional_router_encoder.py:17` does `torch.mean(x, dim=-1)` — averaging over the
*channel* axis, producing one `[B, 384]` series. With `CI=False` all three channels share
one expert selection and one set of gate weights. Since RevIN's learned `gamma` is
`[0.135, 1.590, 0.146]`, that mean is dominated ~11× by `airtemp_96`. The router is in
effect an air-temperature-shape-driven switch.

**Gating softmaxes before the topk**, which is unusual
(`linear_extractor_cluster.py:148-155`): softmax over 11 logits → `topk(k+1)=9` → keep the
first `k=8` → renormalize → scatter. Eight of eleven experts fire; gates sum to 1.

**Noisy gating is training-only.** `linear_extractor_cluster.py:137` gates the noise on
`self.noisy_gating and train`, and `train` is `self.training` = False at inference.
`noisy_gating: True` in the config has zero effect on a forecast.

All eleven experts consume the same `[1,384,3]` and emit `[1,256,3]`, so they are
interchangeable in the sum:

- **Linear experts (0, 1)** — `series_decomp` with a 49-hour moving average splits
  seasonal from trend; two `Linear(384→256)` maps along the *time* axis, summed. Channel
  independent. (Note `self.pred_len = configs.d_model` at `linear_pattern_extractor.py:19`
  — the "prediction length" here is a feature width, not a horizon.)
- **Univariate ESN experts (2–9)** — each channel passes separately through the same
  16-unit reservoir, then `Linear(16→256)`. Channel independent.
- **Multivariate ESN expert (10)** — `W_in (128,3)` mixes all three channels into one
  128-unit reservoir; the structured readout `Linear(128→768)` is reshaped so each channel
  gets its own 256-slice. **The only expert that mixes channels.**

Measured over 222 historical windows: mean gate per expert 0.056–0.113 (fairly flat),
selection frequency 0.41–0.82. Expert 1, a linear expert, is the least used.

## The reservoir experts

The state loop (`reservoir_expert.py:9-18`, `@torch.jit.script`):

```
h ← (1-leak)·h + leak·tanh(h @ W_resᵀ + u_t @ W_inᵀ)
```

`h` starts at zeros, runs 384 steps, and **only the final state is read out** — this is a
fixed-length encoder, not a sequence model. There is no washout, so the first ~1/(1−leak)
steps are transient. With `leak_rate_multi = 0.128` the multivariate reservoir has an
effective memory of roughly 8 hours and takes a long time to forget `h = 0`.

Only `readout` is learned. Measured actual spectral radii: **0.984** univariate (target
0.966) and **0.899** multivariate (target 0.942) — the 20-step power iteration in
`_approximate_spectral_radius` is only approximate. Measured sparsity 0.328 / 0.382,
matching targets.

`input_scaling_multi = 0.0328` is tiny (measured `W_in.std() = 0.0332`). Combined with the
low leak rate, the multivariate reservoir behaves close to a **linear low-pass integrator**
of the three channels.

## Channel masking, and why the channel order is load-bearing

`Mahalanobis_mask` (`masked_attention.py:153-248`) computes an FFT-magnitude distance
between channels through a learned `A (193,193)`, inverts it, zeroes the diagonal,
normalizes each row by that row's max (line 196 — hence asymmetric), adds the identity
back, and samples a hard Bernoulli via Gumbel-softmax. The result gates which channels a
channel may attend to. Row = query channel, column = key channel.

**On real data the mask is stable, and it always cuts water off from air.** Over 222
historical windows built from `data/archive/`:

```
mean p_learned      wassertemp  airtemp_96  pressure_96   ← key
   wassertemp   →      1.0        0.0049        1.0
   airtemp_96   →      0.9988     1.0          0.975
   pressure_96  →      1.0        0.0048       1.0
```

`p[wassertemp ← airtemp_96] > 0.5` in **0 of 222** windows (maximum 0.0088). The water
representation never receives air temperature through the channel transformer. Because
row 0's off-diagonals are ~0 and ~1, the water forecast is deterministic: 30 repeated
calls on one window gave a run-to-run spread of exactly 0.000000.

### Where the positional dependency actually lives

`duet_prob_model.py:134` and `:232-234`:

```python
self.channel_names = list(config.channel_bounds.keys())   # :134
...
for i, name in enumerate(self.channel_names):             # :232
    channel_feature = channel_group_feature[:, i, :]      # :234
    flat_params = self.projection_heads[name](channel_feature)
```

`projection_heads["wassertemp"]` is bound to **input slot 0** purely by the insertion order
of the checkpoint's `channel_bounds` dict. Nothing checks the name against the data.
`api.py:44` (`SERIES_ORDER`) and `api.py:149` are the only things keeping them aligned.

Confirmed failure mode: swapping input columns 0↔1 makes the `wassertemp` head emit
`[19.48, 20.72, 21.62, 22.31]` — air-temperature values — instead of
`[11.87, 12.34, 12.74, 13.06]`. **No error is raised, and in a Munich summer those numbers
pass `validate.PLAUSIBLE_RANGES` and look like a warm day.**

Secondary positional dependencies on the same axis: `RevIN.gamma/beta (1,1,3)`, the
multivariate `W_in (128,3)` columns and its readout split, the Mahalanobis rows and
columns, and the transformer's attention over the channel axis.

## The distribution head

Per channel per horizon step the head emits three raw scalars. The flat 288-vector is
unpacked horizon-major, so step *h* sits at flat indices `3h, 3h+1, 3h+2`, in the order
**df, loc, scale** (`student_t_standalone.py:171`):

```
df    = softplus(raw) + 2.0     ← forces finite variance, floors tail weight
loc   = raw
scale = softplus(raw) + 1e-6
```

Measured on a real window, `df` ranges 4.9–13.8 across horizon and channel: the model does
use genuinely heavy tails, and never has infinite variance.

### What the normalization does to the forecast

Location is the per-channel **median** of the 384-hour window; scale is the **RMS about
that median**; both detached. `DenormalizingDistribution` then applies
`value * std + mean`. Verified numerically:

- The water forecast is **exactly affine-equivariant** in the water input. `+1 °C` on the
  whole input → `+1 °C` on every quantile at every horizon (measured delta 1.0000).
  `×2 − 5` reproduces `2·forecast − 5` to 3.4e-5.
- Every covariate is **affine-invariant**. `airtemp_96 + 3 °C` uniformly → maximum change
  in the water forecast **1.9e-6**. `airtemp_96 × 2.5 + 7` → the same. *The absolute level
  and the amplitude of the air-temperature forecast carry no information whatsoever*; only
  its normalized shape does.
- Predicted interval width is therefore **proportional to the water window's RMS**. On the
  window tested, RMS = 0.787 and the normalized scale at h=96 was 0.92, giving ~0.72 °C.
- `gamma`/`beta` are not undone on output — `RevIN._denormalize` is never called; the
  inverse is folded into the learned heads.

### Monotonicity

Quantile crossing is **structurally impossible** within a (channel, horizon) cell: `loc`,
`scale` and `df` are fixed there and `icdf` is monotone in `q`. Verified. Two caveats:

- Nothing enforces smoothness *across* horizon steps — the 96 steps are independent head
  outputs.
- `channel_bounds`' `lower`/`upper` values are read only for `.keys()`. There is **no
  clipping to physical bounds** anywhere in inference; `eisbach/validate.py` is the only
  guard.

## Traps for whoever trains next

None of these change today's numbers. All four are live hazards for the next training run.

**`input_scaling_uni` is silently ignored.** `reservoir_expert.py:112` reads
`getattr(config, 'input_scaling', 1.0)`, but no such key exists — the config defines
`input_scaling_uni` (`config.py:40`). Confirmed: `hasattr(config, 'input_scaling')` is
False, and the checkpoint's univariate `W_in.std()` is 1.082, i.e. scaling 1.0, not the
0.439 that was searched for. **Optuna tuned a parameter with no effect.**

**The mask diagonal is NaN.** `bernoulli_gumbel_rsample` clamps to `1.0 - 1e-9`, but in
float32 that *is* 1.0, so `log(p/(1-p))` → `inf` and `gumbel_softmax` → NaN. It is harmless
only because `FullAttention` masks with `torch.where(current_mask == 0, ...)` and
`NaN != 0`, so NaN positions are treated as keep. Every entry normalizing to exactly 1.0
takes this path rather than being sampled.

**`icdf(0.5)` survives only through float32 underflow.** Line 88 computes
`1/(z + 1e-8) - 1`; at `q = 0.5` scipy returns `z = 1.0` exactly. In float32,
`1.0/(1.0+1e-8) - 1 == 0.0` → `sqrt(0)`. In float64 it is `-1e-8` → `sqrt` of a negative →
**NaN**. Confirmed: `CustomStudentT(...).icdf([0, 0.01, 0.5, 0.99, 1])` gives
`[-inf, -3.365, 0.0, 3.365, inf]` in float32 and `[-inf, -3.365, nan, 3.365, inf]` in
float64. Promoting the head to double silently destroys the median.

**Projection heads are bound to slots, not names** — see above.

## What is inert for this checkpoint

Inventory only. `eisbach/model/duet/` is kept byte-identical to upstream on purpose; none
of this should be edited here.

**Config knobs with no inference effect.** Never read at all: `fc_dropout`, `freq`,
`loss_coef`, the derived `label_len`/`input_size`, and the checkpoint-only keys `dec_in`,
`c_out`, `distribution_family`, `cvar_alpha`, `optimization_metric`,
`optimization_target_channel`, `data_file`, `log_dir`, `train_ratio_in_tv`,
`warmup_epochs`, `early_stopping_delta`, `min_epochs_for_pruning`, `loss_target_clip`,
`max_memory_gb`, `enable_diagnostic_plots`. Passed but discarded: `factor`,
`output_attention`. Training-only: `lr`, `lradj`, `num_epochs`, `accumulation_steps`,
`batch_size`, `patience`, `num_workers`, `use_agc`, `agc_lambda`, `esn_*_weight_decay`,
`interim_validation_seconds`, `profile_epoch`, `tqdm_*`. No-ops under `eval()`: `dropout`,
`projection_head_dropout`. Overwritten: `quantiles` (at `api.py:92`). Overwritten by
`load_state_dict`: `spectral_radius_*`, `sparsity_*`.

`distribution_family` deserves a callout: `DUETProbModel` hardcodes `StudentTOutput()` at
`:130` and `:174`, so the checkpoint's `student_t` is never consulted. The vendored tree
*cannot* build any other family — which is why upstream HEAD's `ZIEGPD_M1` override is a
load-time error rather than a silent behaviour change.

**Branches never entered.**

| Location | Condition |
| --- | --- |
| `duet_prob_model.py:187-191` | `CI` is False |
| `duet_prob_model.py:156-170` | `use_channel_adjacency_prior` is False |
| `duet_prob_model.py:222-227` | `n_vars > 1` is always true |
| `duet_prob_model.py:264-296` | `get_parameter_groups` — training only |
| `masked_attention.py:205-207` | prior is never set, so `p_final == p_learned` |
| `masked_attention.py:59-66` | `Encoder.conv_layers` — none are ever passed |
| `masked_attention.py:98-101` | default causal mask — one is always supplied |
| `linear_extractor_cluster.py:115-131, 137-143, 157-158` | noisy gating — training only |
| `linear_extractor_cluster.py:18-23, 33-39` | empty-gates path, unreachable with `k=8` |
| `linear_extractor_cluster.py:59-60, 76-77` | `expert_to_gates`, legacy fallback |
| `expert_factory.py:42-50` | legacy fallback and its `DeprecationWarning` |
| `linear_pattern_extractor.py:24-37, 53-62` | `individual=True` — never used |
| `linear_pattern_extractor.py:78` | `dec_out[:, -256:, :]` — a no-op slice |
| `reservoir_expert.py:128-129, 173-174` | empty-batch guards |
| `RevIN.py:63-65, 71-81, 97-117` | `'denorm'` mode, `identity`, `subtract_last` |
| `student_t_standalone.py:130-131, 145` | `num_layers == 0` — this checkpoint uses 3 |
| `duet_prob_model.py:33-54, 83-88` | `batch_shape`, `stddev`, `log_prob`, `normalize_value` |
| `duet_prob_model.py:262` | 7 of 9 returned values dropped at `api.py:169` |

**Inert whole units.** In `Autoformer_EncDec.py` only `moving_avg` and `series_decomp` are
used; `my_Layernorm`, `series_decomp_multi`, `EncoderLayer`, `Encoder`, `DecoderLayer` and
`Decoder` are dead. Careful: the `Encoder`/`EncoderLayer` that *are* used come from
`masked_attention.py` — a name collision worth knowing about.
`_approximate_spectral_radius` runs at init and its result is thrown away by
`load_state_dict`.

**Not dead, contrary to its comment:** `_scipy_betaincinv` (`student_t_standalone.py:14`).
`torch.special.betaincinv` does not exist in torch 2.14, so line 81 takes the SciPy branch.
On CPU that is fine; on CUDA or MPS it forces a `.cpu().numpy()` round-trip per `icdf`
call.
