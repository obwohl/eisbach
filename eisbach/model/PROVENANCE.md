# Provenance of `eisbach/model/`

This package is a **vendored, inference-only** copy of the DUET-Prob model used by
the Eisbach water-temperature forecast. It replaces the `ts_proba_cuda` git
submodule, which carried a full research/training repository (~350 MB of history)
for the sake of a single forward pass.

| | |
| --- | --- |
| Source repository | <https://github.com/obwohl/ts_proba_cuda> |
| Vendored commit | `a8de694266a629124687a8f2b9fcfdba15a3590c` |
| Vendored on | 2026-08-04 |
| Original entrypoint | `ts_proba_cuda/run_single_forecast.py` (was run as a subprocess) |
| Checkpoint | `checkpoints/best_model.pt` @ same commit, SHA256 `1c7a531768d883af0c70aea1d7fe62fe59638000bf70097d61fb90f2bc4309b0`, 10 843 610 bytes |

## ⚠️ Upstream HEAD is NOT compatible with this checkpoint

Vendor from commit `a8de6942…` and nothing else. `ts_proba_cuda` HEAD has diverged
materially from this commit and **does not load `best_model.pt`**. If you ever need
to re-sync this package, check out that exact commit first:

```sh
git clone https://github.com/obwohl/ts_proba_cuda
git -C ts_proba_cuda checkout a8de694266a629124687a8f2b9fcfdba15a3590c
```

A symptom of vendoring the wrong tree is `load_state_dict` reporting missing or
unexpected keys, or the config carrying `distribution_family='ZIEGPD_M1'` instead
of `student_t`. At the pinned commit, loading with `strict=True` produces zero
missing and zero unexpected keys across all 137 state-dict tensors.

Concretely, upstream HEAD (`be96a60`) rewrote `run_single_forecast.py` to hardcode

```python
config_dict['distribution_family'] = 'ZIEGPD_M1'   # not at a8de6942
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

Neither line exists at `a8de6942`. At HEAD the first raises
`ValueError: Unknown distribution_family: ZIEGPD_M1`, which the script's bare
`try/except` swallows before exiting 0 — so the caller's `check=True` sees success
and reads a stale CSV. Vendoring makes that class of silent breakage impossible.

## Files taken from upstream

All paths below are relative to the upstream repo root. They are byte-for-byte
copies **except** for the import rewrites noted in the last column, so a future
reader can diff them directly against upstream.

| Upstream path | Vendored path | Change |
| --- | --- | --- |
| `ts_benchmark/baselines/duet/student_t_standalone.py` | `duet/student_t_standalone.py` | verbatim |
| `ts_benchmark/baselines/duet/models/duet_prob_model.py` | `duet/models/duet_prob_model.py` | 4 absolute `ts_benchmark.…` imports → relative |
| `ts_benchmark/baselines/duet/utils/masked_attention.py` | `duet/utils/masked_attention.py` | 1 absolute `ts_benchmark.…` import → relative |
| `ts_benchmark/baselines/duet/layers/linear_extractor_cluster.py` | `duet/layers/linear_extractor_cluster.py` | verbatim |
| `ts_benchmark/baselines/duet/layers/distributional_router_encoder.py` | `duet/layers/distributional_router_encoder.py` | verbatim |
| `ts_benchmark/baselines/duet/layers/expert_factory.py` | `duet/layers/expert_factory.py` | verbatim |
| `ts_benchmark/baselines/duet/layers/linear_pattern_extractor.py` | `duet/layers/linear_pattern_extractor.py` | verbatim |
| `ts_benchmark/baselines/duet/layers/Autoformer_EncDec.py` | `duet/layers/Autoformer_EncDec.py` | verbatim |
| `ts_benchmark/baselines/duet/layers/RevIN.py` | `duet/layers/RevIN.py` | verbatim |
| `ts_benchmark/baselines/duet/layers/esn/reservoir_expert.py` | `duet/layers/esn/reservoir_expert.py` | verbatim |
| `ts_benchmark/baselines/duet/duet_prob.py`, lines 73–174 | `config.py` (`TransformerConfig`) | class body verbatim, extracted |

Package `__init__.py` files (`duet/`, `duet/layers/`, `duet/layers/esn/`,
`duet/models/`, `duet/utils/`) are **empty**. Upstream's
`ts_benchmark/baselines/duet/__init__.py` contained

```python
from ts_benchmark.baselines.duet.duet import DUET
```

which was emptied — see below.

New, hand-written files (not from upstream): `__init__.py`, `api.py`,
`checkpoint.py`, `PROVENANCE.md`, and the module docstring at the top of
`config.py`.

The directory layout mirrors upstream's `ts_benchmark/baselines/duet/` subtree so
individual files can still be diffed against the source. Model class names and the
internal module structure were **not** changed.

## What was stripped, and why

* **`ts_benchmark/baselines/duet/duet.py`** (22 KB) — the deterministic DUET
  trainer. Reached only via the side-effect import in `duet/__init__.py`. Imports
  `sklearn.preprocessing.StandardScaler` and `torch.utils.tensorboard`, and drags in
  `ts_benchmark/models/{model_base,model_loader}.py`,
  `ts_benchmark/baselines/utils.py`, `ts_benchmark/utils/data_processing.py`,
  `duet/models/duet_model.py`, `duet/utils/{tools,timefeatures,window_search}.py`.
  None of it is touched at inference time. Emptying `duet/__init__.py` removes the
  whole subtree.
* **`ts_benchmark/baselines/duet/duet_prob.py`** (68 KB) — the probabilistic
  trainer / Optuna objective. Only its `TransformerConfig` class (a dependency-free
  kwargs+defaults holder) is needed, so that class was extracted into `config.py`
  and the file was dropped. At module scope it imports `optuna`, `tensorboard`,
  `matplotlib`, `PIL` and `tqdm`.
* **`ts_benchmark/baselines/duet/layers/{Embed,Transformer_EncDec,SelfAttention_Family}.py`**,
  `utils/masking.py` — not in the inference import graph.
  `SelfAttention_Family.py` additionally requires `reformer_pytorch`, which is not
  installed anywhere in this repo.
* Everything else in `ts_benchmark/` (`data/`, `evaluation/`, `report/`,
  `utils/parallel/`, `pipeline.py`, `recording.py`, …) — benchmark harness, never
  imported by the forecast.
* All top-level scripts, notebooks, EDA output, `nohup.out`, and the 6 MB
  `mein_korrekter_timeseries_report.html` in the upstream repo root.

**Resulting runtime dependencies: `torch`, `numpy`, `pandas`, `einops`, `scipy`**
(verified empirically with a `sys.modules` diff). `requests` is imported lazily,
only inside `checkpoint._download`, when the checkpoint has to be fetched.
Dropped from the inference path: `scikit-learn`, `optuna`, `tensorboard`,
`matplotlib`, `PIL`, `tqdm`.

## Deliberate behavioural changes vs `run_single_forecast.py`

The numerical output is unchanged — see `tests/test_model_vendored.py`, which runs
the original script as a subprocess and the new in-process path on identical input
and asserts the results are bit-identical. Only the plumbing differs:

1. **No subprocess, no CSV round-trip.** `eisbach.model.load_model()` /
   `eisbach.model.forecast()` return a `pandas.DataFrame` directly.
2. **`load_state_dict(..., strict=True)` is now explicit.** The pinned script relied
   on the implicit default; upstream HEAD downgraded it to `strict=False`, which
   would silently accept a mismatched checkpoint. Pinning it in source removes the
   possibility. Verified: zero missing, zero unexpected keys.
3. **No `distribution_family` override.** Upstream HEAD forces `'ZIEGPD_M1'` over
   the `student_t` the checkpoint was trained with; the pinned script does not.
   The override is absent here by construction and must never be reintroduced.
4. **Errors raise.** The original caught every exception during model loading,
   printed `❌ ERROR: …` and `return`ed, so the process exited 0 and produced no
   CSV — a failed forecast looked like a successful run. All failure paths here
   raise.
5. **An un-inferable input frequency raises** instead of silently falling back to a
   daily forecast index (`pd.date_range(..., freq=None)` defaults to `'D'`).
6. **The checkpoint is hash-verified** on every load (`checkpoint.py`), and
   downloaded on demand from the pinned commit if it is not cached.

The output contract is unchanged and is asserted in the tests:

* `SERIES_ORDER = ['wassertemp', 'airtemp_96', 'pressure_96']` (training order)
* `QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]`
* flat column names `f"{var}_q{q}"`, variable-major
* index: `config.horizon` (= 96) timestamps starting one step after the last input
  timestamp, at the frequency inferred from the input index
