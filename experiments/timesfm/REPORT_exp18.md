# Experiment 18 — discharge without weather

Run on Apple Silicon, MLX, 14 September 2026. Starting revision: `0a668e9`,
`origin/claude/sleepy-brown-950428`; work isolated on `local/exp18`.

## Result

**Yes: the recorded Bad Tölz discharge adds useful predictive information for this
TimesFM configuration without weather, particularly alongside upstream water
temperature at a year of context.** Adding Q to T reduces MAE by **3.01%** and CRPS
by **2.88%**. The monthly block intervals exclude zero, the quarterly sensitivity
agrees, and the gain survives removing 2026 and restricting missing inputs.

With oracle air weather already present, the same Q addition gives **+0.26% MAE** and
**−0.05% CRPS**, with intervals crossing zero and extending outside ±1%. Thus the
observed benefit is absent in the weather-conditioned point estimates, and the
direct interaction supports attenuation. This is evidence consistent with redundancy;
**an exactly zero or certainly negligible effect with weather is not established**.
The standalone Q result is weaker than its increment alongside T.

The supplied product does not earn an extra input slot here. At full context,
**F alone is 5.81% worse in MAE than the two raw series T+Q**. Adding F to T+Q
changes MAE by only −0.27% on the 100-window context check, with an interval crossing
zero. No conclusion about TimesFM being unable to calculate products follows.

## Losses and paired effects

`W` = the two oracle air-temperature covariates; `F` = `T × Q`.
The following pooled scores are the nine stage-3 variants, all on the same 250 windows
at 8,760 h context. They cover the entire 96-hour forecast.


| Variant | MAE °C | CRPS °C |
| --- | --- | --- |
| W+T | 0.3602 | 0.2360 |
| W+T+Q | 0.3612 | 0.2359 |
| W+Q | 0.3676 | 0.2403 |
| W | 0.3696 | 0.2424 |
| T+Q | 0.6313 | 0.4139 |
| T | 0.6508 | 0.4262 |
| Q | 0.6675 | 0.4346 |
| F | 0.6679 | 0.4351 |
| Eisbach | 0.6796 | 0.4428 |

The table below reports each primary pair at its last ladder stage. Intervals are
monthly cluster-bootstrap intervals; scores are equal-window paired means. The
[paired appendix](REPORT_exp18_tables.md) contains **every stage**, including the
original IID intervals. Do not compare different stages as if they shared a context
and sample.

| Pair | Stage | n | MAE °C | MAE Δ% [95% CI] | CRPS °C | CRPS Δ% [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| T+Q vs T | 3 | 250 | 0.6508 → 0.6312 | -3.01% [-4.97, -1.07] | 0.4262 → 0.4139 | -2.88% [-4.69, -1.07] |
| Q vs Eisbach | 3 | 250 | 0.6796 → 0.6675 | -1.78% [-3.45, -0.11] | 0.4428 → 0.4345 | -1.87% [-3.50, -0.28] |
| F vs T | 3 | 250 | 0.6508 → 0.6679 | +2.63% [+0.66, +4.65] | 0.4262 → 0.4351 | +2.09% [+0.28, +3.93] |
| T+Q+F vs T+Q | 2 | 250 | 0.7008 → 0.7008 | +0.00% [-0.69, +0.65] | 0.4623 → 0.4619 | -0.07% [-0.68, +0.52] |
| W+T+Q vs W+T | 3 | 250 | 0.3603 → 0.3612 | +0.26% [-1.60, +2.06] | 0.2360 → 0.2359 | -0.05% [-1.85, +1.68] |
| W+Q vs W | 3 | 250 | 0.3697 → 0.3676 | -0.55% [-2.30, +1.07] | 0.2424 → 0.2404 | -0.84% [-2.57, +0.80] |
| W+F vs W+T | 1 | 100 | 0.4459 → 0.4596 | +3.07% [+0.57, +5.67] | 0.2915 → 0.3007 | +3.17% [+0.82, +5.56] |

For T+Q vs T, the three-month block intervals are −5.12% to −0.83% (MAE)
and −4.87% to −0.81% (CRPS). The gain persists under both block choices. Q alone versus Eisbach is a smaller gain: −1.78% / −1.87%, and its
MAE interval includes zero after excluding 2026.

The direct interaction compares the Q loss increment with and without weather on
identical windows. Positive values mean the benefit is smaller with weather. The
relative version accounts for the different baseline loss scales; percentage-point
intervals retain the observed reference denominators.

| Bare Q comparison | Metric | Attenuation °C [95% CI] | Attenuation pp [95% CI] |
| --- | --- | --- | --- |
| T+Q vs T | MAE | +0.0205 [+0.0086, +0.0326] | +3.26 [+1.15, +5.41] |
| Q vs Eisbach | MAE | +0.0101 [-0.0010, +0.0208] | +1.23 [-0.69, +3.10] |
| T+Q vs T | CRPS | +0.0122 [+0.0052, +0.0191] | +2.83 [+1.00, +4.73] |
| Q vs Eisbach | CRPS | +0.0062 [-0.0007, +0.0130] | +1.03 [-0.88, +2.90] |

The interaction alongside T is resolved under these descriptive block intervals;
the interaction for Q alone is not. No statement about operational weather-forecast
errors follows from this oracle interaction.

## Which stage answered which question?

The table records the supplied **IID budget rule**, requiring both MAE and CRPS to
stop. “Equivalent” refers only to the chosen ±1% margin for that context and sample.

| Pair | Last stage | MAE / CRPS stopping classification |
| --- | --- | --- |
| T+Q vs T | 3 | helps / helps |
| Q vs Eisbach | 3 | helps / helps |
| F vs T | 3 | harms / harms |
| T+Q+F vs T+Q | 2 | within ±1% / within ±1% |
| W+T+Q vs W+T | 3 | unresolved / unresolved |
| W+Q vs W | 3 | unresolved / unresolved |
| W+F vs W+T | 1 | harms / harms |

Only W+F vs W+T stopped at stage 1; F added to T+Q stopped at stage 2.
**Five comparisons needed the full ladder, and two weather-conditioned Q comparisons
remain unresolved even there.** Stage 1 therefore did not make the rest unnecessary.
It cost about 53 s for all eleven variants; stage 2 about 205 s, and stage 3 about
14.5 minutes including reporting on this machine.

A short-context stopping classification does not settle a year-context question.
The bounded context audit ran the two omitted variants on the same 100 stage-1
anchors at 8,760 h, reusing matching stage-3 reference scores. It added no windows
and no feature constructions. The full-context raw-product comparison is included
below for scale.

| Pair | Stage | n | MAE °C | MAE Δ% [95% CI] | CRPS °C | CRPS Δ% [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| F vs T+Q | 3 | 250 | 0.6312 → 0.6679 | +5.81% [+2.86, +8.91] | 0.4139 → 0.4351 | +5.12% [+2.39, +7.91] |
| T+Q+F vs T+Q | context check | 100 | 0.6265 → 0.6248 | -0.27% [-1.07, +0.48] | 0.4086 → 0.4074 | -0.29% [-0.96, +0.34] |
| W+F vs W+T | context check | 100 | 0.3557 → 0.3583 | +0.74% [-2.06, +3.51] | 0.2332 → 0.2349 | +0.73% [-2.12, +3.65] |
| W+F vs W+T+Q | context check | 100 | 0.3578 → 0.3583 | +0.13% [-2.99, +3.22] | 0.2341 → 0.2349 | +0.34% [-2.68, +3.47] |

The stage-1 harm classification for W+F is **not established at full context**:
its MAE difference against W+T is +0.74%, with a wide interval. The raw-product
addition is small at full context, but its monthly MAE interval narrowly crosses
−1%, so full-context equivalence is also not claimed. These are honest limits of
cheap stopping, not reasons to keep extending a null experiment indefinitely.
Future experiments can start with a small screen, but should include a small
context check before treating an early answer as a full-context decision.

## Movement regimes and lead time

Movement uses exp5's observed swing: mean temperature in forecast hours 73–96 minus
the preceding 24 observed hours. This is a retrospective evaluation split, not a
regime that was known at forecast issue time.

The largest-swing quartile contains **63 windows in 39 occupied months**, with
|swing| ≥ 1.491 °C; 20 adjacent pairs have overlapping 96-hour horizons. A 63-window
IID interval is not evidence from 63 independent events. The monthly-block interval
is a useful exploratory sensitivity, but only 39 occupied clusters, seasonal
selection and cross-month dependence limit its certainty. The quartile is fixed
from the shared truth and applied identically to every variant.

| Pair | Stage | n | MAE °C | MAE Δ% [95% CI] | CRPS °C | CRPS Δ% [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| T+Q vs T | 3, movers | 63 | 1.0199 → 0.9521 | -6.65% [-9.92, -3.44] | 0.6631 → 0.6185 | -6.73% [-10.06, -3.48] |
| W+T+Q vs W+T | 3, movers | 63 | 0.4475 → 0.4490 | +0.33% [-2.40, +3.16] | 0.2940 → 0.2936 | -0.14% [-2.72, +2.36] |

| Observed regime | n | T+Q vs T: MAE % [CI] | T+Q vs T: CRPS % [CI] | W+T+Q vs W+T: MAE % [CI] | W+T+Q vs W+T: CRPS % [CI] |
| --- | --- | --- | --- | --- | --- |
| strong cooling (< -1.5 °C) | 29 | -4.35% [-9.36, +0.14] | -4.32% [-9.35, +0.03] | -0.54% [-3.56, +2.83] | -0.58% [-3.22, +2.15] |
| cooling (-1.5..-0.5) | 51 | -1.99% [-5.48, +1.61] | -1.47% [-4.55, +1.61] | +1.01% [-2.16, +4.36] | +0.27% [-2.83, +3.49] |
| plateau (±0.5) | 92 | +0.15% [-2.92, +3.12] | -0.20% [-2.69, +2.18] | +1.35% [-1.42, +4.20] | +0.90% [-1.91, +3.80] |
| warming (0.5..1.5) | 44 | -0.35% [-5.96, +4.75] | +0.50% [-3.93, +4.74] | -3.27% [-7.50, +0.52] | -2.36% [-6.01, +0.99] |
| strong warming (> 1.5 °C) | 34 | -9.31% [-12.98, -4.96] | -9.50% [-13.34, -5.35] | +1.04% [-3.03, +5.76] | +0.23% [-3.66, +4.36] |

The gain is concentrated in movement, most clearly **strong warming**, not in
plateaus. With weather, no corresponding regime-specific Q benefit is resolved by
these intervals. The regime comparisons are exploratory and multiply the number of
looks; they identify where this dataset's gain occurs, not a validated operating rule.

The full-context T+Q increment is also visible early: first 6 h MAE −3.29%
[−6.05, −0.62], CRPS −3.48% [−5.82, −1.06]; first 24 h MAE −4.68%
[−7.53, −1.87], CRPS −4.70% [−7.18, −2.12]. Thus the pooled result is not
solely a late-horizon effect.

## Sensitivity to data quality

At full context, requiring Q missingness ≤5% retains 249 windows: T+Q vs T MAE
−2.94% [−4.92, −1.00]. Excluding all of 2026 retains 218 windows: −2.87%
[−5.14, −0.84]. Q alone is less robust: without 2026 its MAE effect is −1.53%
[−3.48, +0.37]. W+T+Q vs W+T remains unresolved without 2026 at +0.49%
[−1.63, +2.54].

At short context, the same ≤5% Q missingness restriction retains only 237 windows;
T+Q vs T is still unresolved at −0.25% [−2.19, +1.61]. The contrast between
short and long context is therefore not removed by this simple missingness check.
These sensitivities exclude observations rather than fitting a cleaning rule to
favour Q; none modifies the source data or the primary comparison.


## Question and limits

This tests whether **this zero-shot TimesFM configuration benefits from the recorded
Bad Tölz discharge history**. It cannot prove that discharge carries no information
about the Eisbach, establish causation, or demonstrate what the network can never
compute. A forced physical yes/no would overstate what a forecasting ablation measures.

`T` is `isar_toelz` (water temperature at Bad Tölz B472); `Q` is `q_toelz_kw`
(discharge at Bad Tölz KW). `F` below means the supplied `T × Q` product.
`W` consists only of `airtemp` and `t_catchment`, with **observed future weather**.
It is an oracle experiment. It does not test the complete proposed weather set with
rain and solar radiation or the errors of operational weather forecasts.
The inherited loader averages four nearly identical southern station responses to
form `t_catchment`; it does not literally select a single column as STATUS.md suggests.

## Corrections to the supplied experiment

- The imported evaluator called `np.stack([])` when `future=[]`. It now passes `None`,
  as the backend expects. No weather-enabled input changes.
- `anchors[::2][:100]` sampled only the early 200 of the 250 anchors. The screen now
  spans the entire ordered set using 100 evenly spaced indices, retaining both ends.
  The 250-window stages contain all 100 screen anchors.
- Percent interval denominators now match the equal-window reference mean used by
  the paired estimator, rather than the pooled hour-weighted mean. This does not
  change which questions advance on these data.
- An unresolved interval is uncertainty, not evidence that the true effect is both
  small and larger than 1%. The terminal message was corrected accordingly.
- The original IID stopping rule is retained as an **exploratory compute-budget rule**.
  Calendar-block intervals and a small full-context check of screened-out variants
  are reported separately. Short context and long context are different estimands.

## Evaluation and intervals

The primary ladder is 100/1,024 h, 250/1,024 h, then 250/8,760 h, with a 96-hour
forecast horizon. Pooled headline losses cover **all 96 hours**, not just the first
day. The seven stored lead buckets are 1–6, 7–12, 13–18, 19–24, 25–48, 49–72,
and 73–96 hours. No observed discharge or upstream temperature after the anchor is
passed to inference. Missing inputs are interpolated within the available input;
scoring truth is never filled.

MAE scores the median. The reported “CRPS” is the harness's trapezoidal integral of
pinball losses over deciles 0.1–0.9: a consistently defined **truncated quantile
score**, not an integral over the entire predictive distribution. Both are in °C;
negative candidate-minus-reference differences mean improvement.

Absolute pooled tables weight observed forecast hours. Paired tables first form a
weighted loss within each window, then give every paired window equal weight.
Intervals use 8,000 paired bootstrap draws. IID intervals reproduce the supplied
ladder; the main dependence sensitivity resamples occupied calendar months as whole
clusters and divides resampled loss-difference sums by resampled window counts.
Three-month calendar blocks give a second sensitivity. Relative intervals divide
absolute loss-difference intervals by the observed paired reference mean; uncertainty
in that denominator is not resampled.

There are 250 anchors from 2019-12-31 23:00 UTC through 2026-09-09 23:00 UTC, in
70 occupied months. Counts by year are 1, 43, 45, 44, 43, 21, 21, 32 for 2019–2026.
110 adjacent anchor pairs are less than 96 hours apart; the minimum spacing is 13 h
and the median is 338 h. Calendar clusters reduce pseudoreplication but do not prove
independence across month boundaries or across shared long contexts. All intervals
are descriptive: they are not adjusted for seven comparisons, repeated looks,
context selection, or retrospectively selected movement regimes.

The ±1% margin is retained as a practical screening choice, not a scientifically
derived equivalence margin. An interval inside it is an exploratory equivalence
finding under that design, not “certain irrelevance”. Excluding zero also does not
by itself establish practical usefulness. No extra windows are added to chase a
p-value.

## Discharge quality and physical interpretation

The source series spans 2018-12-31 through 2026-09-14: 66,069 finite hourly Q values,
1,456 missing, with a longest missing run of 984 h. The observed range is
9.12–329.50 m³/s, median 20.375; the 99th percentile is 134.83. Median absolute
hourly change is 0.175 m³/s; the 99th percentile is 6.625. There are 334 changes
larger than 10 m³/s and 75 larger than 20; 14.67% of observed hourly changes are zero.

The series is not simply unrelated turbine noise: its level correlations with
Lenggries and Puppling are 0.975 and 0.942, and hourly-difference correlations are
0.484 and 0.246. Inspection of the daily record and the June 2024 flood shows shared
large-scale variation, but also flat stretches and abrupt local jumps at Bad Tölz KW.
There is a particularly suspicious isolated observation at **2026-01-25 22:00 UTC**:
Q goes 10.425 → 140.650 → 10.100 m³/s. Lenggries and Puppling stay near 6–8 and
14 m³/s. This is not convincing evidence of a natural flood wave; the available
record cannot distinguish a measurement/ingestion problem from plant-related behavior.
The primary experiment retains the original observations. Excluding 2026 is a
sensitivity, not a claim to have repaired the sensor.

The LfU explicitly warns that the discharge determination is influenced by the
power station. That supports caution about the measurement process; it does not
establish that all jumps are turbine scheduling or that managed flows are
unpredictable. [LfU Bad Tölz KW discharge](https://www.gkd.bayern.de/de/fluesse/abfluss/isar/bad-toelz-kw-16004006/monatswerte?addhr=hr_hw_so).

T and Q are **not measured at the same point**. The published UTM coordinates put
B472 and KW about 3.0 km apart in straight-line distance. Multiplying synchronous
records does not account for that separation or travel time.
[LfU B472 metadata](https://www.gkd.bayern.de/de/fluesse/wassertemperatur/isar/bad-toelz-b472-16003207),
[LfU KW metadata](https://www.gkd.bayern.de/de/fluesse/abfluss/isar/bad-toelz-kw-16004006).

`Q × T` has units m³·°C/s, **not watts**. A heat-transport estimate relative to a
reference temperature would be `rho * c_p * Q * (T - T_ref)` under suitable physical
assumptions. More flow at the same temperature transports more mass as well as more
energy, so it does not alone imply a warmer downstream temperature. Here the product
is a physically motivated proxy, not a measured heat flux reaching the Eisbach.
Even a benefit from explicitly supplying it would establish usefulness of that
representation for this configuration, not prove that TimesFM is incapable of
forming products internally.

The 95% annual eligibility check does not guarantee 95% completeness of a 1,024-hour
slice: the worst screen Q window has **33.59% missing inputs**, versus 5.05% at full
context. There are no all-missing target/T/Q windows. The worst target missing
fractions are 3.91% and 0.91%; for T they are 5.96% and 1.10%. A separate score
sensitivity restricts each context to Q windows with at most 5% missing data.

## Fairness of the bare comparison

After repairing the empty weather list, the intended comparison is fair with respect
to input timing and target preprocessing. The target is always passed as one series;
`prepare_inputs` finds its first valid timestamp and trims **all** accompanying
series by that same amount. Adding a past-only column does not trim the target at the
column's own first valid value. In the actual 250 selected windows there are no
leading target gaps at either context length, although T and Q have some leading gaps.

The installed MLX forecaster has a batch fast path for the one-dimensional bare target
and a per-window general path for covariates. Both call the same `model.decode`.
The general path truncates target and covariates together; both tested context lengths
are within the cap. The decoder applies the same left padding, per-variate detrending,
and horizon construction. Target and past-only future values are masked and marked
as predicted variates; known-future weather alone supplies observed horizon values.
The extra variate attention is the intended intervention, not an alignment artifact.

A numerical audit compares the bare fast path with a one-row 2D target that forces the
general path, and compares past-only T/Q with the identical prepared arrays stacked as
targets, both with and without weather. It uses early, middle, late and largest-target-gap
windows at both context lengths and records model source hashes. The synthetic
regression test additionally forces different leading gaps in the target and Q and
checks that no future Q values enter inference.


All **24 numerical comparisons passed** at tolerance 1e-4 °C. The largest fast/general
quantile difference was **1.90735e-5 °C**; prepared past-only versus target-stack
comparisons were **bit-identical** with and without weather. These checks support
preprocessing/decode fairness on this MLX version; they are not a new cross-backend
validation against PyTorch.

## Reproduction and verification

The source data and checkpoints remain in the ignored `data/experiments/` cache.
The scored data-frame hash in the stage manifests is
`ef7345e26aa96e8fffbc15fe2a7bb657658c6e1b3bc22d531efdd1a042814893`.
Environment: macOS 26.6.2 arm64; Python 3.12; timesfm 3.0.2, mlx 0.32.2,
numpy 2.5.3, pandas 3.0.5. The cached model revision is
`43046b85ec22d584a13f8098c2ed39c889e129c2`.

From the repository root, with the existing local data/model cache:

```sh
export TIMESFM_BACKEND=mlx
export HF_HOME="$PWD/data/experiments/hf"
export HF_HUB_OFFLINE=1
.venv-exp11/bin/python -u experiments/timesfm/exp18_discharge_alone.py
.venv-exp11/bin/python -u experiments/timesfm/audit_exp18.py --confirm
.venv-exp11/bin/python experiments/timesfm/report_exp18.py
```

MLX requires Metal access outside this session's default sandbox. The initial
sandbox import failed because no Metal device was exposed; the actual experiment
and numerical audit ran successfully with that access enabled.

The ladder was subsequently resumed with an inference object that raises on any
prediction call: all stages were recovered from provenance-bound, finite, complete
checkpoints without inference. This also verified that the final denominator and
sampling code retains the same stopping decisions. Original timing log:
`experiments/timesfm/exp18.log`; final-code resume log: `exp18_resume_verified.log`;
numerical/context audit log: `exp18_audit.log`.

`report_exp18.py` regenerates the paired appendix and the more extensive local
`data/experiments/exp18_tables.md` / `exp18_summary.json`, including quarterly blocks,
all stage-specific regimes, leave-one-year-out effects and data-quality sensitivities.
`exp18_decoder_audit.json` records the checked source hashes and differences;
`exp18_data_audit.json` records input quality. Generated CSV/PNG and logs are not
committed.

Validation: **192 production tests passed, 2 skipped**; **21 targeted experiment tests
passed**; `ruff check .` passed. The skips and a PyTorch deprecation warning are
reported by the existing production suite. Production source, `main.py`, `tests/`,
and `data/archive/` were not edited. Changes are confined to experiment code,
regression tests, this report/appendix and the corrected discharge interpretation in
`STATUS.md`.
