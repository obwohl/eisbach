# exp17 — incremental water temperatures beside the fixed set

2026-09-14 · `obwohl/eisbach` · base `7d90c8a` · branch `local/exp17`

## Decision

**Keep `loisach_beuerberg`; drop `isar_lenggries` and `isar_puppling`.**
Beuerberg improves full-context MAE by **2.83%** beside the fixed set
(95% block interval **[-5.25%, -0.40%]**). Lenggries makes both the fixed set
and the leading pair worse. Puppling's tiny benefit beside Beuerberg fails on
68 new origins: **+1.55% MAE**, with a 95% interval **[-0.03%, +3.00%]**.
This is a selection decision under the existing favorable-trend rule, not a
claim that all rejected inputs are statistically proven harmful everywhere.

Final past-only inputs: **`isar_toelz`, `loisach_beuerberg`**. The seven
known-future inputs, hourly grid, 8760-hour context and 96-hour horizon stay fixed.

## Question and method

Only three inputs were selectable: `isar_lenggries`, `isar_puppling`, and
`loisach_beuerberg`, all **past-only water temperatures**. Every arm retained
`isar_toelz` as past-only and the seven fixed known-future inputs: `airtemp`,
`t_catchment`, `rain_toelz`, `rain_lenggries`, `rain_kochel`, `rain_garmisch`, and
`solar_hohenpeissenberg`. Target `eisbach`; hourly; 96-hour horizon; TimesFM 3.0.
No discharge, constructed feature, replacement for Bad Tölz, or input-order test.

I used **paired subset ablation**: all eight subsets at 1024 hours, then the
baseline, three singleton additions, Puppling + Beuerberg, and all three at
8760 hours. The two other pairs were not promoted: the screen favored Puppling
+ Beuerberg, and its nested comparisons plus the three-input arm directly test
whether each member contributes and whether Lenggries adds to it. This is not
an exhaustive search for the best subset at full context.

[AutoGluon's time-series feature importance](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.feature_importance.html)
uses permutation or constant replacement. Those measure reliance on perturbed
inputs; removal measures the actual deployment decision here. Permuting these
strongly correlated, autocorrelated river series can create unrealistic
histories. With three candidates, actual subset forecasts are inexpensive and
expose conditional effects without fitting a surrogate or approximating SHAP.
This is predictive utility for this forecaster, not causal hydrological importance.

The [protocol](PROTOCOL_exp17.md) was written before inspecting forecast scores.
Primary metric is 96-hour MAE; secondary is shared-decile CRPS. Negative changes
are improvements. Intervals are **descriptive paired 95% circular moving-block
bootstrap intervals**, 10,000 draws, three chronological origins per block.
Each draw resamples both the error difference and its reference denominator.
One- and six-origin blocks check sensitivity. There is no multiplicity correction
or independent holdout: confirmation changes context on the same dates.

## Data and coverage

63 monthly origins, **2020-08-16 22:00 UTC to 2026-09-05 22:00 UTC**, minimum
spacing 456 hours. Each arm scores the same 6,048 observed target hours; horizons
do not overlap. Origins per year: 2020: 2, 2021: 4, 2022–2025: 12 each, 2026: 9.
Calendar months have five origins each except August (six) and September (seven).
The first two years are too sparse for credible annual interval claims.

An origin is the middle eligible daily candidate in its calendar month, chosen
without inspecting model errors. Eligibility requires at least 95% observed
history in every input over the full 8760 hours, observed history endpoints,
and all 96 future target/weather values observed. It never requires future
water-covariate observations. All arms, including the short screen, share this
eligibility filter. Historical gaps are filled by the existing MLX preparation
inside the supplied window; past-only series never see future water values.
Scoring truth is never filled. Implausible water readings outside [-1, 32] °C
are masked before eligibility. Common coverage excludes missing-data regimes,
especially Puppling's early gaps, so these are not all possible operating dates.

The named Hohenpeißenberg cache was missing locally and was fetched with the
existing `build_solar.fetch_station('02290', 'hohenpeissenberg', 2026)` into the
ignored `data/experiments/exp17_solar_hp.csv`. No coordinate-derived solar series
was substituted. The existing `load_all()` definition of `t_catchment` was
preserved: technically a row mean of four almost-identical coordinate results
from the same station, as documented by the prior work. This run does not
redefine that fixed input.

**All future weather is observed weather (oracle), not archived forecasts.**
The results therefore assess incremental utility conditional on this oracle
weather set. They do not establish the corresponding live-weather gain.
Other limits: sparse event sampling, only a few years, arbitrary bootstrap block
length, noncontiguous early months, and unknown pretraining exposure to this
historical record. Shared histories can still induce dependence despite
nonoverlapping horizons; the block sensitivities are not a proof of independence.

## 1024-hour context

| Added inputs | MAE (°C) | MAE change [95% CI] | CRPS change [95% CI] |
|---|---:|---:|---:|
| Fixed baseline | 0.394629 | — | — |
| + Lenggries | 0.390899 | -0.94% [-2.77, +1.07] | -1.15% [-2.93, +0.85] |
| + Puppling | 0.389663 | -1.26% [-3.62, +0.97] | -1.47% [-3.68, +0.48] |
| + Lenggries + Puppling | 0.386876 | -1.96% [-4.80, +0.80] | -2.14% [-4.72, +0.30] |
| + Beuerberg | 0.386503 | -2.06% [-5.25, +0.91] | -2.35% [-5.56, +0.60] |
| + Lenggries + Beuerberg | 0.385690 | -2.26% [-5.64, +1.19] | -2.68% [-6.07, +0.74] |
| + Puppling + Beuerberg | 0.381248 | -3.39% [-8.29, +0.76] | -3.70% [-8.45, +0.21] |
| + all three | 0.381345 | -3.37% [-8.03, +0.85] | -3.71% [-8.23, +0.25] |

## 8760-hour context

| Added inputs | MAE (°C) | MAE change [95% CI] | CRPS change [95% CI] |
|---|---:|---:|---:|
| Fixed baseline | 0.321795 | — | — |
| + Lenggries | 0.325945 | +1.29% [-0.95, +3.52] | +1.02% [-1.35, +3.29] |
| + Puppling | 0.318385 | -1.06% [-3.39, +1.28] | -1.39% [-3.60, +0.70] |
| + Beuerberg | 0.312674 | -2.83% [-5.25, -0.40] | -3.01% [-5.19, -0.89] |
| + Puppling + Beuerberg | 0.311913 | -3.07% [-6.71, +0.34] | -3.61% [-7.09, -0.56] |
| + all three | 0.316367 | -1.69% [-5.65, +1.97] | -2.30% [-6.37, +1.21] |

## Conditional additions at 8760 hours

| Addition | MAE change [95% CI] | CRPS change [95% CI] |
|---|---:|---:|
| Puppling beside Beuerberg | -0.24% [-2.33, +1.87] | -0.62% [-2.64, +1.32] |
| Beuerberg beside Puppling | -2.03% [-4.01, -0.06] | -2.26% [-4.08, -0.51] |
| Lenggries beside Puppling + Beuerberg | +1.43% [-0.20, +3.03] | +1.36% [-0.16, +2.92] |

## Annual MAE checks at 8760 hours

Three-origin block intervals within each year; fewer than eight origins: no interval.

| Year | Origins | Lenggries vs base | Puppling vs base | Beuerberg vs base | Puppling + Beuerberg vs base | All three vs pair |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 2 | +6.00% | +21.67% | -1.10% | +16.15% | +1.30% |
| 2021 | 4 | +1.36% | +0.32% | +0.64% | +1.78% | -0.14% |
| 2022 | 12 | +2.94% [+0.32, +7.43] | +1.05% [-1.82, +4.88] | -1.87% [-4.30, +1.15] | +0.37% [-2.65, +6.25] | +1.73% [+0.07, +4.07] |
| 2023 | 12 | +3.15% [-2.40, +6.51] | -1.14% [-4.22, +1.26] | -2.09% [-8.11, +5.37] | -2.49% [-6.03, +2.60] | +1.24% [-2.62, +4.58] |
| 2024 | 12 | -3.26% [-7.17, +2.80] | -5.73% [-10.16, +1.61] | -5.66% [-12.53, +3.07] | -9.62% [-18.52, +3.80] | +0.28% [-3.67, +4.09] |
| 2025 | 12 | -1.16% [-6.77, +5.21] | -1.82% [-6.65, +1.74] | -4.88% [-8.87, -1.26] | -5.13% [-11.29, -0.22] | +0.15% [-5.35, +6.29] |
| 2026 | 9 | +4.25% [-1.02, +8.64] | -1.89% [-5.65, +3.41] | -0.83% [-5.50, +4.29] | -3.32% [-9.36, +2.59] | +4.22% [+0.54, +7.24] |

## Lead buckets at 8760 hours

| Hours | Lenggries MAE change | Puppling MAE change | Beuerberg MAE change | Puppling + Beuerberg MAE change |
|---|---:|---:|---:|---:|
| 1–6 | +0.72% [-0.73, +2.15] | +0.49% [-2.18, +3.56] | -1.12% [-3.42, +1.28] | -0.12% [-3.46, +3.91] |
| 7–12 | +0.22% [-2.76, +3.03] | -3.36% [-6.05, -0.97] | -2.93% [-6.10, +0.15] | -4.36% [-7.71, -1.03] |
| 13–18 | +1.77% [-1.05, +4.73] | -5.46% [-8.99, -2.36] | -5.52% [-9.27, -2.16] | -8.74% [-13.90, -4.22] |
| 19–24 | +2.05% [-0.69, +5.63] | -4.16% [-8.05, +0.24] | -3.97% [-7.27, +0.00] | -7.14% [-12.44, -0.95] |
| 25–48 | +1.14% [-1.64, +3.88] | -1.31% [-4.27, +1.68] | -5.09% [-7.27, -2.74] | -4.83% [-8.70, -0.78] |
| 49–72 | +1.87% [-1.25, +5.29] | -0.64% [-3.78, +2.66] | -1.53% [-5.68, +2.74] | -1.62% [-7.61, +4.20] |
| 73–96 | +0.88% [-2.17, +4.15] | +0.19% [-3.09, +3.73] | -1.60% [-4.59, +1.33] | -1.36% [-5.65, +3.06] |

## Block-length sensitivity at 8760 hours

| Block origins | Lenggries vs base | Puppling vs base | Beuerberg vs base | Pair vs base | Lenggries vs pair |
|---|---:|---:|---:|---:|---:|
| 1 | +1.29% [-1.14, +3.75] | -1.06% [-3.05, +1.07] | -2.83% [-5.11, -0.51] | -3.07% [-5.93, -0.04] | +1.43% [-0.41, +3.25] |
| 3 | +1.29% [-0.95, +3.52] | -1.06% [-3.39, +1.28] | -2.83% [-5.25, -0.40] | -3.07% [-6.71, +0.34] | +1.43% [-0.20, +3.03] |
| 6 | +1.29% [-0.88, +3.35] | -1.06% [-3.71, +1.26] | -2.83% [-5.41, -0.43] | -3.07% [-7.20, +0.37] | +1.43% [-0.06, +2.77] |

## Leave-one-year-out MAE at 8760 hours

| Omitted year | Lenggries vs base | Puppling vs base | Beuerberg vs base | Pair vs base |
|---|---:|---:|---:|---:|
| 2020 | +1.16% | -1.69% | -2.88% | -3.60% |
| 2021 | +1.29% | -1.13% | -3.02% | -3.33% |
| 2022 | +0.80% | -1.69% | -3.12% | -4.09% |
| 2023 | +0.95% | -1.04% | -2.97% | -3.18% |
| 2024 | +2.34% | +0.01% | -2.19% | -1.57% |
| 2025 | +1.83% | -0.89% | -2.38% | -2.61% |
| 2026 | +0.68% | -0.89% | -3.24% | -3.02% |

## Interpretation of the main comparison

Beuerberg is useful beside the fixed set: -2.83% MAE, -3.01% CRPS, better in
45/63 MAE windows. Its pooled MAE interval stays below zero for all three block
lengths, and its mean benefit survives omitting any one year. None of the
well-sampled years shows demonstrated MAE harm. This is incremental value
beside Bad Tölz **and all seven settled weather inputs**.

Lenggries fails the full-context test: +1.29% MAE alone beside the fixed set;
+1.43% when added to Puppling + Beuerberg. Both metrics deteriorate, and 2022
shows harm in the annual descriptive intervals. The screen had the opposite
single-addition direction; transfer from short context is empirical guidance,
not a guarantee. No more computation was warranted for Lenggries.

Puppling alone trends favorably, but its -1.06% MAE becomes just -0.24% beside
Beuerberg; its interval [-2.33%, +1.87%] is **not** evidence of equivalence.
The early lead buckets indicate useful signal, but the primary decision concerns
all 96 hours. A targeted follow-up tests whether this small conditional gain
survives different forecast origins.

## Targeted Puppling follow-up: stop here

Only the unresolved increment was escalated: Beuerberg versus Beuerberg +
Puppling, both at 8760 hours. There are **68 new nonoverlapping origins in all
63 original calendar months**, at least 96 hours from every originally scored
origin. Counts by year: 3, 5, 12, 12, 13, 14, 9 for 2020–2026. Dates are the
first and last eligible daily dates in each month, subject to spacing; no
forecast scores entered their selection. Average within each month before
resampling monthly blocks, so months with two new dates do not get double weight.
These are new scored dates, not independent years or a previously untouched
historical dataset; their context windows overlap the original contexts.

| Puppling added beside Beuerberg | MAE change [95% CI] | CRPS change [95% CI] |
|---|---:|---:|
| Original 63 origins | -0.24% [-2.33, +1.87] | -0.62% [-2.64, +1.32] |
| New 68 origins, monthly weighting | +1.55% [-0.03, +3.00] | +1.28% [-0.34, +2.81] |
| All 131 origins, monthly weighting | +0.64% [-0.51, +1.72] | +0.32% [-0.83, +1.36] |

On new origins the monthly-weighted MAE changes from 0.321006 to 0.325996 °C;
only 26/63 monthly averages improve. MAE worsens in every year except sparsely
sampled 2020. New-origin MAE intervals for block lengths one/three/six are
[-0.19%, +3.28%], [-0.03%, +3.00%], [+0.24%, +2.76%]. The harm-significance
label depends on block length; the unfavorable direction does not.

Pooling all origins by month leaves no favorable mean increment and puts a
1% MAE improvement outside the main interval. This is not a narrow equivalence
claim: the interval still permits harm. It is sufficient to prefer Beuerberg
alone and **stop**, rather than compute until some significance gate clears.
The secondary CRPS check points the same way. Puppling remains potentially
useful at some early leads or without Beuerberg; that is a different selection
question and does not earn a slot in this final set.

Compute: 504 short-context forecasts, 378 main full-context forecasts, then
136 targeted full-context forecasts; **1,018 forecasts total**. Inference took
about 47 seconds for the screen, 262 seconds for the six full-context arms,
and 95 seconds for the targeted follow-up (roughly seven minutes total,
excluding data preparation, bootstrap analysis and model loading).

## Reproduction and provenance

New runner: `experiments/timesfm/exp17_selection.py`; targeted follow-up:
`experiments/timesfm/exp17_puppling.py`; table renderer:
`experiments/timesfm/report_exp17.py`. The older `exp17_more_water.py` is
historical preliminary work and was not run: it omits settled weather and
includes replacements for Bad Tölz.

```bash
export TIMESFM_BACKEND=mlx
export HF_HUB_OFFLINE=1
.venv-exp11/bin/python experiments/timesfm/exp17_selection.py
.venv-exp11/bin/python experiments/timesfm/exp17_selection.py --context 8760 --masks 0 1 2 4 6 7
.venv-exp11/bin/python experiments/timesfm/exp17_puppling.py
.venv-exp11/bin/python experiments/timesfm/report_exp17.py
```

The solar cache must exist first; if missing, fetch only the fixed station:

```bash
PYTHONPATH=experiments/timesfm .venv-exp11/bin/python -c \
  'from build_solar import fetch_station, CACHE; fetch_station("02290", "hohenpeissenberg", 2026).to_csv(CACHE / "exp17_solar_hp.csv")'
```

Environment: macOS 26.6.2 arm64, Python 3.12.13, TimesFM package 3.0.2, MLX
0.32.2, NumPy 2.5.3, pandas 3.0.5. Model repository
`google/timesfm-3.0-pytorch`, pinned revision
`43046b85ec22d584a13f8098c2ed39c889e129c2`, weights SHA256
`a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da`.
The first model-loading attempt failed because the cached snapshot had no
`main` reference; no forecasts were produced. The runner uses the explicit
cached revision, and all completed inference used MLX on the Apple GPU.

Input dataframe hash (pandas hash followed by SHA256):
`798e95eea17e89fe171e775b66d648c713b45109bdbe845030252c64b4323202`.
Source-file SHA256 fingerprints:

| Source | SHA256 |
|---|---|
| stations_hourly.csv | `731d84985990ad581b27f2b512ec0e33364267a198273f9c2348ea4df8327bff` |
| long_hourly.csv | `e0b2d4f87db3ba46d7f3b587936788a68718136ab944a65cf109de2fb542da70` |
| weather_south.csv | `4095944d49a97dec6fdd908edd07d4692374d645bb9f1ac5fac7ba48b068e0ae` |
| exp17_solar_hp.csv | `ebd54666ac0b350bb95b33f72c329c6df41147a438d88c6d5efe4ec5c33d7940` |

Local ignored artifacts: `data/experiments/exp17_fixed_{1024,8760}.csv`, paired
manifests, runtime provenance, summary JSON, logs, `exp17_puppling.*`,
`exp17_puppling_summary.json`, `exp17_provenance.json`, and `exp17_tables.md`.
Exact origins are in the manifests. Re-fetching upstream data may change its
fingerprints; the runner rejects stale checkpoint provenance. No generated CSV
or PNG is included in the commit. Production code and archive are untouched.

## Validation

After completing the report and runners: `pytest -q` — **192 passed, 2 skipped**;
`ruff check .` — **all checks passed**. One third-party PyTorch FutureWarning
about `torch.jit.script` deprecation. Additional experiment checks:
`pytest -q experiments/timesfm/test_exp17_selection.py experiments/timesfm/test_bench_prepare.py experiments/timesfm/test_checkpoint.py`
— **20 passed**, including three new tests for paired bootstrap ratios,
future-water-independent eligibility, and the fixed Bad Tölz baseline.
