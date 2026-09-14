# exp17: decision protocol

Recorded before inspecting forecast scores, 2026-09-14. Base commit: `7d90c8a`.

The fixed configuration is TimesFM 3.0 / MLX, hourly, target `eisbach`, horizon
96 hours, final context 8760 hours. Known-future columns are `airtemp`,
`t_catchment`, `rain_toelz`, `rain_lenggries`, `rain_kochel`, `rain_garmisch`,
`solar_hohenpeissenberg`; `isar_toelz` is always past-only. The existing
`t_catchment` loader is retained for compatibility with previous experiments.
The only selectable inputs are the past-only water temperatures
`isar_lenggries`, `isar_puppling`, `loisach_beuerberg`.

## Method and budget

Use a paired, exhaustive subset ablation: eight small-context arms (1024 hours),
including the fixed baseline. This directly tests the decision to include a
series and reveals pair/triple interactions. No fitting of additional models,
no substitution for Bad Tölz, no covariate-order experiments.

AutoGluon's time-series feature importance offers permutation and constant
replacement. We use removal because permutations of strongly correlated,
autocorrelated water temperatures can create unrealistic histories, and feature
reliance is not identical to the benefit of adding an input to this zero-shot
forecaster. Eight actual subsets are affordable here; general SHAP approximation
or another forecasting framework adds no necessary information for three inputs.
Source: [AutoGluon TimeSeriesPredictor.feature_importance](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.feature_importance.html).

Start with one eligible origin per calendar month, the middle eligible date,
selected without reference to model scores. Require >=95% observed history for
all input series over 8760 hours and observed first/last history values; all
96 target and future weather hours must be observed. Never require future
candidate water values. Remove origins within 96 hours of their predecessor.
Use exactly these origins across contexts and arms. Historical interpolation
is confined to the supplied history for target/past-only series; scoring truth
is never filled. This is an oracle-weather experiment, not a forecast replay.

Primary metric: all-96-hour MAE, equally weighted observed hours. Secondary:
shared-decile CRPS; seven lead buckets; calendar-year effects. Pair all
comparisons by origin. Report 95% descriptive percentile intervals from 10,000
paired circular moving-block bootstrap draws, block length three monthly
origins, with lengths one and six as sensitivity checks. These are not
selection-adjusted significance tests and do not establish independent
out-of-sample validation or causal importance.

Use 1% relative MAE as a practical near-zero region for deciding whether more
computation is worthwhile, not as a significance requirement. A narrow interval
inside [-1%, +1%] is sufficient to stop investigating that increment. Confirm
plausibly useful subsets at 8760 hours, including the baseline and the nested
comparisons needed to decide each retained feature. Escalate an unclear candidate
only if its interval still permits a practically useful gain. A negative trend
can justify retention without significance; demonstrated harm, including a
convincing annual reversal, argues against retention. Prefer the smaller set
when its extra inputs have only negligible measured increments. Do not search
new configurations or keep adding origins merely to obtain significance.

Limitations: short-context rankings can fail to transfer; one origin per month
samples events sparsely; intervals rely on limited observed years and a chosen
block length; common coverage excludes missing-data regimes; the pretrained
model's exposure to these historical data is unknown; weather forecasts may
change the incremental utility. Selection and confirmation on the same dates
are a context check, not a fresh holdout.

## One targeted follow-up after the first full-context results

The first full-context comparison left Puppling beside Beuerberg at -0.24% MAE,
95% block interval [-2.33%, +1.87%]. This is not narrow enough to call equivalent.
Therefore test only Beuerberg versus Beuerberg + Puppling at 8760 hours on up to
two previously unscored origins per eligible calendar month: first and last
eligible daily date, >=96 hours from every original and supplemental origin.
Keep the same eligibility and data. Average these new origins within each month
before the monthly block bootstrap. This is a bounded independent-origin check,
not an independent-year holdout; no further origin expansion is planned. Judge
its direction and practical magnitude together with the original comparison,
and report residual uncertainty instead of chasing a significance threshold.
