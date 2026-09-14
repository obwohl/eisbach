# exp15 / exp16 — covariate selection and an audited weather replay

Run date: 2026-09-14; starting commit `13a003f`; branch `local/exp15-exp16`.
Executed scoring sources are captured in `06aacf9`; the final reader docstring
corrects “empty Series” to “one-row NaT Series” with identical executable AST.
TimesFM 3.0.2, MLX, 96-hour horizon, 8760-hour confirmation context. Model cache
revision `43046b85ec22d584a13f8098c2ed39c889e129c2` (weight blob
`a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da`).

## exp15 answer: prefer the four raw hourly series

**Replace the constructed rain series with the four raw hourly stations under the
stated keep rule.** At one year of context they improve MAE by **2.518%** and
CRPS by **2.706%** against the same air-plus-Bad-Tölz reference; paired monthly-block
intervals are **[−4.209%, −0.907%]** and **[−4.306%, −1.199%]**. Against the existing
mean-plus-24h construction, their changes are only **−0.229% MAE [−1.334%, +0.846%]**
and **−0.127% CRPS [−1.145%, +0.886%]**. This supports the simpler representation
under the user's rule; it does **not** prove exact equivalence, superiority or
non-inferiority within a pre-specified margin.

The four individually accumulated stations have the lowest point scores
(**0.350869 MAE / 0.229846 CRPS °C**), versus **0.351384 / 0.230228 °C** for raw hourly
stations. That small, uncertain difference does not justify retaining a constructed
feature when removing it is the stated preference. The existing construction is
**0.352187 / 0.230519 °C**; no-rain is **0.360457 / 0.236630 °C**.

**Neither averaging nor accumulation is shown to carry a distinct necessary gain.**
Accumulating the hourly mean gives **−0.365% MAE / −0.493% CRPS**, with both intervals
crossing zero. Supplying the four raw hourly series instead of their hourly mean
gives **−0.593% / −0.620%**, also crossing zero. Applying accumulation individually
adds only a small uncertain gain to the raw stations. There is a clear rain signal
at full context, but no evidence here that the hand-built operations are needed to
extract it. This conclusion is conditional on oracle rain: real catchment rain
forecasts were not replayed.

The screen would have reached the wrong selection: raw hourly stations were
**+0.051% MAE / +0.128% CRPS** against no rain at 1024 h, then improved both metrics
at 8760 h. The hourly mean was the best screen arm but the worst of the confirmed
rain arms. Confirming all the relevant representations was essential. The annual
means are slightly adverse in 2022 and 2023, but their intervals include zero;
there is no analogous multi-window, annually demonstrated harm to the earlier
Sylvenstein finding. Leaving out any single year retains a raw-hourly MAE gain of 1.78–3.03%; the
result does not depend on one favourable year. The single 2019 window is reported
without a confidence interval.

## exp16 answer and a scoped sentence for STATUS.md

> On 737 strict-availability replay windows (May–September 2026), with Bad Tölz
> history held fixed, replacing observed Munich air with archived weather forecasts
> changes water MAE **0.316468 → 0.335661 °C** (**+0.019193 °C**, +6.065%) and decile
> CRPS **0.210305 → 0.221038 °C** (**+0.010733 °C**, +5.103%). Relative to the same
> no-weather reference, this loses **4.80% of the Munich-air oracle MAE advantage**
> and **4.12% of its CRPS advantage**. The paired 7-day block intervals for the
> lost fractions are **[−7.01%, +13.86%]** and **[−6.91%, +12.73%]**: a seasonal
> point estimate, not a precise or whole-year bound and not a replay of southern
> air or catchment rain.

The denominator matters. The no-weather reference is **0.715933 MAE / 0.470842
CRPS °C**, with exactly the same water/Bad-Tölz histories. The lost fraction is
`(replay − oracle) / (no-weather − oracle)`, not the relative increase of the
oracle score. The MAE improvement over no weather falls from approximately
**55.8% to 53.1%**; saying “the advantage falls by 6.1%” would confuse two different
quantities. About 95% of the oracle improvement survives *as a point estimate on
this sample*.

**The attractive precision of the ordinary paired interval is misleading.** It
puts the MAE increase at +3.753% to +8.533%, but the 7-day block interval is
**−6.797% to +21.011%** (absolute cost **−0.021511 to +0.066493 °C**). The 4-day and
14-day alternatives also cross zero. These bounds are not guarantees that the
forecast sometimes knows more than the oracle; they express sampling uncertainty
in a finite, dependent cohort. On the deterministic 16-window non-overlapping subset the MAE cost is **+12.053%
[−0.444%, +26.332%]**, rather than +6.065%; this is a different weighting of the
same season, not an independent replication. The data do not justify the old
wording that the cost is “small and known” across the historical record.

No single measured shrinkage for the *entire* chosen covariate set can be obtained
from this archive. The remaining tables distinguish the conditional two-air
comparison, actual Munich solar replay, and explicitly simulated southern errors.

Under the *supplied partial-replay design* with the southern oracle retained,
MAE changes **0.319075 → 0.331898 °C**, a cost of **0.012823 °C / 4.019%**; CRPS
changes **0.212707 → 0.219874 °C**, **0.007166 °C / 3.369%**. The lost fractions
of that two-air oracle advantage are **3.23% MAE / 2.78% CRPS**, with 7-day block
intervals **[−2.97%, +8.13%] / [−3.28%, +7.47%]**. These are valid *conditional*
measurements, but calling them a real forecast of both air channels would be false.
The southern series slightly worsens the oracle mean in this cohort (two-air MAE
0.319075 versus Munich-only 0.316468) while slightly helping when Munich is
replayed. That is another warning against transferring multi-year selection
percentages directly into this season's forecast-loss calculation.

## What was wrong with the supplied exp16

The supplied experiment could not support its advertised interpretation unchanged.

1. **The production selector permits later fetches at an exact anchor.** Its
   `own_run = anchor == reference_time` bypasses the fetch-time restriction. This
   is intentional for reconstructing what a production run actually used, but it
   is not a forecast available at the nominal reference time. There are **77**
   otherwise replayable hourly references with later fetches, by **1.89–97.06
   minutes** (median **25.22 minutes**). They are excluded, without changing
   production or silently pretending that an earlier snapshot was selected.
   An older anchor with a future fetch is correctly rejected; regression tests
   exercise both sides. Availability is certified by fetch time, not a DWD model
   issue time: the latter is not in this archive.
2. **Two reader defects hide real old forecasts.** For a wholly absent
   `reference_time` column, `_as_utc_series(None)` returns a **one-row NaT Series**;
   `fillna(fetched)` does not expand it to the frame's index. An old snapshot
   can therefore collapse to its first row. Separately, `_read_partition`
   infers one timestamp format and coerces valid ISO strings in the other format
   to NaT. This loses **120 rows in each of May, June and July**. The experiment
   adapter reparses original timestamp text with `format='mixed'` and explicitly
   supplies the documented fetch-as-anchor fallback on in-memory copies. It then
   calls the production selector and additionally requires `fetched <= reference`.
   Mixed-schema field names are coalesced rowwise. Nothing is written to the archive.
3. **Seven old snapshots really lack timestamps.** June has 360 such rows and
   July 480 (seven 120-hour snapshots). These remain unreplayable: the experiment
   does not invent their timestamps. The adapter distinguishes these actual losses
   from the 360 valid strings the parser had hidden.
4. **Part A retained a southern oracle.** Replacing Munich air while retaining
   observed `t_catchment` is a conditional sensitivity, not all-weather operational
   skill. The audit adds a weather-free reference and separate Munich-only oracle
   and replay arms, all retaining the same Bad Tölz history. This makes the loss
   of a *specified* oracle advantage measurable. No real forecast of the entire
   selected covariate set, especially catchment rain, is available here.
5. **Solar was counted but never forecast with.** `_solar` was absent from the
   supplied model's future covariates. The audited run actually evaluates matched
   Munich air/solar oracle and replay arms. This is a Munich experiment, not a
   replay of exp13's Hohenpeißenberg or Garmisch radiation stations.
6. **`DRAWS = 20` was unused.** The supplied run made one stochastic realization.
   The audit runs 20 realizations for each of the original block method and a whole-trace
   alternative on one fixed non-overlapping subset. Those realizations (RNG seed 0) are not new
   weather observations.
7. **The supplied exp16 had no checkpoint binding or resume.** It rewrote a CSV
   after each variant, unlike exp15. The audited driver uses atomic save, validated
   resume, and manifests binding data, anchors, overrides' error matrix, weather
   archive hashes, relevant source hashes, backend/package versions, and seed
   protocol. An interrupted preliminary run is retained separately as diagnostic
   cache data and is not used in the final tables.

## What the block simulation does and does not preserve

On the complete rectangular error matrix used here, one shared row index per
24-hour block really does select consecutive leads of the same reference forecast.
The claim is correct *within* a block. It is not correct across block boundaries:
measured adjacent-lead correlations are about 0.83–0.88 there, whereas the block
simulation makes them approximately zero. The full numerical autocorrelation audit
below centres each lead across references before pooling, so a lead-dependent mean
bias is not mistaken for error persistence. It includes 10,000 generated traces
per method as a numerical diagnostic, not as an enlarged observational sample.

Drawing one entire empirical 96-hour error trajectory preserves the measured
horizon dependence, lead-specific error distribution and daily pattern in
expectation. The audit measures that alternative directly. The original `error_pool`
would also lose row identity if different references were missing at different leads:
separate grouped arrays plus modulo indexing do not establish forecast identity.
The complete-horizon eligibility requirement prevents that failure in this run;
a general implementation should pivot by reference time and lead and validate rows.

**Neither simulation is calibrated for southern Germany merely by matching Munich.**
It assumes spatial transferability, draws southern errors independently of the
actual Munich forecast error, and does not condition on weather regime, season,
time of day, or forecast issue. The donor pool is estimated from these same replay months, not a separate
calibration period; it also fixes that pool rather than propagating its estimation
uncertainty. Repeating draws measures Monte Carlo
sensitivity; it does not cure these assumptions. A measured joint Munich/south
forecast-error archive would be needed to validate them. The whole-trace method is
a better temporal surrogate, not proof or a statistical bound on operational risk.

## Simulation result: small average optimism from blocks, uncertain magnitude

On the **same 16 windows**, the two-air oracle scores **0.391950 MAE / 0.259308
CRPS °C**, and Munich replay with southern oracle scores **0.406306 / 0.271250 °C**.
Across 20 realizations, the original 24-hour blocks score **0.419171 / 0.278588 °C**;
whole 96-hour traces score **0.424902 / 0.282129 °C**. Thus whole-trace simulated
weather increases MAE by **8.407%** and CRPS by **8.801%** over the two-air oracle
on this subset, with conditional intervals **[+0.712%, +17.585%]** and
**[+1.843%, +16.750%]**. These are increases in oracle error, not percentages of
oracle *advantage lost*. The incremental simulated southern cost relative to
partial replay is **0.018597 MAE / 0.010878 CRPS °C**.

Whole traces are worse than 24-hour blocks by **0.005732 MAE °C
[−0.006951, +0.018165]** and **0.003541 CRPS °C [−0.004188, +0.010820]**. The block
method looks slightly flattering on average, but this difference is not established
precisely. Choose whole traces because they reproduce the measured temporal
structure, not because these model scores prove blocks systematically optimistic.
Across individual realizations, the whole-trace MAE penalty has an SD of **3.17
percentage points** of the oracle score: the original single draw would have been
a materially noisy answer. The tables separate this Monte Carlo sensitivity from
window uncertainty and do not treat the 20 repetitions as 320 observed forecasts.
Do not compare these 16-window percentages directly with the 737-window headline.

## Sample, overlap and interpretation

The audited replay has **737** complete 96-hour windows from **2026-05-01 19:00 UTC
to 2026-09-10 11:00 UTC**, drawn from **81** distinct fetches. They contain **70,752**
scored forecast-hour pairs but only **1,624 distinct verifying hours**; **99.18%**
of neighbouring references have overlapping horizons, and median spacing is one
hour. A chronological greedy selection gives **16 non-overlapping 96-hour windows**.
That count is not a formal effective sample size, and shared seasonal conditions
can correlate even these windows.

Most references are in August and early September: May, June and July contribute
only 24 windows and two usable snapshots each. Ordinary paired-window intervals
therefore overstate precision. The report also gives 4-, 7- and 14-day block
sensitivities, and an explicitly separate non-overlapping-window comparison.
Fixed-block boundaries can still cut overlapping weather events; few occupied
blocks and one warm-season sample limit even these intervals. None establishes
winter or whole-year performance. The all-hour average also weights long stretches
of one snapshot more heavily than a once-per-production-run evaluation would.

All arms use the same cached, finalized historical context and the same observed
verification hours. They are **weather-substitution experiments**, not vintage
replays of every historical input. In particular, hourly water means are labelled
at the start of the aggregation hour and would not all be available at that exact
instant in a live run; historical station revisions and availability are not
replayed. The fetch-time guard prevents future-weather selection but does not turn
the whole benchmark into an end-to-end as-of production backtest.

exp15 uses 250 fixed windows (2019: 1; 2020: 38; 2021: 37; 2022: 36; 2023: 38;
2024: 36; 2025: 37; 2026: 27), shared across all rain arms. The original four-arm
confirmation omitted the hourly mean, making a full-context attribution to
accumulation impossible. `exp15_hourly_confirm.py` adds that pre-specified control
on the identical anchors. All raw rain futures are complete. Historical raw rain
gaps affect 62 full-context windows (four in the screen), so contrasts include the
existing input interpolation as well as representation and slot count. The mean's
12-observation rolling minimum differs from the individual sums' 24 only at 12
startup hours. Separately, the individually accumulated series have 18 missing
future cells in the **2023-02-15 06:00 UTC** window because their lookback contains
a gap; the existing preprocessing interpolates them. Excluding that one window
only as a sensitivity leaves accumulation versus raw hourly at **−0.126% MAE
[−1.263%, +0.989%] / −0.151% CRPS [−1.189%, +0.892%]**, so the conclusion is unchanged. Yearly and leave-one-year-out
tables expose harm and influential years instead of interpreting a pooled interval
crossing zero as a universal no-harm result. The single 2019 window cannot provide
an inferential annual interval.

## Radiation archive boundaries

The archive does contain old global-radiation forecasts, in hourly **kWh/m²**.
The zero-radiation August gap remains a real archive limitation. The supplied
“mid-September” restart date is inaccurate for this checkout: radiation resumes
at the **2026-09-04 15:31:41 UTC fetch**. There are 27 September fetches carrying it;
192 strict, complete temperature-replay windows also have complete solar horizons
(72 in May–July, 120 in September). Requiring *observed* solar to be complete too
leaves **110** matched model-scoring windows; 82 windows with missing solar truth
are excluded from all arms of that comparison. On the complete subset, solar MAE
is **0.05131 kWh/m²** over all hours but **0.07991 kWh/m²** when observed radiation
is positive. About **35.8%** of all hours have zero observed radiation: a small
all-hour radiation error should not be mistaken for equally accurate daylight
forecasts. The full 96-hour requirement and verification
cutoff exclude more recent references, including the current day's fetches.

The solar experiment uses the cached `solar_muenchen` observations and the archived
Munich forecast field in the same units, without filling the August gap. Its
observation source comes from a coordinate lookup rather than a pinned radiation
station identity, so spatial/source mismatch can contribute to measured error.
It cannot validate the usefulness of other radiation stations or extrapolate over
the missing month.

## Solar result: the attractive total-oracle comparison hides the loss

On the 110 matched windows, Munich air replay alone has **0.495586 MAE /
0.319029 CRPS °C**. Adding *observed* solar gives **0.477277 / 0.306957 °C**
(−3.694% / −3.784%), but adding *forecast* solar gives **0.504594 / 0.326569 °C**
(**+1.818% / +2.363% worse than omitting solar**). Its 7-day block intervals
against no solar are **[−4.759%, +15.211%] / [−4.311%, +16.123%]**. The worse
mean fails the keep rule in this experiment; the intervals do not prove harm.

Holding air replay fixed, replacing oracle solar with forecast solar costs
**+5.724% MAE / +6.389% CRPS**. Ordinary paired intervals exclude zero, but
7-day block intervals again do not: **[−1.493%, +17.020%] / [−1.381%, +18.824%]**.
Even the observed-solar gain loses its apparent significance under block resampling.

A superficially excellent number would compare both weather oracles
(**0.503999 MAE**) with both weather replays (**0.504594 MAE**) and conclude
that forecasts cost almost nothing. That joint cost is a valid paired
description of this subset, but it combines two effects that nearly cancel;
it is not evidence that the solar forecast retains the oracle solar value.
The fixed-air contrast above exposes that loss. Moreover, Munich air replay itself is slightly better than its
oracle on this selected subset (0.495586 versus 0.501512 MAE). These are finite,
selected, dependent windows, not evidence that forecasts are intrinsically more
informative than observations. Do not compare this subset's absolute scores with
the 737-window temperature scores, or use it to reject other solar stations from
exp13. It establishes no deployable gain for this Munich solar addition.

## Reproduction and validation

Use the existing `.venv-exp11` environment with `TIMESFM_BACKEND=mlx`,
`HF_HOME=data/experiments/hf` and (after model download) `HF_HUB_OFFLINE=1`.
For an exact resume of source-bound inference checkpoints, use the scoring sources
from `06aacf9`; the final docstring-only correction intentionally changes the text
hash, not execution. Run `exp15_rain.py`, `exp15_hourly_confirm.py`, then `exp16_honesty.py` serially;
run `diagnose_exp16.py` and `report_exp15_exp16.py` to rebuild the tables. Model
execution needs Apple GPU access outside the restricted process sandbox. Completed
variants resume; simultaneous model instances exhausted shared memory in an early
attempt, which was interrupted and resumed serially. No interrupted variant is
accepted as a complete checkpoint.

Final checks: **189 production tests passed, 2 skipped** (the unavailable upstream
comparison dependency), **22 relevant experiment tests passed**, and `ruff check .`
passed. The existing PyTorch `torch.jit.script` deprecation warning remains. All five
weather-partition SHA-256 hashes match the pre-inference manifest; every executed
source hash matches commit `06aacf9`. The production paths have no diff from the
starting commit.

The report and experiment code are the deliverables. Generated CSVs and logs remain
in the ignored local cache; no PNG/CSV is committed. Production code, `main.py`,
`tests/` and the production archive are unchanged.

<!-- generated tables -->
## Paired scores and intervals

Negative deltas favour the named variant. Ordinary paired bootstrap: 8,000 shared draws, 95% percentile intervals; these assume independent windows and are secondary for the overlapping replay. Block intervals resample all windows of a calendar block together. No multiplicity adjustment; neither screen confirmation on the same windows nor retrospective yearly comparisons are held-out validation. Absolute scores pool observed hours; percent deltas average per-window scores. CRPS integrates only deciles 0.1–0.9.

### exp15, screen 1024 h

Reference: **ohne Regen**; MAE 0.447583 °C, CRPS 0.293158 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | 250 | 0.442670 | -1.098% [-2.566, +0.375] | 0.290593 | -0.875% [-2.353, +0.609] |
| Mittel, stündlich | 250 | 0.442125 | -1.220% [-2.377, -0.025] | 0.290124 | -1.035% [-2.165, +0.106] |
| vier Rohstationen, stündlich | 250 | 0.447811 | +0.051% [-1.301, +1.430] | 0.293532 | +0.128% [-1.200, +1.453] |
| vier Rohstationen, je 24h-Summe | 250 | 0.449900 | +0.517% [-1.339, +2.380] | 0.294670 | +0.515% [-1.366, +2.366] |
| nur München, 24h-Summe | 250 | 0.448660 | +0.240% [-1.033, +1.505] | 0.294310 | +0.393% [-0.835, +1.595] |

### exp15, confirmation 8760 h

Reference: **ohne Regen**; MAE 0.360457 °C, CRPS 0.236630 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | 250 | 0.352187 | -2.295% [-3.813, -0.811] | 0.230519 | -2.583% [-4.014, -1.206] |
| vier Rohstationen, stündlich | 250 | 0.351384 | -2.518% [-4.222, -0.882] | 0.230228 | -2.706% [-4.292, -1.156] |
| vier Rohstationen, je 24h-Summe | 250 | 0.350869 | -2.660% [-4.339, -0.971] | 0.229846 | -2.867% [-4.455, -1.321] |
| Mittel, stündlich | 250 | 0.353484 | -1.937% [-3.302, -0.574] | 0.231666 | -2.100% [-3.378, -0.834] |

| Variant vs ohne Regen | MAE % monthly-block CI | CRPS % monthly-block CI |
| --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | [-3.979, -0.740] | [-4.201, -1.101] |
| vier Rohstationen, stündlich | [-4.209, -0.907] | [-4.306, -1.199] |
| vier Rohstationen, je 24h-Summe | [-4.458, -0.915] | [-4.580, -1.259] |
| Mittel, stündlich | [-3.289, -0.596] | [-3.386, -0.872] |

### exp15 direct construction contrast, 8760 h

Reference: **Mittel, stündlich**; MAE 0.353484 °C, CRPS 0.231666 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | 250 | 0.352187 | -0.365% [-1.200, +0.461] | 0.230519 | -0.493% [-1.272, +0.275] |
| vier Rohstationen, stündlich | 250 | 0.351384 | -0.593% [-1.350, +0.152] | 0.230228 | -0.620% [-1.330, +0.056] |

| Variant vs Mittel, stündlich | MAE % monthly-block CI | CRPS % monthly-block CI |
| --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | [-1.231, +0.490] | [-1.322, +0.306] |
| vier Rohstationen, stündlich | [-1.392, +0.177] | [-1.328, +0.071] |

### exp15 direct construction contrast, 8760 h

Reference: **vier Rohstationen, stündlich**; MAE 0.351384 °C, CRPS 0.230228 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| vier Rohstationen, je 24h-Summe | 250 | 0.350869 | -0.145% [-1.270, +0.990] | 0.229846 | -0.165% [-1.208, +0.886] |

| Variant vs vier Rohstationen, stündlich | MAE % monthly-block CI | CRPS % monthly-block CI |
| --- | --- | --- |
| vier Rohstationen, je 24h-Summe | [-1.292, +0.951] | [-1.170, +0.839] |

### exp15 direct construction contrast, 8760 h

Reference: **Mittel, 24h-Summe (gesetzt)**; MAE 0.352187 °C, CRPS 0.230519 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| vier Rohstationen, stündlich | 250 | 0.351384 | -0.229% [-1.334, +0.846] | 0.230228 | -0.127% [-1.145, +0.886] |
| vier Rohstationen, je 24h-Summe | 250 | 0.350869 | -0.374% [-1.265, +0.508] | 0.229846 | -0.291% [-1.092, +0.510] |

| Variant vs Mittel, 24h-Summe (gesetzt) | MAE % monthly-block CI | CRPS % monthly-block CI |
| --- | --- | --- |
| vier Rohstationen, stündlich | [-1.366, +0.875] | [-1.175, +0.907] |
| vier Rohstationen, je 24h-Summe | [-1.304, +0.553] | [-1.124, +0.547] |

### Accumulation sensitivity: complete future rolling sums only

Excluded from this diagnostic only: [Timestamp('2023-02-15 06:00:00+0000', tz='UTC')]. Main tables retain every window.

Reference: **vier Rohstationen, stündlich**; MAE 0.351733 °C, CRPS 0.230480 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| vier Rohstationen, je 24h-Summe | 249 | 0.351286 | -0.126% [-1.263, +0.989] | 0.230131 | -0.151% [-1.189, +0.892] |

### exp15 year 2019

One window: no inferential interval can be estimated.

| Variant | MAE °C | CRPS °C |
| --- | --- | --- |
| ohne Regen | 0.201289 | 0.132734 |
| Mittel, 24h-Summe (gesetzt) | 0.201749 | 0.133967 |
| vier Rohstationen, je 24h-Summe | 0.205692 | 0.134939 |
| Mittel, stündlich | 0.210459 | 0.137151 |
| vier Rohstationen, stündlich | 0.215268 | 0.140520 |

### exp15 year 2020

Reference: **ohne Regen**; MAE 0.410411 °C, CRPS 0.265663 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | 38 | 0.385715 | -6.022% [-10.359, -1.338] | 0.249877 | -5.946% [-10.324, -1.326] |
| vier Rohstationen, stündlich | 38 | 0.385586 | -6.057% [-10.420, -1.630] | 0.249576 | -6.062% [-10.346, -1.707] |
| vier Rohstationen, je 24h-Summe | 38 | 0.387613 | -5.558% [-9.870, -1.228] | 0.251314 | -5.403% [-9.711, -1.100] |
| Mittel, stündlich | 38 | 0.391569 | -4.603% [-8.726, -0.297] | 0.253538 | -4.574% [-8.527, -0.430] |

### exp15 year 2021

Reference: **ohne Regen**; MAE 0.382095 °C, CRPS 0.255491 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | 37 | 0.373286 | -2.303% [-5.519, +1.222] | 0.247582 | -3.091% [-6.223, +0.026] |
| vier Rohstationen, stündlich | 37 | 0.372508 | -2.507% [-6.138, +1.184] | 0.247344 | -3.186% [-6.405, -0.008] |
| vier Rohstationen, je 24h-Summe | 37 | 0.370698 | -2.980% [-6.773, +1.120] | 0.246000 | -3.711% [-7.415, -0.028] |
| Mittel, stündlich | 37 | 0.372189 | -2.594% [-5.313, +0.211] | 0.247757 | -3.028% [-5.351, -0.770] |

### exp15 year 2022

Reference: **ohne Regen**; MAE 0.352908 °C, CRPS 0.229464 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | 36 | 0.359566 | +1.887% [-1.816, +5.599] | 0.231424 | +0.854% [-2.181, +3.855] |
| vier Rohstationen, stündlich | 36 | 0.354912 | +0.568% [-4.518, +5.400] | 0.229954 | +0.214% [-4.330, +4.348] |
| vier Rohstationen, je 24h-Summe | 36 | 0.359129 | +1.763% [-2.954, +6.283] | 0.231458 | +0.869% [-3.295, +4.805] |
| Mittel, stündlich | 36 | 0.354371 | +0.415% [-3.261, +3.738] | 0.229546 | +0.036% [-3.156, +2.958] |

### exp15 year 2023

Reference: **ohne Regen**; MAE 0.320785 °C, CRPS 0.211649 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | 38 | 0.323661 | +0.896% [-2.292, +4.153] | 0.211952 | +0.143% [-2.802, +3.154] |
| vier Rohstationen, stündlich | 38 | 0.323124 | +0.729% [-2.602, +3.850] | 0.211668 | +0.009% [-3.209, +3.071] |
| vier Rohstationen, je 24h-Summe | 38 | 0.318183 | -0.811% [-4.610, +3.019] | 0.209013 | -1.246% [-4.862, +2.462] |
| Mittel, stündlich | 38 | 0.325320 | +1.414% [-1.221, +3.974] | 0.212967 | +0.623% [-1.892, +3.034] |

### exp15 year 2024

Reference: **ohne Regen**; MAE 0.356629 °C, CRPS 0.236505 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | 36 | 0.336395 | -5.674% [-11.005, -1.012] | 0.222790 | -5.799% [-11.014, -1.398] |
| vier Rohstationen, stündlich | 36 | 0.346279 | -2.902% [-8.345, +1.858] | 0.230000 | -2.750% [-8.371, +1.965] |
| vier Rohstationen, je 24h-Summe | 36 | 0.338394 | -5.113% [-10.395, -0.287] | 0.224208 | -5.199% [-10.395, -0.788] |
| Mittel, stündlich | 36 | 0.346742 | -2.772% [-7.409, +1.560] | 0.229768 | -2.849% [-7.442, +1.279] |

### exp15 year 2025

Reference: **ohne Regen**; MAE 0.293642 °C, CRPS 0.193706 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | 37 | 0.283952 | -3.296% [-6.960, +0.136] | 0.187916 | -2.985% [-5.926, -0.128] |
| vier Rohstationen, stündlich | 37 | 0.278943 | -5.002% [-9.919, -0.531] | 0.184897 | -4.544% [-8.642, -0.838] |
| vier Rohstationen, je 24h-Summe | 37 | 0.282529 | -3.780% [-8.334, +0.591] | 0.187485 | -3.207% [-7.079, +0.547] |
| Mittel, stündlich | 37 | 0.280684 | -4.408% [-8.303, -0.578] | 0.185920 | -4.016% [-7.334, -0.715] |

### exp15 year 2026

Reference: **ohne Regen**; MAE 0.429019 °C, CRPS 0.277508 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Mittel, 24h-Summe (gesetzt) | 27 | 0.426563 | -0.573% [-3.243, +2.174] | 0.277097 | -0.148% [-2.779, +2.671] |
| vier Rohstationen, stündlich | 27 | 0.420524 | -1.980% [-5.975, +2.026] | 0.271794 | -2.059% [-6.188, +2.036] |
| vier Rohstationen, je 24h-Summe | 27 | 0.422665 | -1.481% [-4.929, +2.326] | 0.273771 | -1.347% [-4.769, +2.353] |
| Mittel, stündlich | 27 | 0.426787 | -0.520% [-3.183, +2.131] | 0.276722 | -0.283% [-2.933, +2.446] |

### exp15 leave-one-year-out

| Omitted year | Variant | MAE % | CRPS % |
| --- | --- | --- | --- |
| 2019 | Mittel, 24h-Summe (gesetzt) | -2.300 | -2.591 |
| 2019 | vier Rohstationen, stündlich | -2.539 | -2.726 |
| 2020 | Mittel, 24h-Summe (gesetzt) | -1.515 | -1.891 |
| 2020 | vier Rohstationen, stündlich | -1.778 | -2.016 |
| 2021 | Mittel, 24h-Summe (gesetzt) | -2.293 | -2.486 |
| 2021 | vier Rohstationen, stündlich | -2.520 | -2.615 |
| 2022 | Mittel, 24h-Summe (gesetzt) | -2.981 | -3.140 |
| 2022 | vier Rohstationen, stündlich | -3.025 | -3.180 |
| 2023 | Mittel, 24h-Summe (gesetzt) | -2.794 | -3.012 |
| 2023 | vier Rohstationen, stündlich | -3.026 | -3.133 |
| 2024 | Mittel, 24h-Summe (gesetzt) | -1.733 | -2.042 |
| 2024 | vier Rohstationen, stündlich | -2.455 | -2.699 |
| 2025 | Mittel, 24h-Summe (gesetzt) | -2.158 | -2.527 |
| 2025 | vier Rohstationen, stündlich | -2.178 | -2.453 |
| 2026 | Mittel, 24h-Summe (gesetzt) | -2.549 | -2.936 |
| 2026 | vier Rohstationen, stündlich | -2.598 | -2.800 |

### exp16 full replay sample, 8760 h

Reference: **Ohne Wetter**; MAE 0.715933 °C, CRPS 0.470842 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Muenchen Orakel | 737 | 0.316468 | -55.796% [-59.963, -51.749] | 0.210305 | -55.334% [-59.320, -51.428] |
| Muenchen Replay | 737 | 0.335661 | -53.116% [-57.178, -49.161] | 0.221038 | -53.055% [-57.015, -49.212] |
| Zwei Luft Orakel | 737 | 0.319075 | -55.432% [-59.616, -51.344] | 0.212707 | -54.824% [-58.840, -50.852] |
| Muenchen Replay + Sued Orakel | 737 | 0.331898 | -53.641% [-57.707, -49.632] | 0.219874 | -53.302% [-57.270, -49.434] |

### exp16 real Munich weather cost

Reference: **Muenchen Orakel**; MAE 0.316468 °C, CRPS 0.210305 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Muenchen Replay | 737 | 0.335661 | +6.065% [+3.753, +8.533] | 0.221038 | +5.103% [+2.953, +7.408] |

| Variant vs Muenchen Orakel | MAE % 4d block CI | CRPS % 4d block CI |
| --- | --- | --- |
| Muenchen Replay | [-5.355, +19.771] | [-5.261, +17.426] |

| Variant vs Muenchen Orakel | MAE % 7d block CI | CRPS % 7d block CI |
| --- | --- | --- |
| Muenchen Replay | [-6.797, +21.011] | [-6.754, +18.631] |

| Variant vs Muenchen Orakel | MAE % 14d block CI | CRPS % 14d block CI |
| --- | --- | --- |
| Muenchen Replay | [-5.822, +23.089] | [-5.624, +20.714] |

| Metric | Ohne Wetter | Muenchen Orakel | Muenchen Replay | Replay cost °C [7d CI] | Lost oracle advantage % [7d CI] | Bootstrap nonpositive advantage |
| --- | --- | --- | --- | --- | --- | --- |
| mae | 0.715933 | 0.316468 | 0.335661 | +0.019193 [-0.021511, +0.066493] | 4.80% [-7.01, 13.86] | 0.00% |
| crps | 0.470842 | 0.210305 | 0.221038 | +0.010733 [-0.014204, +0.039182] | 4.12% [-6.91, 12.73] | 0.00% |

### exp16 conditional cost with southern oracle retained

Reference: **Zwei Luft Orakel**; MAE 0.319075 °C, CRPS 0.212707 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Muenchen Replay + Sued Orakel | 737 | 0.331898 | +4.019% [+2.488, +5.679] | 0.219874 | +3.369% [+1.992, +4.857] |

| Variant vs Zwei Luft Orakel | MAE % 7d block CI | CRPS % 7d block CI |
| --- | --- | --- |
| Muenchen Replay + Sued Orakel | [-3.335, +12.442] | [-3.410, +11.082] |

| Metric | Ohne Wetter | Zwei Luft Orakel | Muenchen Replay + Sued Orakel | Replay cost °C [7d CI] | Lost oracle advantage % [7d CI] | Bootstrap nonpositive advantage |
| --- | --- | --- | --- | --- | --- | --- |
| mae | 0.715933 | 0.319075 | 0.331898 | +0.012823 [-0.010643, +0.039699] | 3.23% [-2.97, 8.13] | 0.00% |
| crps | 0.470842 | 0.212707 | 0.219874 | +0.007166 [-0.007254, +0.023571] | 2.78% [-3.28, 7.47] | 0.00% |

### exp16 2026-05

Reference: **Muenchen Orakel**; MAE 0.644724 °C, CRPS 0.421850 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Muenchen Replay | 24 | 0.665150 | +3.168% [-0.651, +6.966] | 0.434773 | +3.063% [+0.169, +5.885] |

### exp16 2026-06

Reference: **Muenchen Orakel**; MAE 0.572838 °C, CRPS 0.368806 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Muenchen Replay | 24 | 0.661673 | +15.508% [+5.009, +26.090] | 0.428425 | +16.166% [+5.484, +27.229] |

### exp16 2026-07

Reference: **Muenchen Orakel**; MAE 0.574739 °C, CRPS 0.364644 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Muenchen Replay | 24 | 0.458208 | -20.275% [-27.498, -13.461] | 0.285957 | -21.579% [-29.368, -14.218] |

### exp16 2026-08

Reference: **Muenchen Orakel**; MAE 0.278646 °C, CRPS 0.187012 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Muenchen Replay | 471 | 0.323284 | +16.020% [+12.611, +19.487] | 0.213166 | +13.985% [+10.879, +17.162] |

### exp16 2026-09

Reference: **Muenchen Orakel**; MAE 0.304017 °C, CRPS 0.201983 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Muenchen Replay | 194 | 0.269455 | -11.368% [-14.099, -8.722] | 0.180021 | -10.873% [-13.239, -8.628] |

### Non-overlapping 96h subset (16 windows), actual replay

Reference: **Muenchen Orakel**; MAE 0.379502 °C, CRPS 0.249430 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Muenchen Replay | 16 | 0.425245 | +12.053% [-0.444, +26.332] | 0.279085 | +11.889% [-0.239, +25.565] |

### Simulation on that same subset: 20 draws per method

Intervals resample windows and Monte Carlo realizations independently. They are conditional on the fixed empirical donor pool and the unverified Munich-to-south error transfer, not confidence in real southern forecast skill.

| Method | Metric | Windows | Score °C | Δ vs two-air oracle % [95% CI] | Across-draw Δ SD (percentage points) | Cost vs partial replay °C |
| --- | --- | --- | --- | --- | --- | --- |
| block24 | mae | 16 | 0.419171 | +6.945 [-1.053, +16.181] | 3.635 | +0.012865 |
| block24 | crps | 16 | 0.278588 | +7.435 [+0.093, +15.415] | 3.240 | +0.007337 |
| trace96 | mae | 16 | 0.424902 | +8.407 [+0.712, +17.585] | 3.169 | +0.018597 |
| trace96 | crps | 16 | 0.282129 | +8.801 [+1.843, +16.750] | 2.965 | +0.010878 |

### Whole-trace versus original-block simulation

Paired windows, independently resampled Monte Carlo realizations for each method.

| Metric | Trace96 minus block24 °C | 95% conditional CI °C |
| --- | --- | --- |
| mae | +0.005732 | [-0.006951, +0.018165] |
| crps | +0.003541 | [-0.004188, +0.010820] |

### Munich solar, matched 110 windows

Reference: **Muenchen Replay**; MAE 0.495586 °C, CRPS 0.319029 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Luft Solar Orakel | 110 | 0.503999 | +1.698% [-4.025, +7.123] | 0.325745 | +2.105% [-3.926, +7.819] |
| Luft Solar Replay | 110 | 0.504594 | +1.818% [-0.780, +4.493] | 0.326569 | +2.363% [-0.329, +5.130] |
| Luft Replay Solar Orakel | 110 | 0.477277 | -3.694% [-7.011, -0.463] | 0.306957 | -3.784% [-7.432, -0.256] |
| Muenchen Orakel | 110 | 0.501512 | +1.196% [-3.264, +5.361] | 0.324032 | +1.568% [-2.987, +5.861] |

| Variant vs Muenchen Replay | MAE % 7d block CI | CRPS % 7d block CI |
| --- | --- | --- |
| Luft Solar Orakel | [-18.748, +23.631] | [-19.007, +25.579] |
| Luft Solar Replay | [-4.759, +15.211] | [-4.311, +16.123] |
| Luft Replay Solar Orakel | [-14.137, +8.496] | [-15.420, +8.966] |
| Muenchen Orakel | [-14.738, +17.697] | [-14.562, +18.746] |

Reference: **Luft Solar Orakel**; MAE 0.503999 °C, CRPS 0.325745 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Luft Solar Replay | 110 | 0.504594 | +0.118% [-4.383, +4.883] | 0.326569 | +0.253% [-4.452, +5.252] |

| Variant vs Luft Solar Orakel | MAE % 7d block CI | CRPS % 7d block CI |
| --- | --- | --- |
| Luft Solar Replay | [-11.973, +21.324] | [-12.611, +22.304] |

### Solar forecast substitution with air replay held fixed

Reference: **Luft Replay Solar Orakel**; MAE 0.477277 °C, CRPS 0.306957 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Luft Solar Replay | 110 | 0.504594 | +5.724% [+3.179, +8.364] | 0.326569 | +6.389% [+3.455, +9.532] |

| Variant vs Luft Replay Solar Orakel | MAE % 7d block CI | CRPS % 7d block CI |
| --- | --- | --- |
| Luft Solar Replay | [-1.493, +17.020] | [-1.381, +18.824] |

## DWD error by lead hour

Error = forecast minus cached observation, °C. Lead is relative to the hourly reference time, not an archived DWD model issue time. No such issue timestamp is available. Each hour has the same 737 replay windows.

| Lead h | N | Bias °C | MAE °C | RMSE °C |
| --- | --- | --- | --- | --- |
| 1 | 737 | -0.324 | 0.995 | 1.294 |
| 2 | 737 | -0.359 | 1.005 | 1.309 |
| 3 | 737 | -0.388 | 1.029 | 1.332 |
| 4 | 737 | -0.422 | 1.047 | 1.351 |
| 5 | 737 | -0.436 | 1.072 | 1.376 |
| 6 | 737 | -0.465 | 1.102 | 1.417 |
| 7 | 737 | -0.491 | 1.097 | 1.415 |
| 8 | 737 | -0.496 | 1.097 | 1.426 |
| 9 | 737 | -0.461 | 1.086 | 1.429 |
| 10 | 737 | -0.444 | 1.095 | 1.435 |
| 11 | 737 | -0.427 | 1.095 | 1.436 |
| 12 | 737 | -0.401 | 1.088 | 1.436 |
| 13 | 737 | -0.383 | 1.087 | 1.438 |
| 14 | 737 | -0.382 | 1.082 | 1.436 |
| 15 | 737 | -0.360 | 1.076 | 1.436 |
| 16 | 737 | -0.337 | 1.074 | 1.446 |
| 17 | 737 | -0.322 | 1.073 | 1.444 |
| 18 | 737 | -0.318 | 1.090 | 1.481 |
| 19 | 737 | -0.318 | 1.094 | 1.484 |
| 20 | 737 | -0.310 | 1.108 | 1.496 |
| 21 | 737 | -0.332 | 1.125 | 1.516 |
| 22 | 737 | -0.362 | 1.147 | 1.549 |
| 23 | 737 | -0.381 | 1.148 | 1.543 |
| 24 | 737 | -0.408 | 1.157 | 1.551 |
| 25 | 737 | -0.427 | 1.162 | 1.557 |
| 26 | 737 | -0.445 | 1.167 | 1.561 |
| 27 | 737 | -0.465 | 1.169 | 1.557 |
| 28 | 737 | -0.458 | 1.185 | 1.571 |
| 29 | 737 | -0.474 | 1.179 | 1.555 |
| 30 | 737 | -0.486 | 1.196 | 1.571 |
| 31 | 737 | -0.509 | 1.190 | 1.559 |
| 32 | 737 | -0.513 | 1.191 | 1.557 |
| 33 | 737 | -0.483 | 1.183 | 1.547 |
| 34 | 737 | -0.457 | 1.189 | 1.553 |
| 35 | 737 | -0.414 | 1.187 | 1.555 |
| 36 | 737 | -0.391 | 1.201 | 1.562 |
| 37 | 737 | -0.380 | 1.203 | 1.562 |
| 38 | 737 | -0.363 | 1.206 | 1.565 |
| 39 | 737 | -0.342 | 1.188 | 1.545 |
| 40 | 737 | -0.326 | 1.166 | 1.513 |
| 41 | 737 | -0.287 | 1.149 | 1.494 |
| 42 | 737 | -0.279 | 1.163 | 1.513 |
| 43 | 737 | -0.271 | 1.182 | 1.530 |
| 44 | 737 | -0.253 | 1.194 | 1.542 |
| 45 | 737 | -0.269 | 1.201 | 1.559 |
| 46 | 737 | -0.301 | 1.218 | 1.578 |
| 47 | 737 | -0.315 | 1.220 | 1.595 |
| 48 | 737 | -0.315 | 1.233 | 1.622 |
| 49 | 737 | -0.336 | 1.246 | 1.642 |
| 50 | 737 | -0.352 | 1.259 | 1.661 |
| 51 | 737 | -0.351 | 1.269 | 1.684 |
| 52 | 737 | -0.360 | 1.294 | 1.722 |
| 53 | 737 | -0.357 | 1.309 | 1.738 |
| 54 | 737 | -0.386 | 1.327 | 1.767 |
| 55 | 737 | -0.406 | 1.348 | 1.794 |
| 56 | 737 | -0.404 | 1.355 | 1.798 |
| 57 | 737 | -0.387 | 1.375 | 1.827 |
| 58 | 737 | -0.363 | 1.389 | 1.841 |
| 59 | 737 | -0.358 | 1.385 | 1.830 |
| 60 | 737 | -0.361 | 1.395 | 1.825 |
| 61 | 737 | -0.371 | 1.388 | 1.813 |
| 62 | 737 | -0.361 | 1.380 | 1.794 |
| 63 | 737 | -0.364 | 1.387 | 1.792 |
| 64 | 737 | -0.350 | 1.372 | 1.765 |
| 65 | 737 | -0.327 | 1.354 | 1.741 |
| 66 | 737 | -0.334 | 1.352 | 1.736 |
| 67 | 737 | -0.361 | 1.372 | 1.759 |
| 68 | 737 | -0.353 | 1.386 | 1.777 |
| 69 | 737 | -0.368 | 1.398 | 1.802 |
| 70 | 737 | -0.382 | 1.425 | 1.852 |
| 71 | 737 | -0.385 | 1.450 | 1.893 |
| 72 | 737 | -0.399 | 1.472 | 1.931 |
| 73 | 737 | -0.408 | 1.496 | 1.962 |
| 74 | 737 | -0.423 | 1.524 | 1.996 |
| 75 | 737 | -0.435 | 1.539 | 2.014 |
| 76 | 737 | -0.452 | 1.565 | 2.046 |
| 77 | 737 | -0.461 | 1.583 | 2.073 |
| 78 | 737 | -0.482 | 1.623 | 2.129 |
| 79 | 737 | -0.515 | 1.653 | 2.159 |
| 80 | 737 | -0.541 | 1.672 | 2.168 |
| 81 | 737 | -0.538 | 1.687 | 2.186 |
| 82 | 737 | -0.524 | 1.702 | 2.192 |
| 83 | 737 | -0.508 | 1.720 | 2.205 |
| 84 | 737 | -0.485 | 1.740 | 2.227 |
| 85 | 737 | -0.469 | 1.765 | 2.258 |
| 86 | 737 | -0.460 | 1.788 | 2.284 |
| 87 | 737 | -0.454 | 1.807 | 2.306 |
| 88 | 737 | -0.432 | 1.811 | 2.314 |
| 89 | 737 | -0.401 | 1.808 | 2.308 |
| 90 | 737 | -0.395 | 1.814 | 2.322 |
| 91 | 737 | -0.393 | 1.819 | 2.342 |
| 92 | 737 | -0.389 | 1.822 | 2.374 |
| 93 | 737 | -0.407 | 1.837 | 2.412 |
| 94 | 737 | -0.411 | 1.845 | 2.427 |
| 95 | 737 | -0.403 | 1.844 | 2.444 |
| 96 | 737 | -0.407 | 1.849 | 2.452 |

## Forecast-error autocorrelation

Pooled correlation after centring each lead across forecasts. Simulations use 10,000 traces, seed 0; these are dependence diagnostics, not extra independent weather observations.

| Lag h | Real | Original 24h blocks | Whole 96h trace |
| --- | --- | --- | --- |
| 1 | 0.8711 | 0.8460 | 0.8718 |
| 2 | 0.7426 | 0.7005 | 0.7433 |
| 3 | 0.6347 | 0.5822 | 0.6352 |
| 6 | 0.3909 | 0.3292 | 0.3923 |
| 12 | 0.1648 | 0.1105 | 0.1667 |
| 24 | 0.0443 | -0.0014 | 0.0408 |
| 48 | -0.1016 | -0.0051 | -0.0997 |
| 72 | 0.1197 | -0.0028 | 0.1274 |

| Adjacent leads | Real | 24h blocks | 96h trace |
| --- | --- | --- | --- |
| 12 → 13 | 0.7960 | 0.8029 | 0.8166 |
| 24 → 25 | 0.8337 | -0.0026 | 0.8411 |
| 36 → 37 | 0.8318 | 0.8243 | 0.8210 |
| 48 → 49 | 0.8453 | 0.0287 | 0.8388 |
| 60 → 61 | 0.8832 | 0.8798 | 0.8809 |
| 72 → 73 | 0.8850 | -0.0070 | 0.8856 |
| 84 → 85 | 0.9179 | 0.9194 | 0.9137 |

## Replay coverage

| Month | Windows | Distinct fetches | With solar |
| --- | --- | --- | --- |
| 2026-05 | 24 | 2 | 24 |
| 2026-06 | 24 | 2 | 24 |
| 2026-07 | 24 | 2 | 24 |
| 2026-08 | 471 | 48 | 0 |
| 2026-09 | 194 | 27 | 120 |

## Archived solar-field error (kWh/m²)

110 windows with complete observed solar horizons; zero observed radiation in 35.8% of hours. MAE on positive-observed-radiation hours: 0.07991 kWh/m².

| Lead hours | Forecast-hour pairs | Bias | MAE |
| --- | --- | --- | --- |
| 1–24 | 2640 | -0.01803 | 0.04783 |
| 25–48 | 2640 | -0.00132 | 0.04418 |
| 49–72 | 2640 | -0.00009 | 0.05606 |
| 73–96 | 2640 | -0.00842 | 0.05719 |

Overlap diagnostics: `{"windows": 737, "first": "2026-05-01 19:00:00+00:00", "last": "2026-09-10 11:00:00+00:00", "unique_targets": 1624, "total_targets": 70752, "adjacent_overlap_fraction": 0.9918478260869565, "median_spacing_hours": 1.0, "nonoverlapping_windows": 16, "fetches": 81}`
