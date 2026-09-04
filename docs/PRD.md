# Eisbach Forecast — product requirements

What this project is for, what it now guarantees, and what is deliberately still open.

Written to be read in order by someone picking the work up. It states requirements and
the evidence behind them; it is not a changelog. For how the model itself works, see
[`docs/model.md`](model.md).

## The product

A 96-hour probabilistic forecast of the Eisbach's water temperature, regenerated three
times a day and published to GitHub Pages. One audience: a person deciding whether to
swim. The forecast has to be honest about its own uncertainty, and it has to keep
running unattended without anyone watching it.

Three properties follow from that, and everything below serves one of them.

**P1 — A wrong forecast must fail, not publish.** The pipeline runs unattended. Nobody
checks the plot before it goes out, so the run itself has to refuse to publish something
broken.

**P2 — A backtest must never flatter the model.** Three of the four plotted lines are
backtests, and they are the only thing a reader has to judge the forecast by. A backtest
that saw weather nobody could have known is a lie about skill unless it is labelled as
one.

**P3 — The archive is the only record, and it cannot be rebuilt.** GKD serves a rolling
window; historical DWD forecasts are a paid product. Anything not captured is gone.

## What the forecast is currently worth

Measured on 59 verifiable live runs of checkpoint `1c7a531768d8` (targets 2026-08-04 →
09-04), joined against archived observations. Uncertainty is a block bootstrap over whole
96-hour runs — rows within a run are strongly autocorrelated, and row-level intervals
would be roughly ten times too narrow.

| lead | n | MAE | RMSE | bias | CRPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–96 h | 5 228 | **0.411** | 0.530 | −0.018 | 0.289 |
| 0–24 h | 1 409 | 0.324 | 0.424 | −0.006 | 0.225 |
| 72–96 h | 1 198 | 0.431 | 0.569 | +0.005 | 0.323 |

95 % CI over all leads: MAE [0.376, 0.448], CRPS [0.269, 0.310], bias [−0.109, +0.074].

Against persistence (the observation at the anchor, held flat): **+32 % on MAE**, roughly
uniform across leads — +39 % at 0–24 h, +32 % at 72–96 h. This is the project's only
defensible skill claim, and it is an honest one: live forecasts, scored against what
actually happened.

**Two numbers that must not be quoted.** Pooled all-era figures (MAE 0.627) blend two
different systems. And the apparent improvement from MAE 0.786 (legacy) to 0.411
(current) is **not** evidence of progress: the eras do not overlap in time — legacy ends
2026-08-01, current begins 2026-08-04 — so code change is perfectly confounded with
season. By target month the bias runs June −0.94, July −0.36, August −0.05, September
+0.08. No reprocessing can separate these.

## Calibration: known, quantified, deliberately not corrected

The median is well calibrated (49.5 % of observations below q0.5; bias CI straddles
zero). The spread is not: PI50 captures 63.3 %, PI90 captures 97.8 %, and only 0.09 % of
observations have ever fallen outside q0.01–q0.99. Symmetric at every level, so it is a
pure scale problem — no shift, no skew.

**Requirement: do not ship a post-hoc calibration yet.** This is a standing decision, not
an oversight. Fitting on a 60/40 split of the live archive and testing out of sample:

| correction fitted on the early half | test pinball | vs. raw 0.1022 |
| --- | ---: | --- |
| per-horizon median bias | 0.1259 | **23 % worse** |
| rolling 7 / 14 / 28-day bias | 0.1058 / 0.1055 / 0.1049 | all worse |

The optimal width factor is 1.03 on the early half and 0.70 on the late half; applied
across, both degrade. The cold bias is a **seasonal regime effect** — the model lags the
water through the June/July warming and loses that lag once the season plateaus — not a
fixable offset. Revisit after a full seasonal cycle, and fit conditionally on regime
rather than on lead alone.

## Delivered

Stated as guarantees, so a future change knows what it must not break.

**Validation (P1).** `eisbach/validate.py` fails the run on an implausible range, a NaN
quantile, a forecast index that overlaps real measurements (which would mean the input
was not truncated and the model saw its own answer), or a backtest whose 1–99 % band
covers less than 80 % of observations. 28 tests.

**Backtest honesty (P2).** Precedence `live > replay > oracle` is enforced on write, so a
regenerable row can never overwrite a genuine one. Oracle backtests are drawn dashed and
labelled wherever they appear. The weather-snapshot lookup is one-sided by construction:
a snapshot issued after the reference time is a leak and is never eligible.

**Archive durability (P3).** Snapshots store only rows a replay can reach, plus one
eligibility window of margin derived from `SNAPSHOT_MAX_AGE_HOURS` so read and write
cannot drift apart. Measured: 108 377 → 24 531 bytes per snapshot, **−77.4 %**; the
weather store drops from ~113 to ~26 MiB/yr. Observed air temperature and pressure are
archived alongside water temperature, so forecast error can be split into "wrong about
the river" and "wrong about the air". `issued_at` is refused blank at the door, because
precedence breaks same-kind ties on it and a blank loses every tie it enters. Snapshots
carry a `schema_version`, stamped only when the frame really is in that shape. A snapshot
with no `timestamp` column is refused outright — that is the exact failure that cost us
seven irrecoverable snapshots. 31 tests.

**Covariate capture (P3).** `sunshine`, `solar`, `cloud_cover`, `relative_humidity`,
`wind_speed` and `dew_point` are archived again. Archival only — `CHANNELS` is the
trained order and is untouched. Sums use `min_count=1`: "DWD reported no sunshine value"
must not become "zero sunshine" in a store kept as evidence.

**Operational.** Workflow permissions are per job; the archive push rebases and retries;
a persistent failure comments on the open issue instead of opening three a day; CI skips
archive-only commits and tests both Python versions the project claims to support.

**Naming.** The research tree is `research/`, not `archive/`. Three things were called
"archive", and the gap between deleting the disposable one and deleting the irreplaceable
one was one keystroke wide.

Existing archive partitions were never rewritten, recompacted or migrated — including the
1 824 rows with a blank `issued_at` and the 46 % with a blank `model_id`.

## Open requirements

Ordered by what unblocks the most. Each states why it is not already done.

### R1 — The validation gate must require data to be present, not merely correct

`_check_ranges` iterates the columns that *exist*, so a forecast carrying no
`wassertemp_q*` columns at all passes cleanly — the precise case the module's own
docstring names as its reason to exist. Three siblings: an empty backtest frame passes
and is indistinguishable from one predating the observations; missing observations
disable the leak check and every coverage check at once; and a backtest lacking
`q0.01`/`q0.99` escapes as `KeyError` rather than `ImplausibleForecast`.

All four have one shape: **the gate degrades to a no-op when data is missing, and only
fires when data is present-and-wrong.** Found while writing the tests, left unfixed
because the fix needs a decision this document should make: what exactly must be present?
Proposed answer — all three channels × all seven quantiles × `config.horizon` rows, and
at least one backtest with real overlap. That is a contract, and contracts belong in a
PRD rather than in a test written after the fact.

*Acceptance:* a forecast missing any required column or row count fails the run;
`_coverage` raises `ImplausibleForecast`, never `KeyError`; a run with no usable
observations fails rather than reporting success.

### R2 — Persist a verification table

Every number in this document was recomputed from scratch. Nothing accumulates, so model
drift is invisible and there is no series to fit a calibration against. `validate.py`
already computes an interval coverage per run; the cheap version is to extend it to
per-lead MAE, CRPS and PIT and store it as observations close.

This is the prerequisite for ever revisiting the calibration decision above, and for
noticing a regression that is not large enough to trip the plausibility gate.

*Acceptance:* a store under `data/archive/` with one row per (run, lead bucket) carrying
MAE, CRPS, PIT and coverage; populated retroactively from the existing archive; a
documented way to read it.

### R3 — Record input completeness per run

Nothing records how many gauge readings were missing, how stale the anchor was against
wall clock, how many hours were interpolated, or whether the weather fetch came up short.
The archive contains one non-causal row and a maximum issue lag of 1.6 h, which says runs
do sometimes limp — and there is currently no way to exclude a degraded run from
verification rather than silently averaging it in.

*Acceptance:* `n_missing_input_hours`, `anchor_age_hours` and `weather_rows_fetched` on
every archived forecast; R2's table can filter on them.

### R4 — Stop storing covariate quantiles nothing reads

`airtemp_96_q*` and `pressure_96_q*` are roughly two thirds of the 9.4 MB forecast store
and are, as far as anyone has found, never read back. Deliberately left out of the
archive work because it changes the schema of the store everything else reads, and it
deserves its own change with its own tests.

*Acceptance:* new rows carry water-temperature quantiles only, or the covariates move to
their own store; readers tolerate both shapes; no existing partition rewritten.

### R5 — Pin the dependencies

`forecast.yml` installs against `pandas>=2.2`, `torch>=2.3`, `matplotlib>=3.8`,
`scipy>=1.10` with no lock. A breaking upstream release takes the forecast down at 04:00
UTC, and CI resolves independently at a different time — so CI can be green while the
scheduled run is broken. Not done because it needs a tooling decision (uv lock,
pip-tools constraints, or upper bounds in `pyproject.toml`) and a resolution from the
real runner, not from a development container.

*Acceptance:* both workflows install from the same pinned set; refreshing it is a
deliberate, reviewable commit.

### R6 — Reconnect the covariate pathway

The largest available gain, and the only one that needs a training run. Three independent
measurements say the weather covariates are barely connected to the water head:

- The channel mask never lets air temperature reach water: `p > 0.5` in **0 of 222**
  historical windows (max 0.0088).
- RevIN's per-window normalisation makes every covariate affine-invariant. Shifting
  `airtemp_96` by +3 °C changes the water forecast by **1.9e-6**. A DWD forecast of 30 °C
  and one of 20 °C with the same diurnal shape are identical inputs.
- Replacing `pressure_96` with a constant moves the forecast by at most **0.018 °C** —
  and pressure is the *only* covariate the mask lets water attend to.

Air temperature does still matter (constant air moves the h=96 median by −1.24 °C), but
the traced route is the router selecting different experts, not the value reaching the
head.

Proposed changes at the next training run, in expected value order: normalise covariates
against a climatological location and scale so absolute level survives; give the water
head an unmasked or prior-forced path to `airtemp_96` — the checkpoint already carries
`channel_adjacency_prior` whose first row says "water may see everything", with
`use_channel_adjacency_prior: False`; and drop or replace `pressure_96`. R2 and R3 should
land first, so the result can be measured rather than asserted.

Four latent bugs will bite that run and are documented with evidence in
[`docs/model.md`](model.md): `input_scaling_uni` is silently ignored (Optuna tuned a
parameter with no effect), the mask diagonal is NaN and works only by accident,
`icdf(0.5)` survives only through float32 underflow, and the projection heads are bound
to channel slots with no name check.

## Constraints on any change here

- **`eisbach/model/duet/` stays byte-identical to upstream** so it remains diffable
  against `ts_proba_cuda@a8de694`. Upstream HEAD does not load this checkpoint. The four
  bugs above are notes for a training run, not patches for this repository.
- **`CHANNELS` / `SERIES_ORDER` is the trained order.** The projection heads are indexed
  positionally, so reordering produces plausible-looking nonsense rather than an error —
  air-temperature values in the water column, which pass the plausibility gate in a
  Munich summer.
- **`COVARIATE_SHIFT_HOURS` must equal the model horizon.** Changing one without the
  other silently removes the weather forecast the horizon depends on.
- **`data/archive/` is append-only.** Never rewrite, recompact or migrate a partition —
  not even to make a schema uniform.
- **Errors raise.** The predecessor caught everything, printed a message and exited 0, so
  failures were invisible to the caller.
- **Tests must not touch the network**, the model checkpoint excepted.

## Verification

For anything touching the pipeline: `pytest -q`, `ruff check .`, then `python main.py`
twice against a scratch archive root, confirming the second run's −96 h backtest resolves
as `live` or `replay` and not `oracle`. That one assertion exercises the archive write
path, the precedence rule, the snapshot lookup and the replay splice together.
