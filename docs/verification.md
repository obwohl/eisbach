# The verification store

How well the forecast has actually done, kept as a series rather than recomputed.

Every number in [`PRD.md`](PRD.md) used to be worked out from scratch against an archive
that only grows. That is fine as an analysis and useless as a way to notice that the
model got worse last Tuesday: nothing accumulated, so drift was invisible and there was
no series to fit a calibration against. This store is that series.

## What is in it

`data/archive/verification/YYYY-MM.csv`, partitioned by the month of the run's reference
time like every other store. One row per **(run, kind, lead bucket)**.

A run is scored **once its window has closed** — once the observation store reaches the
last hour the run forecast. Until then there is nothing to score; after that the answer
cannot change, which is what lets this store keep the append-only rule the rest of
`data/archive/` lives by. A row is written once and never restated, so re-running the
scorer over the whole archive costs a few seconds and changes nothing. Deleting a
partition is the deliberate way to ask for a recomputation.

It is also the one store here that is *derived*: forecasts plus observations rebuild it
exactly. A lost partition is a rerun, not a hole in the record.

| column | |
| --- | --- |
| `reference_time` | the run's anchor — its last real gauge reading |
| `kind` | `live`, `replay` or `oracle`. Never pool across it — see below |
| `model_id`, `code_version` | provenance of the forecast being scored, blank for runs that predate it being recorded |
| `lead_lo`, `lead_hi` | the lead bucket in hours, half-open as `(lo, hi]` |
| `n` | hours in the bucket that were measured, and therefore scored |
| `n_forecast` | hours the run predicted in the bucket. Below `n` never; above it exactly when the gauge had a gap — but see the caveat below |
| `mae`, `rmse`, `bias` | of the median, in °C. `bias` is forecast − observed, so negative is too cold |
| `crps` | continuous ranked probability score, in °C — see the caveat below |
| `mae_persistence` | the anchor observation held flat |
| `mae_diurnal` | the same hour on the last whole day the run had already seen |
| `pit_mean` | mean probability integral transform. 0.5 means the median is unbiased and says nothing about spread |
| `pit_le_q*` | fraction of observations at or below each predicted quantile — the PIT histogram at the seven levels the model emits |
| `scored_at`, `schema_version` | when it was scored, and in what shape |

## Reading it

```python
from eisbach.verification import read_scores, pool

scores = read_scores()          # honest kinds only, coverage materialised
pool(scores)                    # by era, kind and lead bucket — the safe default
pool(scores, by=["lead_lo"])    # across eras and kinds: know what you are mixing
pool(scores, by=[])             # one row for everything
```

`read_scores` adds `cov_50`, `cov_90` and `cov_98` — differences of the PIT knots, so
they are exact rather than re-derived. `pool` weights every row by the hours it covers
and knows that RMSE averages in squares; a plain `groupby().mean()` gets both wrong. It
groups by era and kind unless told otherwise, so the default cannot quietly produce the
two numbers the next section says must not be quoted.

`python -m eisbach.verification` scores whatever is newly scorable; `--report` prints the
pooled table. The scheduled workflow runs the first form before each forecast.

## Four ways to get a wrong number out of it

**Pooling across `kind`.** An oracle backtest saw the weather that actually occurred, so
its scores describe a system that has never run. `read_scores` returns `live` and
`replay` only unless you pass `kinds=None`, and if you do, keep the `kind` column in
whatever you report.

**Pooling across `model_id`.** The eras in this archive do not overlap in time, so a
difference between them is code change perfectly confounded with season. The legacy era
scores MAE 0.786 against the current 0.430; almost none of that gap is progress.

**Summing the buckets.** The four lead buckets are disjoint and there is deliberately no
pooled row beside them, because with one there the obvious `scores["mae"].mean()` would
count every observation twice. Use `pool`.

**Reading `crps` as a textbook CRPS.** It is twice the integral of the pinball loss over
the quantile level, trapezoidal across the seven levels the model reports — so the
integral runs over [0.01, 0.99] and omits the tails beyond. That understates the true
CRPS by a constant of the method, not of the forecast, so the series stays comparable
with itself but not necessarily with someone else's.

## What counts as a measurement

`assemble_long_frame` interpolates `wassertemp` across the hourly grid, because the model
must not be fed a hole. Until 2026-09-10 the observation store was written from that
filled series, so it holds values nobody read off an instrument — at least 0.5 % of rows
are provably interpolated (they sit off the gauge's own 0.1 °C grid), and the true share
is higher, because a fill that lands on the grid is indistinguishable from a reading. The
store has no hourly gaps at all over its whole span, which is the symptom rather than a
clean bill of health.

Runs are now archived from the raw gauge frame instead, so an hour the gauge never
sampled is absent rather than invented. Two consequences:

- `n` < `n_forecast` is a real signal **for runs scored from 2026-09-10 onwards** and is
  meaningless before it, where every hour was filled in before anything counted it;
- scores for earlier runs are computed partly against interpolated truth. The affected
  share is small and the interpolation is short-range, so the aggregates above stand —
  but a single run's score is not evidence about a single hour.

The archive is never rewritten, so the old rows keep whatever they hold. R3 is where a
per-run completeness record belongs.

## Two baselines, because one of them is a strawman

`mae_persistence` holds the anchor reading flat for four days. It is the standard naive
baseline and it is weak here: the Eisbach's largest signal at these leads is the daily
cycle, and a flat line is wrong by most of that cycle's amplitude before the forecast has
done anything at all.

`mae_diurnal` repeats the last whole day the run had already seen — "tomorrow looks like
yesterday". It reproduces the diurnal swing, so beating it means beating something worth
beating. It stays causal: the lag is the smallest whole number of days that lands at or
before the anchor, so a lead of 30 h reads 48 h back, not 24.

Both are stored, and the honest headline is the second one.

## What it currently says

66 live runs of checkpoint `1c7a531768d8`, anchored 2026-08-04 → 09-06, 6 336 scored
hours. Regenerate with `python -m eisbach.verification --report`.

| lead | MAE | RMSE | bias | CRPS | vs. flat | vs. diurnal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–24 h | 0.333 | 0.434 | −0.003 | 0.231 | +53 % | +31 % |
| 24–48 h | 0.425 | 0.528 | −0.015 | 0.290 | +53 % | +41 % |
| 48–72 h | 0.471 | 0.585 | +0.052 | 0.322 | +53 % | +39 % |
| 72–96 h | 0.465 | 0.598 | +0.128 | 0.336 | +53 % | +44 % |
| **0–96 h** | **0.423** | **0.540** | **+0.041** | **0.295** | **+53 %** | **+40 %** |

Skill against the diurnal baseline *rises* with lead, and against flat persistence it
holds level. Both are the shape to expect: the baselines decay at least as fast as the
model does. A skill that *fell* with lead would mean the baseline was not a baseline.

Replay backtests of the same checkpoint score MAE 0.441 against live's 0.423. Do not read
that gap as the price of an older weather snapshot. Precedence keeps one kind per
reference time, so the 43 replay runs and the 66 live ones are different runs on
different days rather than two readings of the same ones — the comparison is unadjusted,
and weather over different weeks would move it by more than 0.018 on its own. Separating
snapshot age from season would need the same anchor scored both ways, which the archive
deliberately does not keep.

Calibration is unchanged from what the PRD describes, and the spread columns are here so
that it can be watched rather than re-derived: PI50 covers 61 % against a nominal 50 %,
PI90 covers 97 % against 90 %, mean PIT 0.48. The median is honest and the intervals are
too wide at both levels — a scale problem, symmetric, and deliberately not corrected.
See the PRD for why fitting a correction made things worse out of sample.
