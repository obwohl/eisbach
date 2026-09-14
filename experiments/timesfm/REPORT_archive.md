# Covariate archive audit

Generated 2026-09-14 20:03:53.590690+00:00. All times UTC.

## Coverage

| Series | First measurement | Last measurement | Measured / grid hours | Gaps | Longest h |
|---|---|---|---:|---:|---:|
| eisbach | 2004-03-02 00:00:00+00:00 | 2026-09-14 19:00:00+00:00 | 124384 / 197565 | 83 | 55560 |
| isar_toelz | 2007-11-20 15:00:00+00:00 | 2026-09-14 13:00:00+00:00 | 164382 / 164973 | 221 | 27 |
| loisach_beuerberg | 1980-11-01 07:00:00+00:00 | 2026-09-14 19:00:00+00:00 | 171474 / 402093 | 9973 | 314 |
| airtemp | 2010-01-01 00:00:00+00:00 | 2026-09-14 19:00:00+00:00 | 146320 / 146420 | 7 | 69 |
| t_catchment | 2010-01-01 00:00:00+00:00 | 2026-09-14 19:00:00+00:00 | 146239 / 146420 | 11 | 69 |
| rain_toelz | 2010-01-01 00:00:00+00:00 | 2026-09-13 23:00:00+00:00 | 146041 / 146420 | 11 | 185 |
| rain_lenggries | 2010-01-01 00:00:00+00:00 | 2026-09-13 23:00:00+00:00 | 144382 / 146420 | 26 | 585 |
| rain_kochel | 2010-01-01 00:00:00+00:00 | 2026-09-13 23:00:00+00:00 | 141365 / 146420 | 57 | 1674 |
| rain_garmisch | 2010-01-01 00:00:00+00:00 | 2026-09-14 19:00:00+00:00 | 146189 / 146420 | 17 | 68 |
| solar_hohenpeissenberg | 2010-04-21 01:00:00+00:00 | 2026-09-14 19:00:00+00:00 | 143100 / 146420 | 34 | 2641 |
| pressure | 2010-01-01 00:00:00+00:00 | 2026-09-14 19:00:00+00:00 | 146333 / 146420 | 9 | 69 |

Full per-series ranges and counts: [summary.csv](../../data/archive/covariates/audit/summary.csv).
Every suspicious interval, inclusive endpoints, extrema and adjacent values: [findings.csv](../../data/archive/covariates/audit/findings.csv).
Every gap, including leading/trailing non-reporting intervals: [gaps.csv](../../data/archive/covariates/audit/gaps.csv).

These CSV annexes are an integral part of this report; no finding is sampled or discarded.

## Gap frequencies

| Series | 1h | 2–6h | 7–24h | 25–168h | >168h |
|---|---:|---:|---:|---:|---:|
| airtemp | 1 | 4 | 1 | 1 | 0 |
| eisbach | 42 | 21 | 12 | 6 | 2 |
| isar_toelz | 153 | 48 | 17 | 3 | 0 |
| loisach_beuerberg | 28 | 28 | 9873 | 35 | 9 |
| pressure | 3 | 5 | 0 | 1 | 0 |
| rain_garmisch | 4 | 5 | 6 | 2 | 0 |
| rain_kochel | 21 | 15 | 7 | 7 | 7 |
| rain_lenggries | 1 | 4 | 11 | 6 | 4 |
| rain_toelz | 2 | 2 | 5 | 1 | 1 |
| solar_hohenpeissenberg | 11 | 11 | 7 | 2 | 3 |
| t_catchment | 0 | 5 | 3 | 3 | 0 |

## Policy (applies to backfill and live reads)

| Series | Source | Unit | Review range | Jump / h | Hampel floor | Fill internal gap ≤ h |
|---|---|---|---|---:|---:|---:|
| eisbach | muenchen-himmelreichbruecke-16515005 | degC | (-5, 40) | 2 | 2 | 1 |
| isar_toelz | bad-toelz-b472-16003207 | degC | (-5, 40) | 2 | 2 | 1 |
| loisach_beuerberg | beuerberg-16408504 | degC | (-5, 40) | 2 | 2 | 1 |
| airtemp | 03379 | degC | (-50, 50) | 10 | 6 | 1 |
| t_catchment | 01550 | degC | (-50, 50) | 10 | 6 | 1 |
| rain_toelz | 05262 | mm | (0, 200) | 30 | 10 | 0 |
| rain_lenggries | 06257 | mm | (0, 200) | 30 | 10 | 0 |
| rain_kochel | 06342 | mm | (0, 200) | 30 | 10 | 0 |
| rain_garmisch | 01550 | mm | (0, 200) | 30 | 10 | 0 |
| solar_hohenpeissenberg | 02290 | kWh/m2 | (0, 1.5) | 0.8 | 0.8 | 0 |
| pressure | 03379 | hPa | (850, 1100) | 10 | 8 | 1 |

## Interpretation and decisions

No historical source value has been deleted or replaced by an interpolated value.
`raw/` contains the original compressed HTTP responses, including duplicates and bad
samples. `hourly/` contains unfiltered hourly means for GKD (with sample counts and
raw extrema) and Bright Sky's hourly values. Each precipitation/solar value is already
an hourly accumulation; it is neither averaged across stations nor summed twice.
Missing precipitation is never turned into zero. Pressure is the eleventh series:
it was already a production model channel and therefore also needed caching and audit.

Coverage means **what the supplied source actually returns**, not an assertion that a
station continuously measured from its inauguration. GKD station metadata starts
Eisbach in 2004, Bad Tölz in 2007, Beuerberg in 1980. Every calendar year from those dates
was requested, including empty years, and recorded in `manifest.json`. GKD chunks use
local January 1–December 31, never UTC December 31 23:00 (which is January 1 in Bavaria).
Bright Sky's source catalogue starts these stations in 2010. Forecast sources are
excluded from observations even if their timestamps are in the past. Every returned
`source_id` is checked against the requested `dwd_station_id`; there is no coordinate
fallback. Source IDs and the original source catalogue remain recoverable in the payloads.

The Eisbach has only 54 measured hours in March 2004 before a gap of 55,560 hours;
the useful later record starts July 2010 and also has a 16,919-hour gap in 2012–2014.
Those are source-availability gaps, **not established multi-year river drainages**.
Beuerberg's 1980–2007 record predominantly contains one observation per day, often
07:00 UTC: the roughly 23-hour gaps are a sampling pattern, not thousands of proven
sensor failures. Regular hourly availability starts in 2008. See `audit/yearly.csv`
for annual counts and median spacing. Keeping the sparse earlier data is useful;
pretending it is hourly by interpolating whole days would not be.

### Instrument fault: confirmed episode, retained as evidence

At **2026-09-09 08:00 UTC** (10:00 local), the original Eisbach samples are
154.4, 71.7, 18.0 and 19.0 °C at :00, :15, :30 and :45. Thus the raw hourly mean is
**65.775 °C**, between hourly neighbours 19.1333 and 19.025 °C. The 154.4 °C value is
already confirmed/retracted in the existing observation archive; the adjacent 71.7 °C
sample is an additional impossible reading in the same episode. Absolute range,
subhourly extrema, Hampel and the up/down jump all identify this episode. The return
jump at 09:00 UTC is a transition back to normal, not a second bad hourly reading.
No existing archive partition was rewritten to fix this a second time.

For model reads, the whole contaminated hour is recorded as `never_measured`, with
its raw value, reason and policy version in `decisions/eisbach/2026-09.csv`. The raw
mean and original samples remain in their respective stores. A one-hour interpolation
can then be made in memory for model input; that interpolation never becomes a measurement.

### Physical events and findings that remain undecidable

A steady water temperature over 24–70 hours is not by itself a broken instrument:
rounding to 0.1 °C and slowly varying water can create a plateau. Every plateau is
listed, separately for each constant value, with adjacent values in `findings.csv`.
None is automatically removed. Long zero-rain runs are physically plausible dry
spells, explicitly classified as such; a rain gauge stuck at zero remains possible
without independent evidence. No ≥24-hour solar plateau was detected in this build.
Hampel flags on rain can be real concentrated rainfall, so they remain review-only.

Outside the confirmed Eisbach episode, water/air jump and Hampel cases are explicitly
undecided. For example, Bad Tölz 2020-06-02 09:00 UTC has missing immediate neighbours;
a neighbourhood statistic with that context cannot prove a fault. Air-temperature
changes that persist into the next hour can be fronts or thunderstorm outflow. The
report retains their actual before/after values rather than declaring them faulty.
There are no resolved UTC duplicate timestamps in the final hourly store; duplicate
and conflicting-duplicate detectors run regardless. Raw DST ambiguity is separate.

Bachauskehr is a real hydrological intervention and must never be repaired as a
sensor spike. There is independent evidence for **October 2025**: the city documents
closing the Fabrikbach inflow and reopening it on October 31, including the Eisbach.
It describes spring/autumn work with residual water, so neither “always winter” nor
“always completely dry” is a safe detection rule. [City announcement, 17 October 2025](https://ru.muenchen.de/2025/198/Bachauskehr-in-der-Grossen-Isar-und-am-Fabrikbach-120796).
The city subsequently confirms completion on October 31 and explains that refilling
takes time. [City follow-up, 4 November 2025](https://ru.muenchen.de/2025/210/Ueberpruefung-der-Eisbachwelle-121084).

The temperature record does **not** show a long missing interval during that documented
2025 intervention. Temperature availability cannot establish flow. The two missing
hours on October 26 are unresolved local DST timestamps, not evidence of drainage.
Discharge and water level were not among the requested series; without those records
or a dated operating log, the other water gaps cannot be assigned confidently to
cleaning, sensor exposure, maintenance or transmission failure. Zero flow/level would
be physically allowed, not grounds to reject a temperature. No calendar-based cleanup
or zero-value replacement is implemented.

### Robust checks and their limits

Water uses the existing `data._spikes` two-pass neighbourhood method: three hours on
either side, centre excluded, 12 × 1.4826 × residual MAD, floor 2 °C, absolute bounds
−5…40 °C. This preserves the calibrated Eisbach method as a **detector**. Transferring
its calibration to upstream gauges is not evidence that every flagged upstream change
is false, hence neighbourhood/jump flags no longer automatically erase model observations.
Hard temperature-range failures are the only automatic rejection rule. The whole window
is still unusable if its missing/rejected values cannot be filled by the rule below.

Other variables use a local six-neighbour Hampel median/MAD, also 12 robust standard
deviations with the physical floors in the table. Higher floors are necessary because
air fronts, rain pulses and sunrise do not have a river's smoothness; MAD is often zero
in dry spells. Jump thresholds are review triggers, not claimed physical impossibility.
The range limits for precipitation, radiation and pressure are deliberately broad review
limits, not a licence to discard an extreme event. Non-finite model inputs are refused.
All flags are retrospective, with incomplete neighbourhoods at edges; later observations
can revise a suspicion. Neither flags nor retrospective gap fills claim information
was available in an earlier forecast.

GKD wall-clock times that cannot be assigned uniquely to UTC are kept in the original
HTML and enumerated in `ingestion/<series>/<year>.json`. The UTC projection leaves
both potential autumn hours missing; it does not guess an offset or shift a nonexistent
spring timestamp onto a real observation. This conservative choice can cost two hours
per autumn transition and make a crossing model window unusable. It avoids the older
experiment code's deduplication before localization. Original responses permit a later,
reviewed reconstruction if GKD supplies unambiguous offsets/order evidence.

## Gap rules and window eligibility

For **each of the three water temperatures, both air temperatures and pressure**,
fill only an entire, interior gap of **one hour**, by linear interpolation between real
neighbours. Never extrapolate an edge, and never fill the first hour of a longer gap.
Any longer gap makes a model window using that series unusable. This is a conservative
starting rule, not an empirically validated optimum for each upstream gauge.

For **each of the four rain series and solar**, fill **nothing** automatically. Even
one missing hour makes a window requiring that series unusable. A missing rainfall
accumulation could contain the entire storm; zero or linear interpolation is not
supported. For solar, a future rule could fill a confirmed astronomical night with
zero, but station geometry and interval convention would first need verification;
this build does not infer night from neighbouring zeros.

`load_window(names, start, end)` implements these rules and raises on an unusable
window. For production's three existing channels, `assemble_long_frame` uses the same
one-hour rule; the model wrapper no longer forward/back-fills residual gaps and rejects
non-finite or non-hourly context. Old distant gaps outside the model context do not
invalidate the current forecast. Short interpolation is in memory only; observation
and raw stores remain unfilled. Never train across a 55-day gap by interpolation.

Leading/trailing gaps and gaps within source support are separate in `gaps.csv`.
`gap_histogram.csv` gives frequency by length, `yearly.csv` exposes sparse sampling,
and `summary.csv` gives the complete span and extrema. Gaps include explicit nulls,
absent timestamps and unresolved DST projection; the report does not silently conflate
all three with instrument failure. Ingestion files provide the DST/raw evidence.

## Running and maintaining the archive

- Build/resume: `python experiments/timesfm/build_archive.py`.
- Recompute the complete report without network: `python experiments/timesfm/build_archive.py --report-only`.
- Production: `python main.py` uses `prepare_live`, station-specific cursors and a
  72-hour overlap for delayed measurements, splitting longer catch-up periods by year.
  Failed HTTP/parse/source-validation requests raise; they never advance that station's
  cursor. The one forecast request covers now through eight days ahead, not a year of history.
- Every successful live response is preserved before quality/model execution. Recent
  suspicions and missing hours are written to `quality/`; rejected temperature hours
  also have explicit `never_measured` decisions. All eleven series are updated even
  though the trained channel order remains water, air+96h, pressure+96h.
- The workflow commits archive changes even if subsequent model validation fails.
  Forecast, weather, observations and verification stores were not used as backfill
  destinations. Production observation writes preserve existing water values and retractions, while
  allowing late observed weather to fill missing fields at existing hours. Already
  recorded weather values remain unchanged; new water observations use hourly means.
- Single-writer operation follows the existing archive pattern and the workflow's
  concurrency group. Writes are atomic per partition, not a transaction over all stations.
  A resumed run can therefore reuse successful earlier station updates.
- The 72-hour overlap does not discover arbitrary upstream revisions years later.
  Such revisions require a deliberate backfill/refresh and review of new raw responses.

## Effect on the rain selection

The pinned series must replace the old coordinate-derived experiment input before
claiming the earlier rain selection still holds. On the locally available experiment
cache, **2019-01-01 00:00 through 2026-09-14 11:00 UTC**, 1,328 hours have an old
`rain_lenggries` value but no pinned 06257 observation; all jointly present values agree
within 1e-6 mm. This comparison spans the whole cache, not the user's particular
8,760-hour window with 575 substituted hours. The old cache contains no per-row source
metadata, so this comparison alone cannot attribute every extra hour to Kreuth.

The four pinned rain series have distinct station provenance; sharing legitimate zero
hours is not proof of duplication. Re-run the rain selection on common eligible
anchors, report how many windows the strict missingness rule excludes, and preserve
the previous result as a result on the old data. No new selection or model-training
claim is made by this archive task.

## Validation recorded for this delivery

Final validation: `pytest -q` — **214 passed, 2 skipped**; `ruff check .` — **all checks passed**.
All 226 manifest entries passed SHA-256 verification. The archive contains 1,620,209
non-null hourly values, 5,328 flagged intervals and 10,449 gap intervals. The four
pre-existing archive stores have no diff against the checked-out source branch. Tests cover strict station identity, forecast exclusion,
subhourly extreme preservation, conflicting duplicates, DST, tombstones, month
boundaries, short versus long/edge gaps, physical zeros, bounded refreshes and failed
cursors. A real incremental refresh plus an in-memory model run produced a 96-hour,
21-quantile-column forecast, with median water temperature 16.535–18.484 °C. This smoke
test did not publish forecasts or modify the four existing stores.
