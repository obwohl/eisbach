# Where this stands

One page. What is settled, what is not, and what is deliberately not being attempted.

## The rule for keeping a covariate

A covariate stays if its **paired** effect trends in our favour and the interval does not
show it doing harm. Significance is a bonus, not a gate: something that helps a little,
rarely, and does not hurt is worth keeping. Something whose mean is worse, or whose
interval clearly shows harm, goes.

Every number below is a paired comparison on identical windows. Unpaired intervals here
are roughly ten times too wide to see anything, because the windows differ from each other
far more than the variants do.

## Out of scope, by decision

**No hand-built features.** No discharge-weighted mixtures, no invented factors, no
second model on top. If TimesFM cannot use a series, that is an answer, not an invitation
to do its job by hand. `exp10_hydrology.py` and its `t_mix` are deleted for exactly this
reason.

What remains is **covariate selection** — which raw series to hand over, and as what.

One constructed series survives and should be named as such rather than pretended away:
`rain_catchment_24h`, the 24-hour rolling sum of a four-station rain mean. Those four rain
series are genuinely four measurements — pairwise r between 0.14 and 0.53 on wet hours —
so the mean really is an aggregation, and it is on the open list to try the raw four
instead.

`t_catchment` was on this list too, described as the mean of four air-temperature
stations. **It is not.** Bright Sky answers for a *coordinate*, from whichever station is
nearest and reporting, and all four of my southern coordinates resolve to the same
station: pairwise mean absolute difference 0.002–0.004 °C, r = 0.9999. `t_catchment` is
one southern station, averaged with three copies of itself. That makes it a raw series
rather than a constructed one — but I had been describing it wrongly, and the same lookup
is why `solar_*` had to be rebuilt by station id (see below).

## Settled: method

| | | evidence |
|---|---|---|
| Context | **8760 h (one year)** | The dominant lever, measured twice independently. exp1: a year beats 384 h by 12 %. exp11, different windows, machine and backend: a year beats 160 days by 15.9 % MAE and 16.5 % CRPS. |
| Resolution | **hourly** | exp11: at matched context a 15-minute grid gives −0.6 % MAE and −0.4 % CRPS, intervals containing zero in all seven lead buckets. It also cannot see a year, since TimesFM's 15 360 steps are 160 days at a quarter hour. |
| Horizon | 96 h, scored in 6-hourly buckets over the first day | Upstream gauges are spent once the horizon outruns the travel time; coarse buckets hid that. |

## Settled: covariates to keep

| series | kind | effect | where measured |
|---|---|---|---|
| `airtemp` (Munich) | known-future | water-only 0.684 → 0.380 MAE | exp6 |
| `t_catchment` | known-future | −4.1 % MAE, −4.3 % CRPS, significant | exp8, decomposed |
| | | *One southern station despite the name — see above. Munich's own air differs from it by 2.7 °C on average, so the two are not redundant.* |
| `rain_catchment_24h` | known-future | −3.0 % MAE, −3.1 % CRPS, significant, **on top of everything else** | exp8, decomposed |

Rain is the instructive one: alone against the baseline it was never significant, and it
earns its place only once the model has the air and discharge information to read it
against. A covariate that is useless in isolation can still be worth having.

## Settled: covariates to drop

| series | why |
|---|---|
| `isar_muenchen` | r = 0.999 with the target and **+1.7 % MAE**. A near-duplicate carries no lead time and still costs a variate slot. |
| `schwabinger_bach` | Branches off the Eisbach **below** the gauge. Downstream of the target, so it cannot predict it — and its 2025 gap cost the whole recent record. |
| `isar_mittenwald` | Sits above the Krün diversion, where most of the Isar leaves for the Walchensee. Worst temperature covariate of all at +1.1 %. |
| 15-minute resolution | See above. |

## Open, in order

1. **The upstream chain.** `isar_lenggries`, `isar_toelz` and their discharges were
   significant as a lone addition (exp6 −4.8 %/−4.2 %, replicated in exp7), and are **not
   distinguishable from zero inside the full combination** (−2.2 %, CI [−6.7 %, +2.3 %]).
   Some of what they carry is evidently already in the catchment air temperature.
   `exp12_pairs.py` is separating this station by station: temperature alone, discharge
   alone, and the pair — because a temperature may be unreadable without the volume behind
   it. *Running.*
2. **Covariate or target.** TimesFM 3.0 is natively multivariate, so the upstream gauges
   can be forecast alongside the Eisbach instead of handed over as covariates. Only the
   Eisbach is scored. `exp9_targets.py`. *Queued behind exp12.*
3. **Solar radiation.** `exp13_solar.py`, *queued behind exp12*. Tested inside the full
   combination rather than alone, because the rain result says isolation gives the wrong
   answer. Two candidate stations, because reliability and catchment membership point in
   opposite directions and the data can settle it:

   | station | coverage 2019→now | thin months | in the catchment? |
   |---|---|---|---|
   | Hohenpeißenberg (02290), 977 m | 99.9 % | **0 of 93** | no — Ammer watershed, joins the Isar at Moosburg, below Munich |
   | Garmisch-Partenkirchen (01550), 719 m | 97.3 % | 4 of 93 | yes — Loisach valley |
   | ~~Kreuth (02738)~~, 776 m | 59.3 % | 37 of 73 | no — Tegernsee/Mangfall, drains to the Inn |

   Kreuth is excluded on reliability, and it matters that it is: it is the nearest
   radiation station to Lenggries, so it is what a coordinate lookup reaches for after it
   came online in September 2020. Mittenwald, Jachenau-Tannern and Holzkirchen carry
   radiation in under 0.5 % of hours — the DWD's radiation network is simply sparse here.

   **The old solar columns were never six stations.** `weather_south.csv` has
   `solar_toelz`, `solar_lenggries`, `solar_kochel` and `solar_garmisch` bit-identical
   across the entire record, and the one series behind them switches source in September
   2020, from Garmisch to Kreuth — a seam in the middle of a covariate. `build_solar.py`
   now fetches by DWD station id, so the source cannot move.
4. **The raw catchment rain stations** instead of their mean, letting the model do the
   weighting. Cheap, and it removes the last constructed series.
5. **The tributary and canal gauges** fetched after reading the hydrology — Loisach-Isar
   canal, Jachen, Walchen, Gaißach, Ellbach, Sylvenstein outflow — none yet tested.

## The caveat that applies to every number here

Every historical sweep uses **observed** weather over the horizon, because archived DWD
forecasts only exist from 2026-05. That makes them oracle runs. exp4 measured the cost of
the substitution on the windows where both exist — 0.285 against 0.290 MAE — so it is
small and known for air temperature, and it is **worse for rain**, whose forecasts are
markedly poorer than temperature forecasts. Read any rain gain as an upper bound.
