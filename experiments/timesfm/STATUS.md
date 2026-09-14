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

Two constructed series survive and should be named as such rather than pretended away:
`t_catchment` is the plain mean of four air-temperature stations, `rain_catchment_24h` a
24-hour rolling sum. Both are aggregations rather than fitted weightings, and both were
measured as helping. The cleaner alternative — hand over the four raw stations and let the
model weight them — is a covariate-selection question and is on the open list below.

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
3. **Solar radiation.** One station, chosen for coverage and for representing the
   catchment, tested inside the full combination rather than alone — the rain result says
   testing a covariate in isolation can give the wrong answer. *Data being refetched.*
4. **The raw catchment stations** instead of their mean, letting the model do the
   weighting. Cheap, and it removes one of the two constructed series.
5. **The tributary and canal gauges** fetched after reading the hydrology — Loisach-Isar
   canal, Jachen, Walchen, Gaißach, Ellbach, Sylvenstein outflow — none yet tested.

## The caveat that applies to every number here

Every historical sweep uses **observed** weather over the horizon, because archived DWD
forecasts only exist from 2026-05. That makes them oracle runs. exp4 measured the cost of
the substitution on the windows where both exist — 0.285 against 0.290 MAE — so it is
small and known for air temperature, and it is **worse for rain**, whose forecasts are
markedly poorer than temperature forecasts. Read any rain gain as an upper bound.
