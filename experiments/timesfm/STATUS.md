# Where this stands

One page. What is settled, what is open, and what is deliberately not attempted.

## The rule for keeping a covariate

A covariate stays if its **paired** effect trends in our favour and the interval does not
show it doing harm. Significance is a bonus, not a gate. But a year with demonstrated
harm refutes "does not hurt" — that is what removed Sylvenstein.

Every number below is paired on identical windows. Unpaired intervals here are about ten
times too wide to see anything, because the windows differ from each other far more than
the variants do.

## Out of scope, by decision

**No hand-built features.** No discharge-weighted mixtures, no invented factors, no second
model on top. The scope is **which raw series to hand over, and as what**.

This turned out to be the right call empirically as well as on principle. Both places
where a construction was tested against its own ingredients, the raw series won or tied:
the four raw rain gauges against their 24-hour mean-sum, and raw hourly radiation against
its 24-hour sum. Nothing in the set below is constructed any more.

## Settled: method

| | | evidence |
|---|---|---|
| Context | **8760 h (one year)** | The dominant lever, measured three times. exp1: a year beats 384 h by 12 %. exp11 on another machine and backend: beats 160 days by 15.9 %. exp12, same windows, same variants: 1024 h → 8760 h is **−16 % MAE**. Larger than every covariate put together. |
| Resolution | **hourly** | exp11: at matched context a 15-minute grid gives −0.6 % MAE, intervals containing zero in all seven lead buckets. It also cannot see a year — 15 360 steps are 160 days at a quarter hour. |
| Horizon | 96 h, scored in 6-hourly buckets over the first day | Upstream gauges are spent once the horizon outruns the travel time; coarse buckets hid that. |
| Covariate or target | **Not a question** for past-only series | `timesfm3/torch/model.py` concatenates targets and past-only covariates and sets `patch_is_target` across `num_target + num_past_only`. A past-only covariate *is* a target internally. Found on MLX, verified in the PyTorch source. Scores are bit-identical, not merely tied. |

## Settled: the covariate set

| series | kind | effect | where |
|---|---|---|---|
| `airtemp` München | known-future | water-only 0.684 → 0.380 MAE | exp6 |
| `t_catchment` | known-future | −4.1 % MAE, −4.3 % CRPS | exp8 |
| four raw rain gauges, hourly | known-future | −2.5 % MAE, −2.7 % CRPS on top of air + Bad Tölz | exp15 |
| `isar_toelz` (Wassertemperatur) | past-only | −4.7 % MAE, −4.6 % CRPS at full context | exp12 |
| `loisach_beuerberg` (Wassertemperatur) | past-only | −2.83 % MAE beside the fixed set at full context; paired 95% block interval [−5.25%, −0.40%] | [exp17](REPORT_exp17.md) |
| `solar_hohenpeissenberg`, hourly | known-future | **−5.8 % MAE, −5.9 % CRPS** at a year of context, better in 65 % of 212 windows | exp13 |

`t_catchment` is **one southern station**, not a mean of four: Bright Sky answers for a
coordinate and all four southern coordinates resolve to the same station, differing by
0.002–0.004 °C. The name is wrong and the series is raw.

Radiation is worth having only as a **forecast**: past-only it is −0.4 % and
indistinguishable from nothing. Hohenpeißenberg over Garmisch on both coverage (0 thin
months in 93, against 4 and a 1329-hour gap) and effect (−3.6 % against −2.9 % on the 92
windows where both exist).

## Settled: dropped, with the reason

| series | why |
|---|---|
| **every discharge** | exp12, six gauges: alone indistinguishable from zero, and next to its own temperature −0.4 % to −0.7 % with every interval crossing zero and "better in" at 51–54 %. The idea that a temperature is unreadable without the volume behind it does not survive contact with the data. |
| `isar_mittenwald`, `rissbach_klamm` | Above the Krün diversion, where most of their water leaves for the Walchensee. The two that fail are exactly the two upstream of it. |
| `isar_muenchen` | r = 0.999 with the target and +1.7 % MAE. A near-duplicate carries no lead time. |
| `schwabinger_bach` | Branches off **below** the gauge. |
| `q_sylvenstein` / `q_sylvensteinsee_ab` | Pooled −0.6 %, but the whole gain is 2024: drop that year and it turns +0.3 %. In 2022 it costs +3.8 % MAE, interval excluding zero. And the two correlate at r = 0.996. |
| `loisach_eschenlohe` | Beats bare weather, adds +0.25 % beside Bad Tölz. |
| `isar_lenggries` | Worsens the fixed set and the leading pair in exp17. |
| `isar_puppling` | Its tiny gain beside Beuerberg fails on 68 new origins: +1.55 % MAE, interval [−0.03%, +3.00%]. See exp17. |
| 24-hour sums (rain, radiation) | Raw beats constructed in both cases. |
| heat flux `Q × T` | exp18: nothing over the two raw series. The model forms the product itself when it needs it. |
| 15-minute resolution | See above. |

## Settled: final set against DUET

The incremental effects above cannot be added. An end-to-end comparison now exists:
[exp19](REPORT_exp19.md), 24 daily 96-hour windows in August–September 2026,
**oracle against oracle**. Final TimesFM achieves MAE 0.3028 vs 0.4534 °C and
shared-decile CRPS 0.1979 vs 0.2987 °C for DUET. This is a comparison of the fixed
systems with their respective contexts and inputs, not an isolated architecture test.

[exp20](REPORT_exp20.md) reruns the old DUET adapter before the production changes
on 23 starts before the September sensor fault. Its quantiles exactly reproduce
the comparison baseline. TimesFM still improves MAE by **32.0 %** and CRPS by
**32.7 %**. These overlapping seasonal windows do not prove a universal advantage.

The huge production bands of early September were caused by one historical **154.4 °C**
water reading. The guard is deployed: the run after the merge went from a 24.07 °C band
back to 3.84 °C with unchanged weights and weather.

The live comparison is no longer open, it is **running**. `eisbach/timesfm.py` publishes
this exact set beside DUET three times a day on a real MOSMIX forecast, and archives both
its forecasts and the weather it was handed. What is open is the *result*, which needs a
few months of closed 96-hour windows before it means anything.

What *is* measured end to end is older and narrower: TimesFM zero-shot with the replayed
DWD forecast beat the live DUET on 76 identical windows, 0.290 against 0.422 MAE and
0.192 against 0.286 CRPS — with air temperature as the only covariate.

## The oracle cost, now measured instead of restated

Every sweep here uses the weather that **actually occurred**, because archived DWD
forecasts begin in May 2026. exp16 put a number on what that flatters: replacing observed
Munich air with the archived forecast loses **4.8 % of the oracle advantage** in MAE and
4.1 % in CRPS — about 95 % of the gain survives.

Read it as a seasonal point estimate and nothing more. The block interval is
[−7.0 %, +13.9 %], the windows are one season and overlap heavily, and southern air,
catchment rain and radiation have **no archived forecast at all** — their errors were
simulated from Munich's, which assumes a spatial transfer nobody has verified.

**Production now archives the forecasts for this set.** It used to archive only Munich's,
which is why every number above had to be caveated. `write_covariate_forecast` keeps all
seven as they were forecast, per run, with `source_kind` recording the handful of horizon
hours Bright Sky served as measurement rather than prediction. The archive only ever grows
forward, so the clock on an honest verification started the day this shipped and not
before.
