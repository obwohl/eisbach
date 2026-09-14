# exp9 / exp14 — MLX replication and method audit

Run date: 2026-09-14. Starting commit: `677886d` on
`origin/claude/sleepy-brown-950428`; work branch: `local/exp9-exp14`.

## exp9 answer

**Keep the upstream series as past-only covariates.** Forecasting them as additional
targets gives no benefit: both the one-gauge and two-gauge comparisons are exactly tied
at both context lengths, across all 150 windows and seven lead buckets. Six supplemental
comparisons of the full 96 × 9 Eisbach quantile arrays (including a window with the
leading-upstream gap) are also bit-identical. This is implementation equivalence, not
merely a statistically inconclusive difference.

At 8760 h, Bad Tölz alone has MAE **0.364906 °C** and decile CRPS **0.239902 °C**,
versus **0.368866 / 0.241895 °C** for Bad Tölz plus Lenggries. The paired changes are
−1.074% MAE and −0.825% CRPS; both intervals include zero. In the screen Bad Tölz alone
was instead +0.969% MAE / +1.025% CRPS worse. **The count ranking reverses**, while the
framing tie survives. Keeping Bad Tölz alone is a reasonable parsimonious choice under
the stated rule, not proof that Lenggries is harmful.

Duplicating the two gauges as both targets and covariates costs +1.099% MAE and
+1.167% CRPS at full context. Its ordinary paired CRPS interval excludes zero on the
harmful side, but the monthly-block CRPS interval crosses zero ([−0.034%, +2.410%]);
that significance is sensitive to dependence. Its worse mean already fails the keep
rule; there is no evidence of complementary paths to exploit. Differences by lead bucket show that the two-gauge
set has slightly better early forecasts but worse later ones; the conclusion concerns
the requested whole 96-hour score, not every lead time separately.

## exp14 keep decision

**Under the stated pooled rule, either of the two Sylvenstein discharge series may
remain as one additional candidate beside Bad Tölz T; do not automatically allocate
two slots to them.** Nominally prefer `q_sylvenstein` (the Isar gauge below the reservoir),
with **−0.627% MAE / −0.751% CRPS** incrementally. The direct reservoir release
`q_sylvensteinsee_ab` gives **−0.531% / −0.651%**. Neither incremental interval excludes
zero, including under monthly-block resampling. Their tiny difference does not establish
one as superior. Their joint addition was not tested.

**This is a weak, time-dependent keep decision, not a settled no-harm finding.**
The gain is driven by 2024: omitting that year turns the two incremental MAEs into
**+0.307% and +0.314%**, and their CRPS changes into **+0.089% and +0.111%**. In 2022,
adding `q_sylvenstein` causes **+3.836% MAE [1.328%, 6.495%]** and **+3.472% CRPS
[0.906%, 6.221%]**; the direct release shows the same pattern. These retrospective
subgroup intervals are unadjusted, but they make “it never hurts” untenable. The full
leave-one-year-out table below exposes this sensitivity rather than selectively
removing an inconvenient period. No period was removed from the primary results.

| Candidate(s) | Decision under this experiment |
| --- | --- |
| Sylvenstein / Isar; Sylvensteinsee Abgabe | Both pass the **pooled** incremental rule, as alternatives for **one** slot; small uncertain gains with observed year dependence. |
| Loisach Eschenlohe T | No added slot: +0.253% MAE / +0.453% CRPS versus Bad Tölz, despite improving on weather alone. |
| Große Gaißach; Ellbach; Loisach Kochel; virtual Beuerberg; “Zuflüsse bei Bad Tölz” | Screen mean worsens both metrics; no keep case from this run. |
| Rißbachdüker / Isar; Jachen; Walchen; “Walchensee-System”; “alle Zweige” | Slight favourable screen means, but not shortlisted and no measured incremental benefit beside Bad Tölz. Remain unconfirmed research candidates, not justified additional slots. This does not prove they cannot help in a combination. |
| Loisach–Isar canal, Bruggen | Exclude on data quality. Its apparent screen gain, −0.700% MAE / −0.671% CRPS, is not actionable evidence. |

## Did the screen ordering survive?

**Not exactly.** In exp9, the one-versus-two-gauge ordering reverses for both metrics;
the target/covariate ties remain exact. In exp14 the shortlisted standalone ordering
changes from **Sylvenstein → Eschenlohe → reservoir release** to **Eschenlohe →
Sylvenstein → reservoir release** for both MAE and CRPS (Spearman 0.5 among these three).
Their very small screen separation makes this less dramatic than the ordinal change
suggests. Bad Tölz remains the best standalone addition among the confirmed variants.

Screen-to-full changes against weather for the three branches are, respectively,
−1.426% → −0.600% MAE, −1.396% → −1.439%, and −1.057% → −0.521%. Thus their favourable
standalone signs survive, but their effect sizes and precise ranks do not all transfer.
More importantly, the best standalone branch at full context (Eschenlohe) is the one
that fails the incremental Bad Tölz comparison. A standalone screen cannot substitute
for that conditional test. The favourable exp12 example does not establish a generally
reliable ranking protocol.

## exp14 reference and comparison with exp12

The local paired Bad Tölz benchmark is **−2.800% MAE / −2.918% CRPS** in the screen,
and **−2.278% / −2.416%** at 8760 h. Its full-context monthly-block intervals also favour
Bad Tölz (MAE [−4.041%, −0.536%], CRPS [−4.087%, −0.756%]). The supplied exp12 figures
of −4.2% / −4.3% are not numerically replicated here. The original exp12 score files
were not present locally, and its original anchors cannot be paired with these results.
Different eligible windows, cached records and backend/preprocessing are possible
contributors; this run does not identify their separate effects. Also, exp12 intervals
containing zero do not establish that the discharge effects are exactly zero. A branch must therefore
be assessed against the **local paired reference**, not a cross-experiment percentage.

## What the experiments can actually establish

The original exp9 premise is false for the installed TimesFM 3.0.2 MLX decoder:
**past-only covariates are reconstructed through the same path as targets**.
`timesfm3/mlx/model.py`, `decode`, concatenates targets and past-only covariates,
masks both over the future, and sets `num_pred = num_target + num_past_only` for
`patch_is_target`. The forecaster merely selects the requested output rows afterward.
There is no separate, weaker covariate path here. The result applies to this backend
and version, not an assertion about every multivariate forecasting model.

The first screen nevertheless produced tiny differences. Two of 150 windows started
with missing upstream readings, and `bench.prepare_inputs` trimmed a target stack to
the first jointly present value, shortening its context by up to eight hours. A
past-only covariate did not trigger that trim. This was a **preprocessing confound**,
not the already-fixed variate-axis bug. The twelve supplied regression tests passed.

`exp9.prepare_framing` now prepares the Eisbach and all auxiliary series before
assigning their roles. Only the Eisbach determines the trim; historical auxiliary gaps
use the existing interpolation. The Eisbach remains variate zero; no target truth is
filled. The corrected screen gives exactly identical MAE and CRPS in every window and
lead bucket for both framing pairs. The diagnostic screen and interrupted first
confirmation are preserved separately in the ignored cache as `exp9_diagnostic_*`.

Exp14 needs a **paired Bad Tölz comparison**, not a comparison with a percentage from
another experiment. Its screen therefore includes Bad Tölz T on the same 250 windows.
The top three eligible branches are confirmed alone and added to Bad Tölz T at 8760 h.
The operator-flagged Loisach–Isar canal is screened as a diagnostic but excluded from
selection. A negative mean against weather alone cannot establish that a branch adds
anything beyond Bad Tölz.

## Station identity corrections

Three legacy exp14 labels invite incorrect physical interpretations. The table renderer
uses corrected descriptions; CSV keys remain unchanged for traceability.

| Cached series / legacy label | What the official station actually represents |
| --- | --- |
| `q_rissbachdueker` / Rißbach-Düker (Ableitung) | **Isar** at Rißbachdüker, station 16001303, catchment 523.9 km². It does not directly measure the removed water. [LfU station metadata](https://www.gkd.bayern.de/de/fluesse/abfluss/isar/rissbachdueker-16001303). |
| `q_sylvenstein` / Sylvenstein Zufluss | **Isar below the reservoir**, station 16002500, not reservoir inflow. LfU explicitly describes its flow as predominantly reservoir release. [LfU flood report, p. 40–41](https://files.hnd.bayern.de/berichte/Gewaesserkundl_Bericht_HW200508.pdf). |
| `q_loisach_beuerberg_kanal` / Loisach-Beuerberg-Kanal | **Virtual Beuerberg: Loisach with canal**, station 16408506, not a separate canal-only observation. [LfU station metadata](https://www.gkd.bayern.de/de/fluesse/abfluss/bayern/beuerberg-virtuell-16408506). |

Consequently, the legacy “Walchensee-System” combination is simply the supplied Isar,
Walchen and Jachen series, not a measured system balance or a direct test of diversion
volume. “alle Zweige” is the original **eight-series subset**, not every candidate in
the list. Neither the data nor model results justify the stronger labels. The two supplied
reservoir-related discharge columns have Pearson r = **0.99563** over their common
cached hours; treating them as independent evidence or automatically retaining both
would be misleading.

The retrieved [LfU Bruggen measurement page](https://www.gkd.bayern.de/de/fluesse/abfluss/isar/bruggen-16495000/messwerte)
confirms the backwater / incorrect-discharge warning (the retrieved page shows August
2026 readings). The warning does not establish its historical start date, so it cannot
be cleanly assigned to every historical sample. An apparent gain there would not prove
a bug in TimesFM: a biased or imputed series can still correlate with useful conditions.
It would be a failure to treat that score as evidence for a reliable physical covariate.

## Protocol and limits

- exp9: 150 fixed windows; all six variants at 1024 h; both framing pairs plus the
  duplicate target-and-covariate variant at 8760 h. Confirming only the top two screen
  winners would not answer the requested one-gauge versus two-gauge comparison.
- exp14: 250 fixed windows; weather, Bad Tölz, and fourteen branch variants at 1024 h;
  weather, Bad Tölz, three selected branches, and their three Bad Tölz combinations at
  8760 h. Shortlisting uses screen MAE and excludes the known-invalid canal.
- Horizon is 96 h; seven lead buckets cover it without overlap. The principal scores
  weight hours, so the last three 24-hour buckets carry more weight than an early
  six-hour bucket. Missing observations remain unscored.
- All weather futures are **observed weather (oracle)**. These are covariate-selection
  results, not demonstrated live forecasting gains. Solar and rain are not model inputs
  to either experiment, so no solar download was necessary.
- Intervals are paired bootstrap intervals, not unpaired intervals or intervals for
  individual forecast errors. The report uses the same 8,000 draws (seed 0) for each
  candidate. The original console printer advances its random stream across candidates,
  so its interval endpoints differ slightly. The borderline Sylvenstein screen CRPS
  interval crosses zero in this report but narrowly excluded it in the console;
  this Monte Carlo sensitivity does not change the mean-based keep decision. The main intervals reproduce the window bootstrap;
  supplemental intervals resample calendar months together. Of consecutive anchors,
  68/149 in exp9 and 87/249 in exp14 have overlapping 96-hour horizons. Thus independent
  window resampling can understate uncertainty. Monthly resampling does not remove
  all longer-term dependence.
- Screening and full-context confirmation use the **same windows**, so confirmation
  tests context sensitivity, not independent out-of-sample selection performance.
  Intervals are unadjusted for multiple candidates; selection optimism remains.
- The keep rule is interpreted as favourable mean MAE and CRPS without clear evidence
  of harm, not a requirement that the upper bound be below zero. An interval crossing
  zero is uncertainty, not proof of no harm. Branches need favourable **incremental**
  results versus Bad Tölz to earn an additional slot.
- exp9 windows per year, 2019–2026: 1, 22, 22, 22, 23, 23, 21, 16.
  exp14: 1, 38, 37, 36, 38, 36, 37, 27. The original all-series filter gave only
  three 2023 windows and 23 in 2024 because the invalid canal has just 44.1% coverage
  during 2023. That short screen was interrupted and preserved as `exp14_diagnostic_*`.
  The corrected filter excludes this diagnostic series; all other candidates and Bad
  Tölz share the same 250 windows. The canal's diagnostic now also includes substantial
  imputation across missing periods, another reason not to interpret its score as skill.
  A one-window 2019 interval is degenerate and carries no independent seasonal evidence.
- The Bruggen exclusion is supported by the retrieved official warning above.
  Peternerbrücke/Jachen's disturbed live transmission is from the handoff; its present
  status was not verified. Historical completeness does not establish live reliability.

## Reproduction and integrity

The existing `.venv-exp11` supplies Python 3.12.13, TimesFM 3.0.2, MLX 0.32.2,
NumPy 2.5.3 and pandas 3.0.5. The local Hugging Face checkpoint is reused offline: snapshot
`43046b85ec22d584a13f8098c2ed39c889e129c2` of `google/timesfm-3.0-pytorch`.
The sandbox could not access Metal; the requested GPU runs used the approved local
execution path outside it.

`long_hourly.csv` was missing. Existing cached Eisbach and Isar hourly measurements
were joined to weather fetched by the existing `build_dataset.fetch_weather` function
for 2019–2026, explicitly using DWD station `03379`. No gauge archive was rewritten.
The resulting cache has 67,525 rows; Munich air temperature coverage is 99.886%.
Existing `stations_hourly.csv` and `weather_south.csv` were reused unchanged.

SHA-256 fingerprints of the input CSV files:

```text
stations_hourly.csv 731d84985990ad581b27f2b512ec0e33364267a198273f9c2348ea4df8327bff
weather_south.csv   4095944d49a97dec6fdd908edd07d4692374d645bb9f1ac5fac7ba48b068e0ae
long_hourly.csv     e0b2d4f87db3ba46d7f3b587936788a68718136ab944a65cf109de2fb542da70
```

The handoff also overstated restart support: exp9 never read its checkpoints;
exp14's inherited `_resume` discarded the last label even after a completed write.
Both now write atomically and validate full window/bucket coverage and finite scores
when resuming. JSON manifests bind cached results to input data, columns, windows,
context, variants and backend/package versions. Missing or changed provenance fails
closed rather than silently mixing results.

```bash
export TIMESFM_BACKEND=mlx
export HF_HOME="$PWD/data/experiments/hf"
export HF_HUB_OFFLINE=1
.venv-exp11/bin/python -u experiments/timesfm/exp9_targets.py > experiments/timesfm/exp9.log 2>&1
.venv-exp11/bin/python -u experiments/timesfm/exp14_branches.py > experiments/timesfm/exp14.log 2>&1
.venv-exp11/bin/python experiments/timesfm/report_exp9_exp14.py
.venv-exp11/bin/python -m pytest experiments/timesfm/test_bench_prepare.py experiments/timesfm/test_checkpoint.py experiments/timesfm/test_exp9_framing.py -q
.venv-exp11/bin/python -m pytest -q
.venv-exp11/bin/ruff check .
```

Raw scores and manifests live under `data/experiments/exp9_{targets,confirm}.{csv,json}`
and `data/experiments/exp14_{branches,confirm}.{csv,json}`. Logs and generated data are
ignored; the report and reproduction code are committed. Changes are confined to
`experiments/timesfm/`; production code, tests and `data/archive/` receive no edits
from this work.

Validation completed: **189 passed, 2 skipped** in the repository suite; **21 passed**
in the explicit experiment regression suite; `ruff check .` passed. The two skips
require the absent original `ts_proba_cuda` submodule for comparison. One existing
PyTorch `torch.jit.script` deprecation warning remains. No failing tests or
nonfinite checkpoint scores were observed in the corrected runs.

<!-- generated tables -->
# Paired tables: exp9 and exp14

Generated by `report_exp9_exp14.py`. Negative changes favour the candidate. Intervals are paired 95% bootstrap intervals (8,000 shared draws); percentages and intervals use the mean per-window reference score as denominator. Absolute scores pool observed hours. CRPS is integrated over deciles 0.1–0.9.

## exp9 screen, 1024 h

Reference: **upstream as covariates**; MAE 0.435544 °C, CRPS 0.287697 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| baseline: air only | 150 | 0.470108 | +7.944% [+3.841, +12.160] | 0.310229 | +7.839% [+3.960, +11.809] |
| upstream as targets | 150 | 0.435544 | +0.000% [+0.000, +0.000] | 0.287697 | +0.000% [+0.000, +0.000] |
| Bad Tölz only, as covariate | 150 | 0.439773 | +0.969% [-0.289, +2.245] | 0.290652 | +1.025% [-0.154, +2.258] |
| Bad Tölz only, as target | 150 | 0.439773 | +0.969% [-0.289, +2.245] | 0.290652 | +1.025% [-0.154, +2.258] |
| upstream as targets and covariates | 150 | 0.438677 | +0.721% [-0.289, +1.657] | 0.289484 | +0.623% [-0.323, +1.519] |

### One-gauge framing comparison

Reference: **Bad Tölz only, as covariate**; MAE 0.439773 °C, CRPS 0.290652 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz only, as target | 150 | 0.439773 | +0.000% [+0.000, +0.000] | 0.290652 | +0.000% [+0.000, +0.000] |

## exp9 confirmation, 8760 h

Reference: **upstream as covariates**; MAE 0.368866 °C, CRPS 0.241895 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| upstream as targets | 150 | 0.368866 | +0.000% [+0.000, +0.000] | 0.241895 | +0.000% [+0.000, +0.000] |
| Bad Tölz only, as covariate | 150 | 0.364906 | -1.074% [-2.369, +0.239] | 0.239902 | -0.825% [-2.021, +0.390] |
| Bad Tölz only, as target | 150 | 0.364906 | -1.074% [-2.369, +0.239] | 0.239902 | -0.825% [-2.021, +0.390] |
| upstream as targets and covariates | 150 | 0.372927 | +1.099% [-0.076, +2.273] | 0.244723 | +1.167% [+0.089, +2.265] |

### One-gauge framing comparison

Reference: **Bad Tölz only, as covariate**; MAE 0.364906 °C, CRPS 0.239902 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz only, as target | 150 | 0.364906 | +0.000% [+0.000, +0.000] | 0.239902 | +0.000% [+0.000, +0.000] |

## exp14 screen, 1024 h

Reference: **weather only**; MAE 0.460478 °C, CRPS 0.301970 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz T | 250 | 0.447583 | -2.800% [-4.633, -0.993] | 0.293158 | -2.918% [-4.591, -1.187] |
| Loisach Eschenlohe T | 250 | 0.454047 | -1.396% [-3.066, +0.276] | 0.297678 | -1.421% [-3.028, +0.191] |
| Rißbachdüker / Isar | 250 | 0.457239 | -0.703% [-2.219, +0.841] | 0.300059 | -0.633% [-2.048, +0.815] |
| Sylvensteinsee Abgabe | 250 | 0.455609 | -1.057% [-2.570, +0.461] | 0.298991 | -0.986% [-2.422, +0.469] |
| Sylvenstein / Isar (below reservoir) | 250 | 0.453910 | -1.426% [-2.929, +0.112] | 0.297663 | -1.426% [-2.832, +0.015] |
| Jachen (Peternerbrücke) | 250 | 0.459298 | -0.258% [-1.583, +1.033] | 0.301355 | -0.205% [-1.447, +1.022] |
| Walchen | 250 | 0.458771 | -0.372% [-1.659, +0.921] | 0.301167 | -0.267% [-1.441, +0.930] |
| Große Gaißach | 250 | 0.461625 | +0.248% [-1.088, +1.587] | 0.302672 | +0.232% [-0.985, +1.477] |
| Ellbach | 250 | 0.465853 | +1.166% [-0.267, +2.623] | 0.305217 | +1.075% [-0.245, +2.419] |
| Loisach Kochel | 250 | 0.463190 | +0.589% [-1.001, +2.146] | 0.304059 | +0.692% [-0.800, +2.171] |
| Beuerberg virtual: Loisach with canal | 250 | 0.460753 | +0.060% [-1.468, +1.629] | 0.302404 | +0.143% [-1.311, +1.659] |
| Loisach-Isar-Kanal (LfU: Werte nicht korrekt) | 250 | 0.457252 | -0.700% [-1.898, +0.505] | 0.299942 | -0.671% [-1.830, +0.503] |
| Walchensee-System | 250 | 0.457868 | -0.569% [-2.430, +1.293] | 0.300214 | -0.583% [-2.322, +1.149] |
| Zuflüsse bei Bad Tölz | 250 | 0.465617 | +1.115% [-0.593, +2.857] | 0.304659 | +0.889% [-0.715, +2.522] |
| alle Zweige | 250 | 0.456592 | -0.845% [-3.352, +1.720] | 0.299019 | -0.979% [-3.450, +1.518] |

## exp14 confirmation, 8760 h

Reference: **weather only**; MAE 0.368851 °C, CRPS 0.242485 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz T | 250 | 0.360457 | -2.278% [-4.264, -0.345] | 0.236630 | -2.416% [-4.295, -0.577] |
| Sylvenstein / Isar (below reservoir) | 250 | 0.366635 | -0.600% [-2.195, +0.997] | 0.240495 | -0.820% [-2.262, +0.583] |
| Loisach Eschenlohe T | 250 | 0.363542 | -1.439% [-3.176, +0.300] | 0.239505 | -1.229% [-2.891, +0.462] |
| Sylvensteinsee Abgabe | 250 | 0.366926 | -0.521% [-2.101, +1.041] | 0.240540 | -0.801% [-2.214, +0.547] |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | 250 | 0.358190 | -2.890% [-5.564, -0.284] | 0.234848 | -3.149% [-5.593, -0.756] |
| Bad Tölz T + Loisach Eschenlohe T | 250 | 0.361370 | -2.030% [-4.452, +0.366] | 0.237702 | -1.974% [-4.293, +0.355] |
| Bad Tölz T + Sylvensteinsee Abgabe | 250 | 0.358534 | -2.796% [-5.439, -0.206] | 0.235084 | -3.051% [-5.507, -0.674] |

## Incremental branch value at 8760 h

Reference: **Bad Tölz T**; MAE 0.360457 °C, CRPS 0.236630 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | 250 | 0.358190 | -0.627% [-2.147, +0.881] | 0.234848 | -0.751% [-2.090, +0.549] |
| Bad Tölz T + Loisach Eschenlohe T | 250 | 0.361370 | +0.253% [-0.797, +1.366] | 0.237702 | +0.453% [-0.542, +1.516] |
| Bad Tölz T + Sylvensteinsee Abgabe | 250 | 0.358534 | -0.531% [-2.001, +0.947] | 0.235084 | -0.651% [-1.932, +0.625] |

## Dependence sensitivity

Resample calendar months with all their windows together; this accommodates within-month dependence, but is not an independent temporal holdout.

| Variant vs weather only | MAE % monthly-block CI | CRPS % monthly-block CI |
| --- | --- | --- |
| Bad Tölz T | [-4.041, -0.536] | [-4.087, -0.756] |
| Sylvenstein / Isar (below reservoir) | [-2.104, +0.946] | [-2.197, +0.538] |
| Loisach Eschenlohe T | [-3.254, +0.315] | [-2.957, +0.479] |
| Sylvensteinsee Abgabe | [-2.020, +1.012] | [-2.141, +0.562] |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | [-5.424, -0.457] | [-5.479, -0.923] |
| Bad Tölz T + Loisach Eschenlohe T | [-4.380, +0.226] | [-4.205, +0.181] |
| Bad Tölz T + Sylvensteinsee Abgabe | [-5.278, -0.417] | [-5.339, -0.867] |

| Variant vs Bad Tölz T | MAE % monthly-block CI | CRPS % monthly-block CI |
| --- | --- | --- |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | [-2.236, +0.958] | [-2.189, +0.644] |
| Bad Tölz T + Loisach Eschenlohe T | [-0.924, +1.378] | [-0.624, +1.501] |
| Bad Tölz T + Sylvensteinsee Abgabe | [-2.120, +1.024] | [-2.064, +0.741] |

| Variant vs upstream as covariates | MAE % monthly-block CI | CRPS % monthly-block CI |
| --- | --- | --- |
| Bad Tölz only, as covariate | [-2.574, +0.396] | [-2.209, +0.537] |
| upstream as targets and covariates | [-0.178, +2.414] | [-0.034, +2.410] |

## exp9: ranking among jointly evaluated variants

| Metric | Variant | Screen rank | Full rank | Screen score | Full score |
| --- | --- | --- | --- | --- | --- |
| MAE | Bad Tölz only, as covariate | 4 | 1 | 0.439681 | 0.364888 |
| MAE | Bad Tölz only, as target | 4 | 1 | 0.439681 | 0.364888 |
| MAE | upstream as covariates | 1 | 3 | 0.435462 | 0.368850 |
| MAE | upstream as targets | 1 | 3 | 0.435462 | 0.368850 |
| MAE | upstream as targets and covariates | 3 | 5 | 0.438601 | 0.372902 |
| CRPS | Bad Tölz only, as covariate | 4 | 1 | 0.290589 | 0.239884 |
| CRPS | Bad Tölz only, as target | 4 | 1 | 0.290589 | 0.239884 |
| CRPS | upstream as covariates | 1 | 3 | 0.287641 | 0.241879 |
| CRPS | upstream as targets | 1 | 3 | 0.287641 | 0.241879 |
| CRPS | upstream as targets and covariates | 3 | 5 | 0.289432 | 0.244702 |

## exp14: ranking among jointly evaluated variants

| Metric | Variant | Screen rank | Full rank | Screen score | Full score |
| --- | --- | --- | --- | --- | --- |
| MAE | Bad Tölz T | 1 | 1 | 0.447544 | 0.360462 |
| MAE | Loisach Eschenlohe T | 3 | 2 | 0.454008 | 0.363554 |
| MAE | Sylvenstein / Isar (below reservoir) | 2 | 3 | 0.453870 | 0.366648 |
| MAE | Sylvensteinsee Abgabe | 4 | 4 | 0.455571 | 0.366940 |
| MAE | weather only | 5 | 5 | 0.460436 | 0.368863 |
| CRPS | Bad Tölz T | 1 | 1 | 0.293132 | 0.236631 |
| CRPS | Loisach Eschenlohe T | 3 | 2 | 0.297652 | 0.239510 |
| CRPS | Sylvenstein / Isar (below reservoir) | 2 | 3 | 0.297637 | 0.240502 |
| CRPS | Sylvensteinsee Abgabe | 4 | 4 | 0.298966 | 0.240547 |
| CRPS | weather only | 5 | 5 | 0.301943 | 0.242490 |

## exp14 leave-one-year-out sensitivity

Post-hoc influence diagnostic, not a new holdout or an altered primary sample.

| Omitted year | Remaining windows | Addition to Bad Tölz | Δ MAE | Δ CRPS |
| --- | --- | --- | --- | --- |
| 2019 | 249 | Bad Tölz T + Sylvenstein / Isar (below reservoir) | -0.625% | -0.753% |
| 2019 | 249 | Bad Tölz T + Loisach Eschenlohe T | +0.277% | +0.470% |
| 2019 | 249 | Bad Tölz T + Sylvensteinsee Abgabe | -0.535% | -0.656% |
| 2020 | 212 | Bad Tölz T + Sylvenstein / Isar (below reservoir) | -0.251% | -0.489% |
| 2020 | 212 | Bad Tölz T + Loisach Eschenlohe T | +0.303% | +0.548% |
| 2020 | 212 | Bad Tölz T + Sylvensteinsee Abgabe | -0.006% | -0.241% |
| 2021 | 213 | Bad Tölz T + Sylvenstein / Isar (below reservoir) | -0.982% | -0.907% |
| 2021 | 213 | Bad Tölz T + Loisach Eschenlohe T | +0.480% | +0.808% |
| 2021 | 213 | Bad Tölz T + Sylvensteinsee Abgabe | -1.102% | -0.998% |
| 2022 | 214 | Bad Tölz T + Sylvenstein / Isar (below reservoir) | -1.359% | -1.436% |
| 2022 | 214 | Bad Tölz T + Loisach Eschenlohe T | -0.065% | +0.161% |
| 2022 | 214 | Bad Tölz T + Sylvensteinsee Abgabe | -1.209% | -1.289% |
| 2023 | 212 | Bad Tölz T + Sylvenstein / Isar (below reservoir) | -0.785% | -0.899% |
| 2023 | 212 | Bad Tölz T + Loisach Eschenlohe T | +0.414% | +0.508% |
| 2023 | 212 | Bad Tölz T + Sylvensteinsee Abgabe | -0.611% | -0.730% |
| 2024 | 214 | Bad Tölz T + Sylvenstein / Isar (below reservoir) | +0.307% | +0.089% |
| 2024 | 214 | Bad Tölz T + Loisach Eschenlohe T | -0.185% | +0.021% |
| 2024 | 214 | Bad Tölz T + Sylvensteinsee Abgabe | +0.314% | +0.111% |
| 2025 | 213 | Bad Tölz T + Sylvenstein / Isar (below reservoir) | -0.785% | -0.939% |
| 2025 | 213 | Bad Tölz T + Loisach Eschenlohe T | +0.324% | +0.438% |
| 2025 | 213 | Bad Tölz T + Sylvensteinsee Abgabe | -0.575% | -0.777% |
| 2026 | 223 | Bad Tölz T + Sylvenstein / Isar (below reservoir) | -0.523% | -0.662% |
| 2026 | 223 | Bad Tölz T + Loisach Eschenlohe T | +0.474% | +0.672% |
| 2026 | 223 | Bad Tölz T + Sylvensteinsee Abgabe | -0.513% | -0.616% |

## exp14 confirmation by calendar year

### 2019

Reference: **Bad Tölz T**; MAE 0.201289 °C, CRPS 0.132734 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | 1 | 0.198595 | -1.338% [-1.338, -1.338] | 0.132716 | -0.014% [-0.014, -0.014] |
| Bad Tölz T + Loisach Eschenlohe T | 1 | 0.180696 | -10.231% [-10.231, -10.231] | 0.123331 | -7.084% [-7.084, -7.084] |
| Bad Tölz T + Sylvensteinsee Abgabe | 1 | 0.203672 | +1.184% [+1.184, +1.184] | 0.134655 | +1.448% [+1.448, +1.448] |

### 2020

Reference: **Bad Tölz T**; MAE 0.410411 °C, CRPS 0.265663 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | 38 | 0.400436 | -2.426% [-5.529, +0.603] | 0.260265 | -2.028% [-5.136, +0.919] |
| Bad Tölz T + Loisach Eschenlohe T | 38 | 0.410445 | +0.015% [-2.529, +2.442] | 0.265626 | -0.009% [-2.489, +2.366] |
| Bad Tölz T + Sylvensteinsee Abgabe | 38 | 0.397904 | -3.042% [-6.128, -0.009] | 0.258624 | -2.645% [-5.732, +0.298] |

### 2021

Reference: **Bad Tölz T**; MAE 0.382095 °C, CRPS 0.255491 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | 37 | 0.386984 | +1.282% [-3.218, +6.604] | 0.255656 | +0.068% [-3.689, +4.150] |
| Bad Tölz T + Loisach Eschenlohe T | 37 | 0.378420 | -0.964% [-3.105, +1.158] | 0.251879 | -1.416% [-3.390, +0.485] |
| Bad Tölz T + Sylvensteinsee Abgabe | 37 | 0.391789 | +2.539% [-1.895, +7.710] | 0.258473 | +1.170% [-2.447, +5.081] |

### 2022

Reference: **Bad Tölz T**; MAE 0.352908 °C, CRPS 0.229464 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | 36 | 0.366446 | +3.836% [+1.328, +6.495] | 0.237430 | +3.472% [+0.906, +6.221] |
| Bad Tölz T + Loisach Eschenlohe T | 36 | 0.360648 | +2.193% [-0.394, +4.768] | 0.234636 | +2.254% [-0.340, +4.826] |
| Bad Tölz T + Sylvensteinsee Abgabe | 36 | 0.365604 | +3.598% [+1.328, +5.992] | 0.236985 | +3.278% [+1.018, +5.739] |

### 2023

Reference: **Bad Tölz T**; MAE 0.320785 °C, CRPS 0.211649 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | 38 | 0.322018 | +0.384% [-2.462, +3.390] | 0.212046 | +0.188% [-2.334, +2.782] |
| Bad Tölz T + Loisach Eschenlohe T | 38 | 0.318300 | -0.775% [-4.200, +3.054] | 0.211863 | +0.101% [-2.844, +3.650] |
| Bad Tölz T + Sylvensteinsee Abgabe | 38 | 0.320718 | -0.021% [-3.106, +3.288] | 0.211329 | -0.151% [-3.077, +2.921] |

### 2024

Reference: **Bad Tölz T**; MAE 0.356629 °C, CRPS 0.236505 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | 36 | 0.334337 | -6.251% [-10.791, -2.023] | 0.222917 | -5.745% [-9.549, -2.225] |
| Bad Tölz T + Loisach Eschenlohe T | 36 | 0.366935 | +2.890% [+0.250, +5.563] | 0.243645 | +3.019% [+0.585, +5.470] |
| Bad Tölz T + Sylvensteinsee Abgabe | 36 | 0.336588 | -5.619% [-10.107, -1.329] | 0.224238 | -5.187% [-8.926, -1.573] |

### 2025

Reference: **Bad Tölz T**; MAE 0.293642 °C, CRPS 0.193706 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | 37 | 0.295160 | +0.525% [-5.233, +6.067] | 0.194871 | +0.609% [-4.198, +5.230] |
| Bad Tölz T + Loisach Eschenlohe T | 37 | 0.292879 | -0.263% [-3.331, +3.059] | 0.194796 | +0.560% [-2.212, +3.518] |
| Bad Tölz T + Sylvensteinsee Abgabe | 37 | 0.293000 | -0.209% [-5.376, +4.844] | 0.194198 | +0.262% [-3.984, +4.354] |

### 2026

Reference: **Bad Tölz T**; MAE 0.429019 °C, CRPS 0.277508 °C.

| Variant | Windows | MAE °C | Δ MAE % [95% CI] | CRPS °C | Δ CRPS % [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Bad Tölz T + Sylvenstein / Isar (below reservoir) | 27 | 0.423312 | -1.330% [-4.300, +1.729] | 0.273713 | -1.368% [-4.395, +1.705] |
| Bad Tölz T + Loisach Eschenlohe T | 27 | 0.423697 | -1.241% [-4.409, +1.611] | 0.274566 | -1.060% [-4.221, +1.719] |
| Bad Tölz T + Sylvensteinsee Abgabe | 27 | 0.426203 | -0.657% [-3.923, +2.415] | 0.275036 | -0.891% [-4.278, +2.221] |
