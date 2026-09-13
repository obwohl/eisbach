# experiments/timesfm — TimesFM 3.0 zero-shot on the Eisbach

A side branch, deliberately outside the pipeline. Nothing here is imported by `main.py`
or `eisbach/`, nothing here runs in CI, and nothing here writes to `data/archive/`. The
production forecast is untouched.

Unlike `research/`, this tree **is** meant to run: `pip install 'timesfm[torch]'`, then
`python build_dataset.py` once and the `exp*.py` scripts in any order.

## What was measured

Zero-shot `google/timesfm-3.0-pytorch` (330 M parameters, 20 layers, 9 deciles, context
up to 15 360 steps, native past-only *and* past-and-future covariates). No fine-tuning.

**The honest head to head** — 76 archived live runs, the same windows, the same truth,
both models' quantiles integrated over the same decile grid, and both fed the DWD
forecast *as it was really issued* through the production snapshot lookup:

| model | MAE | CRPS | 80 % band width | 80 % coverage |
|---|---:|---:|---:|---:|
| TimesFM + DWD forecast | **0.290** | **0.192** | 1.10 | 85.5 % |
| DUET (ours, live) | 0.422 | 0.286 | 2.08 | 93.5 % |
| TimesFM, water only | 0.644 | 0.429 | 2.43 | 85.7 % |

31 % better on MAE, 33 % on CRPS, with bands half as wide that still cover. The margin
*grows* with lead: at 72–96 h it is 0.302 against 0.477.

**Both models use the weather; only one of them sees its level.** Shifting `airtemp_96`
by +3 °C moves DUET's water forecast by 1.9e-6 °C and TimesFM's by **+1.09 °C at 96 h**,
symmetric in sign and roughly linear (+10 °C gives +3.15 °C). But that is a difference in
*level* sensitivity only: `experiments/duet_covariates/` shows DUET responds strongly to
the forecast's shape — replacing the forecast part of the channel with a real cold spell
moves it 2.36 °C — and that a wrong weather forecast doubles its MAE. So the gap in the
table above is **not** the one-line explanation "DUET ignores the weather", which is what
PRD R6 used to claim and no longer does. What separates them here is left open: level
blindness is a candidate, so are the twice-too-wide bands and the lag through regime
changes.

**Weather forecast error is not the bottleneck.** Replay (0.290) and oracle (0.285) are
within noise of each other, so DWD's four-day air temperature is already good enough that
a perfect one would buy almost nothing.

**A full year of context is the sweet spot.** Sweeping context on 442 windows over
2023–2026: 384 h (DUET's window) gives MAE 0.785, 8 760 h gives 0.688, and the 15 360 h
maximum gives 0.693 for three times the compute. Half a year (4 380 h) is *worse* than a
quarter — the model appears to want whole seasonal cycles, and the maximum context is
1.75 years, which is not one.

**The gap is widest exactly where it matters.** On the 25 % of windows that move most,
DUET scores 0.467 and TimesFM 0.329. Across strong cooling, DUET's bias runs −0.46 °C:
it lags the river through regime changes, which is what the PRD's calibration section
already suspected. Water-only TimesFM collapses there (1.28) — without the weather,
nothing can know a cold snap is coming.

## Two cautions

* **The weights are `timesfm-non-commercial-license-v1.0`** — non-commercial *and
  non-production*. The source is Apache-2.0; the checkpoint is not. Read it before
  putting this anywhere near the scheduled run.
* **TimesFM reports only the nine deciles.** There is no q0.01/q0.99, so
  `validate.py`'s 1–99 % band check and `verification.py`'s seven PIT knots would both
  need a decision, not a patch.

## Files

| file | what it does |
|---|---|
| `build_dataset.py` | Fetches 16 years of hourly Eisbach + Isar from GKD (chunked per year; longer ranges return empty) and Bright Sky weather, into `data/experiments/long_hourly.csv` (gitignored). |
| `quality.py` | Data quality report: gaps, plausibility, flat runs. |
| `bench.py` | Shared harness — shared decile grid, CRPS, PIT, pooling, baselines. |
| `exp1_context.py` | Context-length sweep, univariate. |
| `exp2_covariates.py` | Covariate variants plus the shift probe. |
| `exp3_headtohead.py` | Against the archive, with oracle weather. |
| `exp4_replay.py` | Against the archive, with the DWD forecast as issued. **The honest one.** |
| `exp5_regimes.py` | Splits any score table by how far the river moved. |
| `plot_*.py`, `viz.py` | Plots in `plots/`. |

`../duet_covariates/` holds the companion investigation into the live model: `probe.py`
(what it sees in its weather covariate), `usefulness.py` (whether the weather forecast
improves its score), and `deblind.py` (whether its level blindness can be removed without
retraining — it can, and it does not help).

## Traps

* **The CRPS grid is not free.** The production CRPS integrates the pinball loss over
  the levels the model reports. DUET reports [0.01, 0.99] and TimesFM [0.1, 0.9], so
  integrating each over its own range hands TimesFM a smaller number for nothing.
  Everything here uses the deciles for both.
* **Oracle is not skill.** `exp3` feeds observed air temperature over the horizon. It is
  labelled oracle everywhere and `exp4` is the one to quote.
* **Two windows are confounded.** Runs anchored at or after 2026-09-09 08:00 UTC had a
  154.4 °C instrument fault in DUET's real input but not in the cleaned series TimesFM
  gets. Excluding them changes the headline by 0.002 °C; the case-study plot excludes
  them anyway.
* **GKD's gauge holds one 65.8 °C hour and the production archive holds 154.4 °C for the
  same hour.** `bench.load_dataset()` masks values outside −1..32 °C. The pipeline has no
  such gate on its input — see the note in the session summary.
