# archive/

**Nothing in this directory is maintained, imported, run, or tested.**

No production code depends on anything here. Nothing under `archive/` is imported by
`main.py` or `eisbach/`, exercised by `tests/`, or invoked by any CI workflow. The
dependencies these files need are not in the root `pyproject.toml`, and several of
the scripts will not run at all without edits (see the traps below).

This tree is kept for the **research record only** — so that findings and pipelines
that took real effort to produce do not have to be reconstructed from scratch if
someone wants to revisit them.

**The production pipeline is `main.py` + `eisbach/`.** If you are looking for the code
that actually generates the forecast, it is there, not here.

---

## Contents

### `autogluon_chronos_bolt.py`
Cell 7 of the former `eisbach_2025_backup.ipynb`: a self-contained AutoGluon /
Chronos-Bolt (`bolt_base`) forecasting pipeline written for Kaggle. Needs `autogluon`,
which is not declared anywhere; imports `kaggle_secrets` and `boto3`, so it cannot run
outside Kaggle as-is. Benchmarking it against the production DUET-Prob model is
**parked, not abandoned**.

### `isar_eisbach_comparison/`
A 10-year Eisbach-vs-Isar water temperature study, plus a Chronos-2 covariate and
backtesting comparison against the custom baseline model.

| Item | What it is |
|---|---|
| `README.md` | The research write-up. **This is the reason the directory was kept** — the findings in it are genuine. |
| `metrics.py` | MAE / CRPS (pinball integral) / mean interval score. The only shared helper. |
| `analyze_diff_patterns.py` | Generates the `diff_*.png` daily/yearly/timeseries pattern plots. |
| `analyze_combinations.py` | Chronos-2 covariate-coupling sweep → `forecasting_combinations.csv`. |
| `analyze_rivers.py` | Per-river Chronos-2 forecast evaluation → `forecasting_results_10y.csv`. |
| `volatile_showdown.py` | Searches the 10-year series for high-volatility windows, then runs baseline vs Chronos-2 → `plots/volatile_*.png`. |
| `isar_eisbach_10_years.csv`, `weather_10_years.csv`, `weather_10_years_pressure.csv` | Input data. |
| `forecasting_*.csv`, `model_showdown.csv`, `finetuned_showdown.csv` | Result tables. |
| `diff_*.png`, `plots/` | Result plots. |
| `requirements.txt` | The extra deps (`chronos`, `seaborn`) these scripts need. |

Deleted during the archive pass: `compare_models.py` (superseded by
`volatile_showdown.py`), `compare_finetuned.py` and `finetune_chronos.py` (see below),
`scatter.png` and `diff_dist.png` (no generator anywhere in the repo).

---

## Traps for a future reader

These were established during the repository audit and are recorded here because each
one is easy to get wrong from the filenames alone.

### 1. `finetuned_showdown.csv` does NOT contain fine-tuned results

`compare_finetuned.py` was deleted because it never actually loaded a fine-tuned model.
It called `BaseChronosPipeline.from_pretrained("amazon/chronos-2", ...)` — the **stock**
zero-shot checkpoint — and then wrote its result row under the label
`Chronos-2 (Fine-Tuning execution timeout)`.

So `finetuned_showdown.csv`, which is still present in this directory, is a **stock
chronos-2 evaluation despite its name**. Its numbers are not evidence about fine-tuning
and must not be cited as such. (Its MAE column is in fact identical to
`model_showdown.csv`; only the CRPS column differs.) The fine-tuning work was never
completed — the training script `finetune_chronos.py` was written but never
successfully run.

**Which CRPS the top-level README cites, and why.** The two files disagree on CRPS
(0.216 / 0.503 here versus 0.311 / 0.420 in `finetuned_showdown.csv`) because the later
run recomputed it as a proper pinball-loss integral over non-uniform quantile spacing.
The README quotes the later pair. That is legitimate — it is a valid *stock* chronos-2
comparison, which is exactly what the README claims it to be — but note that only the
label on that file is wrong, not the arithmetic. Both runs put this model ahead on both
metrics, so the conclusion does not depend on which pair you take.

### 2. `weather_10_years_pressure.csv` cannot be regenerated

There is **no generator for `weather_10_years_pressure.csv` anywhere in the repository**
— not in `eisbach/`, not in this directory, not in the deleted files. It is consumed
(by `volatile_showdown.py`) but never produced. If it is ever lost it cannot be
reproduced from this repo. The same caveat applies in weaker form to
`isar_eisbach_10_years.csv` and `weather_10_years.csv`.

### 3. `volatile_showdown.py` only runs as a path from the repo root

It does `from metrics import mean_absolute_error, crps, mean_interval_score` — a bare
module import, not a package-relative one. It therefore only resolves when the script's
own directory is on `sys.path`, i.e. when it is invoked as a **path**
(`python archive/isar_eisbach_comparison/volatile_showdown.py`) and **not** via `-m`.
Conversely `analyze_combinations.py` uses the package-style
`from isar_eisbach_comparison.metrics import ...`, which wants the opposite invocation.
The two are mutually inconsistent, and neither import style survived the move into
`archive/` unchanged.

Related: every data path inside these scripts is hardcoded relative to the repo root as
`isar_eisbach_comparison/...`, which is now `archive/isar_eisbach_comparison/...`. Those
paths are stale. They have deliberately **not** been fixed — fixing them would imply the
scripts are expected to run, and they are not.
