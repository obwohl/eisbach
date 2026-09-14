# Handoff: run experiment 11 (15-minute vs hourly) locally

Copy everything below the line into the local agent.

---

You are working in the `obwohl/eisbach` repository, on the branch
**`claude/sleepy-brown-950428`**. Check it out and pull before you start:

```bash
git fetch origin && git checkout claude/sleepy-brown-950428 && git pull
```

## What this repo is

A thrice-daily probabilistic forecast of the Eisbach's water temperature. Read
`CLAUDE.md` first — it is short and its rules are real.

`experiments/timesfm/` is a **side branch that must not touch the production pipeline**.
Nothing under it is imported by `main.py` or `eisbach/`, runs in CI, or writes to
`data/archive/`. Keep it that way. `data/archive/` is append-only and irreplaceable.

Read `experiments/timesfm/README.md` and then the module docstring of
`experiments/timesfm/exp11_resolution.py` in full before running anything. The docstring
is the experiment's design and most of its value.

## Your task

Run `exp11_resolution.py` and report what it prints. That is the whole job. It answers:
does forecasting the Eisbach on a 15-minute grid beat an hourly one?

### Setup

```bash
pip install 'timesfm[mlx]'          # Apple silicon: several times faster than torch on CPU
export TIMESFM_BACKEND=mlx          # bench.load_forecaster() reads this
export HF_HOME="$PWD/data/experiments/hf"
```

`data/experiments/` is gitignored, so the datasets are not in your clone and must be
rebuilt. Each takes 15 to 30 minutes and only needs doing once:

```bash
python experiments/timesfm/build_stations.py        # 29 gauge series, hourly
python experiments/timesfm/build_weather_south.py   # 6 locations, air/rain/solar, hourly
python experiments/timesfm/build_fine.py            # 5 series at native 15 minutes
```

They scrape GKD Bayern and Bright Sky. Be patient with them; GKD returns HTTP 500
sporadically and the fetchers retry. `build_stations.py` is incremental — it keeps what
the cache already holds.

Then:

```bash
python experiments/timesfm/exp11_resolution.py 2>&1 | tee /tmp/exp11.log
```

Expect it to take a while: the 15-minute variant uses a 15 360-step context, which is the
model's maximum.

### What to report back

The printed tables, verbatim, plus your own reading of them:

* the pooled table, and the paired comparisons against **both** hourly references;
* MAE and CRPS by lead bucket — the buckets are 6-hourly over the first day on purpose;
* the secondary quarter-hour table, clearly marked as indicative;
* your verdict: is 15-minute worth it, and if so at which leads?

Say plainly if the answer is "no" or "not distinguishable from zero". A negative result
here is a useful result and nobody is invested in the alternative.

## Traps — do not "fix" these, they are deliberate

1. **The hourly series is subsampled from the 15-minute series at the full hour, not
   averaged.** This is what makes the two models predict the same numbers at the same
   instants. Averaging would give the hourly model a smoother target and it would win a
   race it never ran. If you find yourself reaching for `.resample("1h").mean()`, stop.
2. **Both models are scored only at the 96 full-hour marks.** The fine model's
   intermediate steps are ignored on purpose. Do not "improve" this by scoring the fine
   model on all 384 steps in the headline comparison — that compares different quantities.
3. **The context is 15 360 *steps*, not hours.** At 15 minutes that is 160 days, at an
   hour 640. The fine model therefore cannot see a year, and `exp1_context.csv` measured a
   year as worth about 12 % MAE. That is why there are two hourly references: one matched
   to 160 days, one given the full year. Report both.
4. **The weather covariates are hourly and get interpolated onto the 15-minute grid.**
   Bright Sky has nothing finer. This is an approximation; it hands the fine model no
   information the hourly model lacks. Say so if you write it up.
5. **Every comparison is paired** via `bench.paired()`, because the windows differ from
   each other far more than the variants do. Unpaired intervals here are roughly ten times
   too wide to see anything. The intervals are unadjusted and each row carries `n_tested`.

## Ground rules

* Before any commit: `pytest -q` and `ruff check .` from the repo root, both clean.
* Do not modify anything under `eisbach/`, `main.py`, `tests/`, `data/archive/` or
  `.github/`. If you believe a production bug is in your way, report it, do not fix it.
* If you must change experiment code — a genuine bug, not a preference — commit it to a
  **new branch** `local/exp11-fixes` and say so. Do not push to
  `claude/sleepy-brown-950428`; another agent is pushing to it concurrently and you will
  collide.
* Do not run `exp12_pairs.py`; it is running elsewhere right now.
* The result CSVs under `data/experiments/` are gitignored. Paste the tables back rather
  than trying to commit them.

## Context you may want

* `experiments/timesfm/README.md` — what has been measured so far and the two cautions.
* `docs/PRD.md` §R6 — why covariates are the theme of this whole investigation.
* Measured throughput with six covariates on a 4-core container, for calibrating your
  expectations: 5.0 s per window at 8760 h of context, 1.55 s at 4096, 0.75 s at 2048,
  0.42 s at 1024. MLX on Apple silicon should be materially faster.
