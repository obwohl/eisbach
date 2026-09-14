"""Experiment 11 — does forecasting on a 15-minute grid beat an hourly one?

The comparison is easy to get wrong, so the design is the substance here.

**The trap.** Score a 15-minute forecast on 15-minute steps and an hourly forecast on
hourly steps and the two numbers are not comparable. They are averages of |error| over
different instants, against different truths: our hourly series elsewhere is the *mean* of
four samples, and a mean is smoother than what it averages. Predicting a smoothed target
is easier, so the hourly model would win a race it never ran.

**The fix, in three parts.**

1. *One target.* The hourly series here is the 15-minute series **subsampled** at the full
   hour, not averaged. It is then a strict subset: identical values at identical instants.
2. *One set of scored instants.* Both models are scored at the same 96 full-hour marks.
   The 15-minute model simply has its intermediate steps ignored. With the same truth at
   the same times, MAE and CRPS are directly comparable and need no frequency correction.
3. *Context stated honestly.* TimesFM's context is 15 360 **steps**, whatever a step is.
   At 15 minutes that is 160 days; at an hour it is 640. So a quarter-hourly model cannot
   see a year, and the sweep in exp1 showed a year is worth roughly 12 % MAE. That is a
   real property of the choice, not an artefact, so it is measured both ways: against an
   hourly model given the *same 160 days*, and against one given the year it wants.

**What is approximated.** Bright Sky has nothing finer than hourly, so the weather
covariates are interpolated onto the 15-minute grid. That is an approximation, and it
does not favour the fine model — it hands it no information the hourly model lacks.

The secondary question, whether the finer grid captures anything *within* the hour, is
reported separately and flagged: comparing there needs the hourly forecast interpolated,
and interpolating quantiles assumes the errors at neighbouring steps move together. For a
smooth, strongly autocorrelated river that is nearly true, but it is an assumption, so
that table is indicative and the headline does not rest on it.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

logger = logging.getLogger(__name__)

TARGET = "eisbach"
BATCH = 2
HORIZON_HOURS = 96

#: What TimesFM can look back over, in steps. 640 days hourly, 160 days quarter-hourly.
MAX_STEPS = 15360
#: The hourly context exp1 found best, and the one that matches the fine model's reach.
HOURLY_BEST = 8760
HOURLY_MATCHED = MAX_STEPS // 4          # 3840 hours = the same 160 days

PAST_ONLY = ["isar_lenggries", "isar_toelz", "q_lenggries", "q_toelz_kw"]
FUTURE = ["airtemp", "t_catchment"]


def load_fine() -> pd.DataFrame:
    """15-minute gauges joined with hourly weather interpolated onto the same grid."""
    df = pd.read_csv(bench.CACHE / "stations_15min.csv", index_col=0, parse_dates=[0])
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    df = df.asfreq("15min")

    w = pd.read_csv(bench.CACHE / "weather_south.csv", index_col=0, parse_dates=[0])
    w.index = pd.DatetimeIndex(w.index).tz_convert("UTC")
    w["t_catchment"] = w[[f"t_{s}" for s in ("toelz", "lenggries", "kochel", "garmisch")]].mean(axis=1)
    w = w.rename(columns={"t_muenchen": "airtemp"})[["airtemp", "t_catchment"]]
    # Interpolated, not measured. Stated in the docstring and again here so a reader of
    # the code does not mistake these for quarter-hourly observations.
    w = w.reindex(df.index.union(w.index)).interpolate(method="time").reindex(df.index)
    return df.join(w)


def window(df: pd.DataFrame, anchor: pd.Timestamp, *, steps_per_hour: int,
           context_steps: int, horizon_steps: int):
    """Context and horizon for one anchor at the given resolution."""
    grid = df.iloc[::4] if steps_per_hour == 1 else df
    pos = grid.index.get_loc(anchor)
    lo = pos - context_steps + 1
    if lo < 0 or pos + horizon_steps >= len(grid):
        return None
    ctx = grid.iloc[lo:pos + 1]
    hor = grid.iloc[pos + 1:pos + 1 + horizon_steps]
    return ctx, hor


def model_inputs(ctx: pd.DataFrame, hor: pd.DataFrame):
    """Model inputs for one window, with missing values filled as PyTorch fills them.

    Forced on regardless of backend: this experiment was run and reported on MLX, where
    the preparation is the difference between a forecast and a column of NaN, and the
    numbers in REPORT_exp11.md are the prepared ones. See ``bench.prepare_inputs``.
    """
    target = ctx[TARGET].to_numpy(dtype=np.float32)
    po = np.stack([ctx[c].to_numpy(dtype=np.float32) for c in PAST_ONLY])
    pf = np.stack([
        np.concatenate([ctx[c].to_numpy(dtype=np.float32),
                        hor[c].to_numpy(dtype=np.float32)]) for c in FUTURE])
    return bench.prepare_inputs(target, po, pf, force=True)


def evaluate(fc, df: pd.DataFrame, anchors, *, label: str, steps_per_hour: int,
             context_steps: int) -> tuple[list[dict], dict]:
    """Forecast at one resolution; score only at the full-hour marks."""
    horizon_steps = HORIZON_HOURS * steps_per_hour
    # 0-based positions of the full hours inside the horizon.
    hour_marks = np.arange(steps_per_hour - 1, horizon_steps, steps_per_hour)
    grid = df.iloc[::4] if steps_per_hour == 1 else df
    truth_series = grid[TARGET]

    rows, traces = [], {}
    for i in range(0, len(anchors), BATCH):
        chunk = anchors[i:i + BATCH]
        contexts, po_list, pf_list, metas = [], [], [], []
        for ts in chunk:
            got = window(df, ts, steps_per_hour=steps_per_hour,
                         context_steps=context_steps, horizon_steps=horizon_steps)
            if got is None:
                continue
            ctx, hor = got
            target, po, pf = model_inputs(ctx, hor)
            contexts.append(target)
            po_list.append(po)
            pf_list.append(pf)
            metas.append((ts, hor.index))
        if not contexts:
            continue
        outs = list(fc.predict_batch(contexts, horizon=horizon_steps,
                                     past_only_covariates=po_list,
                                     past_future_covariates=pf_list,
                                     return_quantiles=True))
        for (ts, hor_index), o in zip(metas, outs, strict=True):
            q_full = o.quantiles if o.quantiles.ndim == 2 else o.quantiles[0]
            if not np.isfinite(q_full).all():
                raise ValueError(f"{label} at {ts}: non-finite forecast quantiles")
            traces[ts] = (hor_index, q_full)
            q = q_full[hour_marks]
            targets = hor_index[hour_marks]
            run = bench.Run(label=label, reference_time=ts, target_times=targets,
                            quantiles=q,
                            truth=truth_series.reindex(targets).to_numpy(dtype=float))
            rows.extend(bench.score(run, buckets=bench.FINE_BUCKETS))
    return rows, traces


def pick_anchors(df: pd.DataFrame, n: int) -> list[pd.Timestamp]:
    """Full-hour anchors where both resolutions have a complete context and horizon."""
    from exp6_upstream import _stratify

    hourly = df.iloc[::4]
    need = [TARGET, *PAST_ONLY, *FUTURE]
    candidates = []
    for pos in range(HOURLY_BEST, len(hourly) - HORIZON_HOURS, 13):
        ts = hourly.index[pos]
        block_h = hourly.iloc[pos - HOURLY_BEST + 1: pos + 1 + HORIZON_HOURS]
        if block_h[need].notna().mean().min() < 0.95:
            continue
        fpos = df.index.get_loc(ts)
        if fpos < MAX_STEPS:
            continue
        block_f = df.iloc[fpos - MAX_STEPS + 1: fpos + 1 + HORIZON_HOURS * 4]
        if block_f[need].notna().mean().min() < 0.95:
            continue
        if not np.isfinite(hourly[TARGET].iloc[pos]):
            continue
        candidates.append(ts)
    return _stratify(candidates, n)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = load_fine()
    anchors = pick_anchors(df, 120)
    logger.info("%d anchors, %s .. %s", len(anchors), anchors[0], anchors[-1])
    logger.info("per year: %s", pd.Series(anchors).dt.year.value_counts().sort_index().to_dict())

    fc = bench.load_forecaster()

    variants = [
        ("hourly, 1 year of context", 1, HOURLY_BEST),
        ("hourly, 160 days (matched)", 1, HOURLY_MATCHED),
        ("15-minute, 160 days (its max)", 4, MAX_STEPS),
    ]
    rows, traces = [], {}
    for label, sph, ctx in variants:
        t0 = time.time()
        r, tr = evaluate(fc, df, anchors, label=label, steps_per_hour=sph, context_steps=ctx)
        rows += r
        traces[label] = tr
        logger.info("%-32s %.0fs", label, time.time() - t0)

    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "exp11_resolution.csv", index=False)

    print(f"\n=== scored at the same 96 full-hour marks, {len(anchors)} windows ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "crps", "cov_80", "width_80"]]
          .to_string(index=False))
    for metric in ("mae", "crps"):
        for ref in ("hourly, 160 days (matched)", "hourly, 1 year of context"):
            print(f"\n--- {metric.upper()} against '{ref}', paired ---")
            p = bench.paired(scores, ref, metric=metric)
            print(p[["label", metric, "pct", "ci_lo", "ci_hi", "better_in", "verdict"]]
                  .to_string(index=False))
    print("\n--- MAE by lead bucket ---")
    print(bench.pool(scores, by=["label", "lead_lo"]).pivot(
        index="label", columns="lead_lo", values="mae").round(3).to_string())

    # Secondary, and indicative only: everything the fine model predicts, against the
    # hourly forecast interpolated onto the same grid. Interpolating quantiles assumes
    # neighbouring errors move together, which is nearly true here and still an assumption.
    fine = traces["15-minute, 160 days (its max)"]
    coarse = traces["hourly, 1 year of context"]
    truth = df[TARGET]
    mae_f, mae_c = [], []
    for ts, (idx_f, q_f) in fine.items():
        if ts not in coarse:
            continue
        idx_c, q_c = coarse[ts]
        med_c = pd.Series(q_c[:, 4], index=idx_c).reindex(idx_f).interpolate().bfill()
        y = truth.reindex(idx_f)
        ok = y.notna().to_numpy()
        if ok.sum() < 200:
            continue
        mae_f.append(np.mean(np.abs(q_f[ok, 4] - y.to_numpy()[ok])))
        mae_c.append(np.mean(np.abs(med_c.to_numpy()[ok] - y.to_numpy()[ok])))
    if mae_f:
        d = np.array(mae_f) - np.array(mae_c)
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(8000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"\n--- all 384 quarter-hour marks, {len(d)} windows (indicative) ---")
        print(f"  15-minute forecast      MAE {np.mean(mae_f):.4f}")
        print(f"  hourly, interpolated    MAE {np.mean(mae_c):.4f}")
        print(f"  difference {d.mean():+.4f} ({100*d.mean()/np.mean(mae_c):+.1f} %), "
              f"95 % CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
