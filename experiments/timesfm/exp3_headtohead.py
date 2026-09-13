"""Experiment 3 — TimesFM 3.0 zero-shot against our own archived live forecasts.

Apples to apples: the same anchors, the same 96-hour windows, the same truth, and both
models' quantiles integrated over the same decile grid.

The comparison is deliberately unfair *against* TimesFM in the univariate variant: DUET
saw the DWD forecast for the whole horizon, TimesFM sees nothing but the river's own
past. The ``+air(oracle)`` variant is the other extreme and is labelled as oracle —
observed air temperature over the horizon is weather nobody could have known.

Truth is the raw gauge series, not the production observation store: that store holds
interpolated values before 2026-09-10 (PRD R3) and one 154.4 C instrument fault, and
scoring a model against an invented number tells you nothing.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

logger = logging.getLogger(__name__)

CONTEXT = 8760
BATCH = 4


def duet_runs(truth_series: pd.Series) -> list[bench.Run]:
    """The archived live forecasts, re-scored on the shared decile grid."""
    fc = bench.archived_forecasts("live")
    cols = [f"wassertemp_q{q}" for q in bench.DUET_QUANTILES]
    runs = []
    for ref, g in fc.groupby("reference_time"):
        g = g.sort_values("target_time")
        if len(g) < bench.HORIZON:
            continue
        g = g.iloc[:bench.HORIZON]
        values = g[cols].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        targets = pd.DatetimeIndex(g["target_time"])
        runs.append(bench.Run(
            label="duet_live", reference_time=ref, target_times=targets,
            quantiles=bench.to_deciles(values, bench.DUET_QUANTILES),
            truth=truth_series.reindex(targets).to_numpy(dtype=float),
            extra={"model_id": g["model_id"].iloc[0]}))
    return runs


def timesfm_runs(forecaster, df: pd.DataFrame, anchors, *, label: str,
                 past_future: list[str] | None = None,
                 past_only: list[str] | None = None,
                 future_offset: float = 0.0) -> list[bench.Run]:
    """One TimesFM variant over the given anchors.

    ``past_future`` columns are handed to the model over context *and* horizon using
    their observed values — that is an oracle. ``future_offset`` shifts only the horizon
    part, which is the covariate-sensitivity probe.
    """
    series = df["eisbach"]
    idx = df.index
    runs = []
    for i in range(0, len(anchors), BATCH):
        chunk = anchors[i:i + BATCH]
        contexts, po_list, pf_list, metas = [], [], [], []
        for ts in chunk:
            pos = idx.get_loc(ts)
            lo = pos - CONTEXT + 1
            contexts.append(series.iloc[lo:pos + 1].to_numpy(dtype=np.float32))
            po_list.append(
                np.stack([df[c].iloc[lo:pos + 1].to_numpy(dtype=np.float32) for c in past_only])
                if past_only else None)
            if past_future:
                rows = []
                for c in past_future:
                    hist = df[c].iloc[lo:pos + 1].to_numpy(dtype=np.float32)
                    fut = df[c].iloc[pos + 1:pos + 1 + bench.HORIZON].to_numpy(dtype=np.float32)
                    rows.append(np.concatenate([hist, fut + future_offset]))
                pf_list.append(np.stack(rows))
            else:
                pf_list.append(None)
            metas.append((ts, idx[pos + 1:pos + 1 + bench.HORIZON]))
        outs = list(forecaster.predict_batch(
            contexts, horizon=bench.HORIZON, past_only_covariates=po_list,
            past_future_covariates=pf_list, return_quantiles=True))
        for (ts, targets), o in zip(metas, outs, strict=True):
            q = o.quantiles if o.quantiles.ndim == 2 else o.quantiles[0]
            runs.append(bench.Run(label=label, reference_time=ts, target_times=targets,
                                  quantiles=q, truth=np.full(bench.HORIZON, np.nan)))
    return runs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from timesfm3 import TimesFM3Forecaster

    df = bench.load_dataset()
    truth = df["eisbach"]

    duet = duet_runs(truth)
    logger.info("%d archived live runs with a full horizon", len(duet))

    # Only anchors whose context and horizon the long dataset actually covers.
    idx = df.index
    anchors = [r.reference_time for r in duet
               if r.reference_time in idx
               and idx.get_loc(r.reference_time) >= CONTEXT
               and idx.get_loc(r.reference_time) + bench.HORIZON < len(idx)
               and np.isfinite(truth.reindex(r.target_times).to_numpy(dtype=float)).sum() >= 48]
    logger.info("%d anchors usable for a head to head", len(anchors))
    duet = [r for r in duet if r.reference_time in set(anchors)]

    fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

    variants = [
        dict(label="timesfm_uni", past_future=None, past_only=None),
        dict(label="timesfm_air_past", past_future=None, past_only=["airtemp"]),
        dict(label="timesfm_air_oracle", past_future=["airtemp"], past_only=None),
        dict(label="timesfm_air_oracle_isar_past", past_future=["airtemp"], past_only=["isar"]),
    ]

    all_runs = list(duet)
    for v in variants:
        runs = timesfm_runs(fc, df, anchors, **v)
        for r in runs:
            r.truth = truth.reindex(r.target_times).to_numpy(dtype=float)
        all_runs.extend(runs)
        logger.info("%s done", v["label"])

    rows = []
    for r in all_runs:
        rows.extend(bench.score(
            r, diurnal=bench.diurnal_baseline(truth, r.target_times),
            persistence=float(truth.loc[r.reference_time])
            if r.reference_time in truth.index else float("nan")))
    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "exp3_headtohead.csv", index=False)

    print(f"\n=== head to head on {len(anchors)} identical windows, decile grid ===")
    print(bench.pool(scores).to_string(index=False))
    print("\n=== MAE by lead bucket ===")
    print(bench.pool(scores, by=["label", "lead_lo"]).pivot(
        index="label", columns="lead_lo", values="mae").round(3).to_string())


if __name__ == "__main__":
    main()
