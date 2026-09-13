"""Experiment 1 — does a longer context help? Univariate, zero-shot.

Sweeps the context length from DUET's 384 hours up to TimesFM's 15 360, on one fixed
set of anchors, so the only thing that changes is how far back the model may look.
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

CONTEXTS = (384, 1024, 2048, 4380, 8760, 15360)
BATCH = 8


def pick_anchors(series: pd.Series, *, start: str, end: str, every_hours: int,
                 max_context: int, min_complete: float = 0.98) -> list[pd.Timestamp]:
    """Anchors whose longest context and whose 96-hour target are both well measured."""
    idx = series.index
    candidates = pd.date_range(start=start, end=end, freq=f"{every_hours}h", tz="UTC")
    anchors = []
    for ts in candidates:
        if ts not in idx:
            continue
        pos = idx.get_loc(ts)
        if pos < max_context or pos + bench.HORIZON >= len(idx):
            continue
        ctx = series.iloc[pos - max_context + 1: pos + 1]
        tgt = series.iloc[pos + 1: pos + 1 + bench.HORIZON]
        if ctx.notna().mean() < min_complete or not np.isfinite(ctx.iloc[-1]):
            continue
        if tgt.notna().mean() < 0.9:
            continue
        anchors.append(ts)
    return anchors


def run(forecaster, series: pd.Series, anchors, context_len: int) -> list[bench.Run]:
    idx = series.index
    out = []
    for i in range(0, len(anchors), BATCH):
        chunk = anchors[i:i + BATCH]
        contexts, metas = [], []
        for ts in chunk:
            pos = idx.get_loc(ts)
            ctx = series.iloc[pos - context_len + 1: pos + 1].to_numpy(dtype=np.float32)
            contexts.append(ctx)
            metas.append((ts, idx[pos + 1: pos + 1 + bench.HORIZON],
                          series.iloc[pos + 1: pos + 1 + bench.HORIZON].to_numpy(dtype=float)))
        outs = list(forecaster.predict_batch(contexts, horizon=bench.HORIZON,
                                             return_quantiles=True))
        for (ts, targets, truth), o in zip(metas, outs, strict=True):
            out.append(bench.Run(label=f"timesfm_uni_ctx{context_len}", reference_time=ts,
                                 target_times=targets, quantiles=o.quantiles, truth=truth,
                                 extra={"context_len": context_len}))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from timesfm3 import TimesFM3Forecaster

    df = bench.load_dataset()
    series = df["eisbach"]

    anchors = pick_anchors(series, start="2023-01-01", end="2026-09-05",
                           every_hours=73, max_context=max(CONTEXTS))
    logger.info("%d anchors, %s .. %s", len(anchors), anchors[0], anchors[-1])

    fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

    rows = []
    for ctx_len in CONTEXTS:
        t0 = time.time()
        runs = run(fc, series, anchors, ctx_len)
        for r in runs:
            rows.extend(bench.score(
                r, diurnal=bench.diurnal_baseline(series, r.target_times),
                persistence=float(series.loc[r.reference_time])))
        logger.info("ctx=%d: %d runs in %.1fs (%.2fs/run)",
                    ctx_len, len(runs), time.time() - t0, (time.time() - t0) / len(runs))

    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "exp1_context.csv", index=False)

    print("\n=== univariate zero-shot, pooled over all leads ===")
    print(bench.pool(scores, by=["context_len"]).to_string(index=False))
    print("\n=== by lead bucket (MAE) ===")
    piv = bench.pool(scores, by=["context_len", "lead_lo"]).pivot(
        index="context_len", columns="lead_lo", values="mae")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
