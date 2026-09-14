"""Experiment 2 — do the covariates actually reach the water channel?

Two questions, one script.

**Does the weather help?** Univariate against air temperature as a past-only covariate,
as a past-and-future covariate (oracle), and with the Isar alongside.

**Does the absolute level survive?** PRD R6 established that DUET is blind to a +3 °C
shift of ``airtemp_96``: the forecast moves by 1.9e-6 °C, because RevIN normalises every
covariate per window. TimesFM normalises per variate too, so the same trap is plausible
and the same probe answers it — shift only the horizon part of the future covariate and
see whether the water forecast moves at all.
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
from exp1_context import pick_anchors  # noqa: E402
from exp3_headtohead import timesfm_runs  # noqa: E402

logger = logging.getLogger(__name__)

CONTEXT = 8760
VARIANTS = (
    dict(label="uni", past_future=None, past_only=None),
    dict(label="+air(past)", past_future=None, past_only=["airtemp"]),
    dict(label="+air(oracle)", past_future=["airtemp"], past_only=None),
    dict(label="+air+pressure(oracle)", past_future=["airtemp", "pressure"], past_only=None),
    dict(label="+air(oracle)+isar(past)", past_future=["airtemp"], past_only=["isar"]),
    dict(label="+isar(past)", past_future=None, past_only=["isar"]),
)


def shift_probe(fc, df, anchors, offsets=(0.0, 3.0, -3.0, 10.0)) -> pd.DataFrame:
    """How far does the water forecast move when the future air temperature is shifted?"""
    base = None
    rows = []
    for off in offsets:
        runs = timesfm_runs(fc, df, anchors, label=f"shift{off:+.0f}",
                            past_future=["airtemp"], future_offset=off)
        med = np.stack([r.median for r in runs])
        if base is None:
            base = med
            continue
        delta = med - base
        rows.append({
            "offset_C": off,
            "mean_abs_delta": float(np.abs(delta).mean()),
            "max_abs_delta": float(np.abs(delta).max()),
            "delta_at_h96": float(delta[:, -1].mean()),
            "delta_at_h24": float(delta[:, 23].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from timesfm3 import TimesFM3Forecaster

    df = bench.load_dataset()
    truth = df["eisbach"]

    # Anchors need every covariate present too, so require a complete recent window.
    anchors = pick_anchors(truth, start="2023-06-01", end="2026-09-05",
                           every_hours=97, max_context=CONTEXT)
    idx = df.index
    anchors = [ts for ts in anchors
               if df[["airtemp", "isar"]].iloc[idx.get_loc(ts) - CONTEXT + 1:
                                               idx.get_loc(ts) + 1 + bench.HORIZON]
               .notna().all(axis=None)]
    logger.info("%d anchors with complete covariates", len(anchors))

    fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

    rows = []
    for v in VARIANTS:
        t0 = time.time()
        runs = timesfm_runs(fc, df, anchors, **v)
        for r in runs:
            r.truth = truth.reindex(r.target_times).to_numpy(dtype=float)
            rows.extend(bench.score(
                r, diurnal=bench.diurnal_baseline(truth, r.target_times),
                persistence=float(truth.loc[r.reference_time])))
        logger.info("%-24s %d runs, %.0fs", v["label"], len(runs), time.time() - t0)

    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "exp2_covariates.csv", index=False)
    print(f"\n=== covariate variants, {len(anchors)} windows, decile grid ===")
    print(bench.pool(scores).to_string(index=False))
    print("\n=== MAE by lead bucket ===")
    print(bench.pool(scores, by=["label", "lead_lo"]).pivot(
        index="label", columns="lead_lo", values="mae").round(3).to_string())

    probe_anchors = anchors[:24]
    print(f"\n=== shift probe: future air temperature moved, {len(probe_anchors)} windows ===")
    probe = shift_probe(fc, df, probe_anchors)
    probe.to_csv(bench.CACHE / "exp2_shift_probe.csv", index=False)
    print(probe.round(4).to_string(index=False))
    print("\n(DUET's answer to the same probe, from PRD R6: 1.9e-6 °C for +3 °C.)")


if __name__ == "__main__":
    main()
