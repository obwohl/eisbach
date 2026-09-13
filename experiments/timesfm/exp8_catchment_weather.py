"""Experiment 8 — weather where the river is made, not where it is measured.

Munich's air temperature describes the last kilometre of the Eisbach. The water arriving
tomorrow was rained on and warmed in the Isarwinkel and the Loisach valley. Rain is the
sharper version of the argument: a storm over Munich adds almost nothing to the Isar,
while the same storm over Lenggries arrives as volume, at its own temperature.

Every series here is a **past-and-future** covariate, because Bright Sky serves the DWD
forecast at any coordinate — so unlike the upstream gauges, these would be genuinely
known ahead at run time. In this historical sweep their horizon is the observed value,
which makes every run an oracle run; `exp4_replay` puts that substitution at 0.285
against 0.290 for Munich air, and rain forecasts are worse than temperature forecasts, so
treat any rain gain here as an upper bound.

Scored on six-hour buckets over the first day: an upstream signal can only be ahead of
the river while the horizon is shorter than the travel time, and Beuerberg is hours away,
not half a day.
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
from exp6_upstream import CONTEXT, TARGET, load, pick_anchors  # noqa: E402

logger = logging.getLogger(__name__)

BATCH = 8
SOUTH = ["toelz", "lenggries", "kochel", "garmisch"]


def load_all() -> pd.DataFrame:
    df = load()
    w = pd.read_csv(bench.CACHE / "weather_south.csv", index_col=0, parse_dates=[0])
    w.index = pd.DatetimeIndex(w.index).tz_convert("UTC")
    df = df.join(w, how="left")
    # One aggregate each: the catchment is upstream of everything, and four near-identical
    # series cost four variate slots to say one thing.
    df["t_catchment"] = df[[f"t_{s}" for s in SOUTH]].mean(axis=1)
    df["rain_catchment"] = df[[f"rain_{s}" for s in SOUTH]].mean(axis=1)
    # Rain matters as accumulated volume, not as the millimetre in one hour.
    df["rain_catchment_24h"] = df["rain_catchment"].rolling(24, min_periods=12).sum()
    df["rain_muenchen_24h"] = df["rain_muenchen"].rolling(24, min_periods=12).sum()
    return df


def evaluate(fc, df, anchors, truth, *, label: str, future: list[str],
             past_only: list[str] | None = None) -> list[dict]:
    """``future`` are handed over context and horizon; ``past_only`` over context alone."""
    past_only = past_only or []
    idx = df.index
    rows = []
    for i in range(0, len(anchors), BATCH):
        chunk = anchors[i:i + BATCH]
        contexts, po_list, pf_list, metas = [], [], [], []
        for ts in chunk:
            pos = idx.get_loc(ts)
            lo = pos - CONTEXT + 1
            contexts.append(truth.iloc[lo:pos + 1].to_numpy(dtype=np.float32))
            po_list.append(
                np.stack([df[c].iloc[lo:pos + 1].to_numpy(dtype=np.float32) for c in past_only])
                if past_only else None)
            pf_list.append(
                np.stack([df[c].iloc[lo:pos + 1 + bench.HORIZON].to_numpy(dtype=np.float32)
                          for c in future]) if future else None)
            metas.append((ts, idx[pos + 1: pos + 1 + bench.HORIZON]))
        outs = list(fc.predict_batch(contexts, horizon=bench.HORIZON,
                                     past_only_covariates=po_list,
                                     past_future_covariates=pf_list,
                                     return_quantiles=True))
        for (ts, targets), o in zip(metas, outs, strict=True):
            q = o.quantiles if o.quantiles.ndim == 2 else o.quantiles[0]
            run = bench.Run(label=label, reference_time=ts, target_times=targets,
                            quantiles=q, truth=truth.reindex(targets).to_numpy(dtype=float))
            rows.extend(bench.score(run, buckets=bench.FINE_BUCKETS))
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from timesfm3 import TimesFM3Forecaster

    df = load_all()
    truth = df[TARGET]
    needed = ["airtemp", "t_catchment", "rain_catchment", "isar_toelz", "isar_lenggries",
              "q_toelz_kw", "q_lenggries", TARGET]
    anchors = pick_anchors(df, needed, n=150)
    logger.info("%d anchors, %s .. %s", len(anchors), anchors[0], anchors[-1])

    fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")
    chain = ["isar_lenggries", "isar_toelz", "q_lenggries", "q_toelz_kw"]

    variants = [
        ("air München (baseline)", ["airtemp"], []),
        ("air München + air catchment", ["airtemp", "t_catchment"], []),
        ("air catchment instead of München", ["t_catchment"], []),
        ("air München + air Tölz", ["airtemp", "t_toelz"], []),
        ("air München + air Garmisch", ["airtemp", "t_garmisch"], []),
        ("air München + rain München", ["airtemp", "rain_muenchen"], []),
        ("air München + rain catchment", ["airtemp", "rain_catchment"], []),
        ("air München + rain catchment 24h", ["airtemp", "rain_catchment_24h"], []),
        ("air München + rain Garmisch", ["airtemp", "rain_garmisch"], []),
        ("air + air catchment + rain catchment", ["airtemp", "t_catchment", "rain_catchment"], []),
        ("air + upstream chain", ["airtemp"], chain),
        ("air + air catchment + upstream chain", ["airtemp", "t_catchment"], chain),
        ("everything above", ["airtemp", "t_catchment", "rain_catchment_24h"], chain),
    ]

    rows = []
    for label, future, past_only in variants:
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, future=future, past_only=past_only)
        logger.info("%-40s %.0fs", label, time.time() - t0)

    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "exp8_catchment.csv", index=False)

    print(f"\n=== catchment weather, {len(anchors)} windows (oracle future) ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "crps", "cov_80", "width_80"]]
          .to_string(index=False))
    for metric in ("mae", "crps"):
        print(f"\n--- {metric.upper()} against the baseline, paired ---")
        p = bench.paired(scores, "air München (baseline)", metric=metric)
        print(p[["label", metric, "pct", "ci_lo", "ci_hi", "better_in", "verdict"]]
              .to_string(index=False))
        print(f"\n--- {metric.upper()} by lead bucket ---")
        print(bench.pool(scores, by=["label", "lead_lo"]).pivot(
            index="label", columns="lead_lo", values=metric).round(3).to_string())


if __name__ == "__main__":
    main()
