"""Experiment 7 — do experiment 6's winners hold up on a much wider window set?

Experiment 6 required every candidate series to be nearly complete over every window,
which is the right rule for comparing combinations but an expensive one: two gauges —
the Isar at Munich, which is missing 2020-01 to 2022-07, and the Rißbachklamm — cut the
usable anchors to 80, all between December 2023 and July 2025.

This drops those two from the candidate set, which unlocks the rest of the record, and
re-runs only the combinations worth re-running. Same rules otherwise: upstream gauges are
past-only, air temperature over the horizon is observed and therefore oracle, and every
combination sees the same windows.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from exp6_upstream import CHAIN_Q, CHAIN_T, LOCAL, TARGET, evaluate, load, pick_anchors, report  # noqa: E402

logger = logging.getLogger(__name__)

#: The two gauges whose gaps cost the most windows. Dropped here, measured in exp6.
EXCLUDED = {"isar_muenchen", "rissbach_klamm"}

N_WINDOWS = 150


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from timesfm3 import TimesFM3Forecaster

    df = load()
    truth = df[TARGET]
    candidates = [c for c in CHAIN_T + CHAIN_Q + LOCAL
                  if c in df.columns and c not in EXCLUDED]

    anchors = pick_anchors(df, candidates + [TARGET, "airtemp"], n=N_WINDOWS)
    logger.info("%d anchors, %s .. %s", len(anchors), anchors[0], anchors[-1])
    logger.info("years covered: %s",
                sorted({a.year for a in anchors}))

    fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

    chain_t = [c for c in CHAIN_T if c in candidates]
    chain_q = [c for c in CHAIN_Q if c in candidates]
    # Experiment 6 found exactly two series that clear an unadjusted significance bar,
    # and they are the two the travel time predicts: Lenggries and Bad Tölz, the mid-Isar
    # temperature gauges. Neither survives a correction for having tested 21 of them, so
    # this window set is the replication that decides it.
    sets = {
        "water only": None,
        "+ air": [],
        "+ air + Lenggries (T)": ["isar_lenggries"],
        "+ air + Lenggries (T+Q)": ["isar_lenggries", "q_lenggries"],
        "+ air + Tölz (T)": ["isar_toelz"],
        "+ air + Tölz (T+Q)": ["isar_toelz", "q_toelz_kw"],
        "+ air + Lenggries + Tölz (T)": ["isar_lenggries", "isar_toelz"],
        "+ air + Eisbach Q": ["q_eisbach"],
        "+ air + whole T chain": chain_t,
        "+ air + whole T chain + their Q": chain_t + chain_q,
        "+ air + everything": candidates,
    }

    rows = []
    for label, cols in sets.items():
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label,
                         past_only=[] if cols is None else cols,
                         with_air=cols is not None)
        logger.info("%-34s %2d past-only  %.0fs", label,
                    0 if cols is None else len(cols), time.time() - t0)

    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "exp7_broad.csv", index=False)
    report(scores, df, f"{len(anchors)} windows, {anchors[0]:%Y-%m} .. {anchors[-1]:%Y-%m}")

    print("\n--- by year ---")
    s = scores.copy()
    s["year"] = s.reference_time.dt.year
    print(bench.pool(s, by=["label", "year"]).pivot(
        index="label", columns="year", values="mae").round(3).to_string())


if __name__ == "__main__":
    main()
