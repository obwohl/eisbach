"""Experiment 5 — how do the models do when the river is actually changing?

A model looks good on a plateau by doing nothing. The question worth asking is what
happens across a cold snap or a warm spell. Every window is classified by its *swing*:
the mean water temperature over the last 24 forecast hours minus the mean over the 24
hours the run had already seen. A window with a swing near zero is a plateau; the tails
are the transitions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

SWING_BINS = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
SWING_NAMES = ["strong cooling\n(< -1.5 °C)", "cooling\n(-1.5..-0.5)", "plateau\n(±0.5)",
               "warming\n(0.5..1.5)", "strong warming\n(> 1.5 °C)"]


def swing(series: pd.Series, reference_time: pd.Timestamp) -> float:
    """Mean of forecast hours 73-96 minus mean of the last 24 observed hours."""
    before = series.loc[reference_time - pd.Timedelta(hours=23): reference_time]
    after = series.loc[reference_time + pd.Timedelta(hours=73):
                       reference_time + pd.Timedelta(hours=96)]
    if before.notna().sum() < 12 or after.notna().sum() < 12:
        return np.nan
    return float(after.mean() - before.mean())


def classify(scores: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    swings = {ts: swing(series, ts) for ts in scores.reference_time.unique()}
    scores = scores.copy()
    scores["swing"] = scores.reference_time.map(swings)
    scores["abs_swing"] = scores.swing.abs()
    scores["regime"] = pd.cut(scores.swing, SWING_BINS, labels=SWING_NAMES)
    return scores


if __name__ == "__main__":
    df = bench.load_dataset()
    series = df["eisbach"]
    for name in ("exp4_replay", "exp3_headtohead"):
        path = bench.CACHE / f"{name}.csv"
        if not path.exists():
            continue
        s = classify(pd.read_csv(path, parse_dates=["reference_time"]), series)
        s.to_csv(bench.CACHE / f"{name}_regimes.csv", index=False)
        print(f"\n################ {name} ################")
        counts = s[s.label == s.label.iloc[0]].groupby("regime", observed=False).reference_time.nunique()
        print("windows per regime:", counts.to_dict())
        print("\n=== MAE by regime ===")
        print(bench.pool(s.dropna(subset=["regime"]), by=["label", "regime"]).pivot(
            index="label", columns="regime", values="mae").round(3).to_string())
        print("\n=== CRPS by regime ===")
        print(bench.pool(s.dropna(subset=["regime"]), by=["label", "regime"]).pivot(
            index="label", columns="regime", values="crps").round(3).to_string())
        print("\n=== the 25 % of windows that move most ===")
        cut = s.abs_swing.quantile(0.75)
        print(bench.pool(s[s.abs_swing >= cut]).to_string(index=False))
