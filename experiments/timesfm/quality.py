"""Data quality report for the long hourly dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "experiments"


def load() -> pd.DataFrame:
    df = pd.read_csv(CACHE / "long_hourly.csv", index_col=0, parse_dates=[0])
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df.asfreq("1h")


def gaps(series: pd.Series) -> pd.DataFrame:
    """Contiguous runs of missing hours, longest first."""
    missing = series.isna()
    if not missing.any():
        return pd.DataFrame(columns=["start", "end", "hours"])
    block = (missing != missing.shift()).cumsum()[missing]
    rows = [
        {"start": idx[0], "end": idx[-1], "hours": len(idx)}
        for _, idx in block.groupby(block).groups.items()
    ]
    out = pd.DataFrame(rows)
    return out.sort_values("hours", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = load()
    print(f"span      {df.index.min()} .. {df.index.max()}  ({len(df)} hourly slots)")
    print(f"duplicates {df.index.duplicated().sum()}   monotonic {df.index.is_monotonic_increasing}\n")

    print(f"{'column':<20}{'present':>9}{'missing':>9}{'miss%':>8}{'min':>8}{'max':>8}{'mean':>8}")
    for col in df.columns:
        s = df[col]
        print(f"{col:<20}{s.notna().sum():>9}{s.isna().sum():>9}{100*s.isna().mean():>7.2f}%"
              f"{s.min():>8.1f}{s.max():>8.1f}{s.mean():>8.2f}")

    for col in ("eisbach", "isar", "airtemp"):
        g = gaps(df[col])
        total = int(g["hours"].sum()) if len(g) else 0
        print(f"\n--- {col}: {len(g)} gaps, {total} missing hours ---")
        for _, r in g.head(6).iterrows():
            print(f"  {r['hours']:>6} h   {r['start']:%Y-%m-%d %H:%M} .. {r['end']:%Y-%m-%d %H:%M}")
        if len(g):
            print(f"  gaps of 1h: {(g['hours'] == 1).sum()}   <=3h: {(g['hours'] <= 3).sum()}")

    # Physical plausibility of the target.
    e = df["eisbach"]
    print("\n--- eisbach plausibility ---")
    print(f"  below 0 C: {(e < 0).sum()}   above 30 C: {(e > 30).sum()}")
    step = e.diff().abs()
    print(f"  hourly |delta|: mean {step.mean():.3f}  p99 {step.quantile(0.99):.2f}  max {step.max():.2f}")
    big = step[step > 2.0]
    print(f"  jumps > 2 C/h: {len(big)}")
    for ts, v in big.sort_values(ascending=False).head(5).items():
        print(f"    {ts:%Y-%m-%d %H:%M}  {v:.2f} C   ({e.shift(1)[ts]:.1f} -> {e[ts]:.1f})")
    flat = (step == 0)
    run = (flat != flat.shift()).cumsum()[flat]
    if len(run):
        longest = run.value_counts().max()
        print(f"  longest run of identical consecutive values: {longest} h")
