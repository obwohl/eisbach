"""The gauges at their native 15-minute resolution.

Everything else in this tree averages GKD's 15-minute samples to hours. This keeps them,
so the question "does a finer grid forecast better?" can be asked without the answer
being decided by how the data was prepared.

Only the series experiments 6 to 8 left standing are fetched — the Eisbach, the two
mid-Isar gauges that earned their place and their discharge. Weather stays hourly,
because Bright Sky has nothing finer; exp11 interpolates it and says so.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eisbach.data import LOCAL_TIMEZONE, fetch_data_from_url, localize_local_time  # noqa: E402

logger = logging.getLogger(__name__)

CACHE = REPO / "data" / "experiments"
URL = "https://www.gkd.bayern.de/de/fluesse/{kind}/isar/{slug}/messwerte/tabelle"
FIRST_YEAR = 2019

SERIES = [
    ("eisbach", "wassertemperatur", "muenchen-himmelreichbruecke-16515005"),
    ("isar_lenggries", "wassertemperatur", "lenggries-16003003"),
    ("isar_toelz", "wassertemperatur", "bad-toelz-b472-16003207"),
    ("q_lenggries", "abfluss", "lenggries-16003003"),
    ("q_toelz_kw", "abfluss", "bad-toelz-kw-16004006"),
]
COLUMN = {"wassertemperatur": "wassertemp", "abfluss": "abfluss"}
RANGE = {"wassertemperatur": (-5.0, 40.0), "abfluss": (0.0, 3000.0)}


def fetch(label: str, kind: str, slug: str, last_year: int) -> pd.Series:
    column = COLUMN[kind]
    frames = []
    for year in range(FIRST_YEAR, last_year + 1):
        url = URL.format(kind=kind, slug=slug) + f"?beginn=01.01.{year}&ende=31.12.{year}"
        for attempt in range(4):
            df = fetch_data_from_url(url, column)
            if not df.empty:
                frames.append(df)
                break
            time.sleep(2 ** attempt)
        else:
            logger.warning("%s: no data for %d", label, year)
        time.sleep(0.3)

    df = pd.concat(frames).dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["timestamp"] = localize_local_time(df["timestamp"], LOCAL_TIMEZONE)
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    s = df.set_index("timestamp")[column].sort_index()
    s = s[~s.index.duplicated(keep="first")]
    lo, hi = RANGE[kind]
    bad = s.notna() & ~s.between(lo, hi)
    if bad.any():
        logger.info("%s: masking %d readings outside %s", label, int(bad.sum()), RANGE[kind])
        s = s.mask(bad)
    # Onto a strict 15-minute grid. No aggregation: a sample that is not on the grid is
    # snapped to it, and a slot with no sample stays empty.
    return s.resample("15min").mean().rename(label)


def build(last_year: int | None = None) -> pd.DataFrame:
    last_year = last_year or pd.Timestamp.now(tz="UTC").year
    CACHE.mkdir(parents=True, exist_ok=True)
    parts = []
    for label, kind, slug in SERIES:
        s = fetch(label, kind, slug, last_year)
        parts.append(s)
        logger.info("%-16s %7d slots, %s .. %s, %.0f%% present",
                    label, len(s), s.index.min().date(), s.index.max().date(),
                    100 * s.notna().mean())
    df = pd.concat(parts, axis=1, sort=True).sort_index()
    valid = df["eisbach"].notna()
    df = df.loc[valid.idxmax(): valid[::-1].idxmax()]
    path = CACHE / "stations_15min.csv"
    df.to_csv(path)
    logger.info("Wrote %s: %d rows x %d columns", path, len(df), df.shape[1])
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()
