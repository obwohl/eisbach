"""Fetch the upstream river network the Eisbach hangs off.

The Eisbach branches off the Isar at the Oberföhring weir, so everything that happens to
the Isar upstream of Munich arrives at the Eisbach a few hours later. That travel time is
the one thing a purely local model cannot know, and it is exactly the horizon where a
96-hour forecast is judged hardest at its short end.

Stations are the GKD gauges on the Isar's mainstem above Munich, the two tributaries that
join above it (Rißbach, Loisach), the Eisbach's sibling arm through the Englischer Garten,
and discharge wherever it exists — because 50 m³/s at 11 °C and 200 m³/s at 11 °C arrive
as very different things.

Chunked per calendar year: GKD returns an empty table for longer ranges. Writes
``data/experiments/stations_hourly.csv``, which is gitignored.
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

#: ``(label, kind, slug)``, ordered downstream: Mittenwald is ~100 river km above Munich,
#: the Eisbach is the target. Labels prefixed ``q_`` are discharge in m³/s; the rest are
#: water temperature in °C.
STATIONS = [
    ("isar_mittenwald", "wassertemperatur", "mittenwald-16000708"),
    ("rissbach_klamm", "wassertemperatur", "rissbachklamm-16145008"),
    ("isar_lenggries", "wassertemperatur", "lenggries-16003003"),
    ("isar_toelz", "wassertemperatur", "bad-toelz-b472-16003207"),
    ("isar_puppling", "wassertemperatur", "puppling-16004403"),
    ("loisach_eschenlohe", "wassertemperatur", "eschenlohe-bruecke-16404106"),
    ("loisach_beuerberg", "wassertemperatur", "beuerberg-16408504"),
    ("isar_muenchen", "wassertemperatur", "muenchen-16005701"),
    ("schwabinger_bach", "wassertemperatur", "muenchen-tieraerztl-hochschule-16516008"),
    ("eisbach", "wassertemperatur", "muenchen-himmelreichbruecke-16515005"),
    ("q_mittenwald", "abfluss", "mittenwald-16000708"),
    ("q_rissbachdueker", "abfluss", "rissbachdueker-16001303"),
    ("q_rissbachklamm", "abfluss", "rissbachklamm-16145008"),
    ("q_sylvenstein", "abfluss", "sylvenstein-16002500"),
    ("q_lenggries", "abfluss", "lenggries-16003003"),
    ("q_toelz_kw", "abfluss", "bad-toelz-kw-16004006"),
    ("q_puppling", "abfluss", "puppling-16004403"),
    ("q_loisach_kochel", "abfluss", "kochel-16407002"),
    ("q_loisach_beuerberg", "abfluss", "beuerberg-16408504"),
    ("q_isar_muenchen", "abfluss", "muenchen-16005701"),
    ("q_eisbach", "abfluss", "muenchen-himmelreichbruecke-16515005"),
    ("q_schwabinger_bach", "abfluss", "muenchen-tieraerztl-hochschule-16516008"),
]

COLUMN = {"wassertemperatur": "wassertemp", "abfluss": "abfluss"}

#: Outside these a reading is an instrument fault rather than a river. Discharge is
#: bounded below by zero and above by well over any Isar flood; temperature by the same
#: bounds the production gate uses.
RANGE = {"wassertemperatur": (-5.0, 40.0), "abfluss": (0.0, 3000.0)}


def fetch_series(label: str, kind: str, slug: str, last_year: int) -> pd.Series:
    """One station as an hourly series in UTC, gaps left as gaps."""
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

    if not frames:
        raise RuntimeError(f"No data at all for {label}")

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
    return s.resample("1h").mean().rename(label)


def build(last_year: int | None = None) -> pd.DataFrame:
    last_year = last_year or pd.Timestamp.now(tz="UTC").year
    CACHE.mkdir(parents=True, exist_ok=True)

    series = []
    for label, kind, slug in STATIONS:
        s = fetch_series(label, kind, slug, last_year)
        series.append(s)
        logger.info("%-22s %6d hours, %s .. %s, %.0f%% present",
                    label, len(s), s.index.min().date(), s.index.max().date(),
                    100 * s.notna().mean())

    df = pd.concat(series, axis=1, sort=True).sort_index()
    valid = df["eisbach"].notna()
    df = df.loc[valid.idxmax(): valid[::-1].idxmax()]
    path = CACHE / "stations_hourly.csv"
    df.to_csv(path)
    logger.info("Wrote %s: %d rows x %d columns", path, len(df), df.shape[1])
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()
