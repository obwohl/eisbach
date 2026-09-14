"""Global radiation from one named station, not from a coordinate.

``build_weather_south.py`` asks Bright Sky for weather *at a coordinate*, and Bright Sky
answers from whichever station happens to be nearest and reporting. For air temperature
that is harmless. For radiation it is not: the DWD measures global radiation at far fewer
sites than it measures temperature, so six different coordinates across the Isarwinkel and
the Loisach valley all resolved to the *same* station — and to a different same station
before and after September 2020, when Kreuth came online and displaced Garmisch as the
nearest. What looked like six series was one series with a source change buried in the
middle of it, which is exactly the kind of seam a covariate must not have.

So this asks for a station by its DWD id and gets that station or nothing.

Two candidates, both measured over 2019-2026 before choosing:

``02290`` **Hohenpeißenberg**, 977 m — the DWD's mountain observatory, radiation measured
without a gap in 93 of 93 months. It sits on the watershed between the Ammer and the
Loisach, so its own runoff reaches the Isar only at Moosburg, *below* Munich: it is not in
the catchment. What it has is an unobstructed horizon and an unbroken record.

``01550`` **Garmisch-Partenkirchen**, 719 m — in the Loisach valley, which does reach the
Isar above Munich, both as the Loisach itself and as the upper Isar and Rißbach water the
Krün diversion sends through Walchensee and Kochelsee. 97.3 % coverage, four thin months.

``02738`` Kreuth is deliberately absent. It is the nearest radiation station to Lenggries
and it is the one the coordinate lookup reaches for after 2020 — and it carries data in
only 59 % of hours, 37 of its 73 months below nine tenths. Nearness is not reliability.

Which of the two earns its place is `exp13_solar.py`'s question, not this file's.

Writes ``data/experiments/solar_stations.csv``, gitignored.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eisbach.data import BRIGHTSKY_URL  # noqa: E402

logger = logging.getLogger(__name__)

CACHE = REPO / "data" / "experiments"
FIRST_YEAR = 2019

#: DWD station id -> the column suffix it gets. Named, not nearest.
STATIONS = {
    "02290": "hohenpeissenberg",
    "01550": "garmisch",
}


def fetch_station(station_id: str, name: str, last_year: int) -> pd.DataFrame:
    frames = []
    for year in range(FIRST_YEAR, last_year + 1):
        start = datetime(year, 1, 1, tzinfo=UTC)
        end = min(datetime(year, 12, 31, 23, tzinfo=UTC), datetime.now(UTC))
        if start > end:
            continue
        for attempt in range(4):
            try:
                r = requests.get(BRIGHTSKY_URL, params={
                    "dwd_station_id": station_id,
                    "date": start.isoformat(timespec="seconds"),
                    "last_date": end.isoformat(timespec="seconds")}, timeout=180)
                r.raise_for_status()
                rows = r.json().get("weather", [])
                break
            except requests.exceptions.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        if not rows:
            logger.warning("%s (%s): nothing for %d", name, station_id, year)
            continue
        frames.append(pd.DataFrame(rows))
        time.sleep(0.2)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    if "solar" not in df.columns:
        raise RuntimeError(f"{name} ({station_id}) returned no solar field")
    # Hourly means, so a station reporting every ten minutes does not outvote an hourly one.
    return df[["solar"]].resample("1h").mean().rename(columns={"solar": f"solar_{name}"})


def build(last_year: int | None = None) -> pd.DataFrame:
    last_year = last_year or datetime.now(UTC).year
    CACHE.mkdir(parents=True, exist_ok=True)
    parts = []
    for station_id, name in STATIONS.items():
        part = fetch_station(station_id, name, last_year)
        col = part.columns[0]
        monthly = part[col].notna().resample("MS").mean()
        logger.info("%-18s (%s) %6d hours, %s .. %s, %.1f%% present, %d/%d months below 90%%",
                    name, station_id, len(part), part.index.min().date(),
                    part.index.max().date(), 100 * part[col].notna().mean(),
                    int((monthly < 0.9).sum()), len(monthly))
        parts.append(part)
    df = pd.concat(parts, axis=1, sort=True).sort_index()
    path = CACHE / "solar_stations.csv"
    df.to_csv(path)
    logger.info("Wrote %s: %d rows x %d columns", path, len(df), df.shape[1])
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()
