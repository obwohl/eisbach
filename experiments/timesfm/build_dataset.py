"""Build the long hourly dataset the TimesFM experiments run on.

Side project: nothing here is imported by ``main.py`` or ``eisbach/``. It reuses the
production scrapers read-only and writes a cache under ``data/experiments/``, which is
gitignored.

GKD serves the gauge at 15-minute resolution back to 2010, but refuses ranges longer
than roughly a year, so the fetch is chunked per calendar year. Bright Sky is chunked
the same way.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eisbach.data import (  # noqa: E402
    LOCAL_TIMEZONE,
    WEATHER_STATION_ID,
    fetch_brightsky_data,
    fetch_data_from_url,
    localize_local_time,
)

logger = logging.getLogger(__name__)

CACHE = REPO / "data" / "experiments"

GAUGES = {
    "eisbach": "muenchen-himmelreichbruecke-16515005",
    "isar": "muenchen-16005701",
}
GAUGE_BASE = "https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/{station}/messwerte/tabelle"

FIRST_YEAR = 2010


def _fetch_gauge_year(station: str, year: int, column: str) -> pd.DataFrame:
    url = (
        f"{GAUGE_BASE.format(station=station)}"
        f"?beginn=01.01.{year}&ende=31.12.{year}"
    )
    for attempt in range(4):
        df = fetch_data_from_url(url, column)
        if not df.empty:
            return df
        time.sleep(2 ** attempt)
    logger.warning("No data for %s in %d", station, year)
    return pd.DataFrame()


def fetch_gauge(name: str, station: str, last_year: int) -> pd.Series:
    """Hourly water temperature for one gauge, indexed in UTC."""
    frames = []
    for year in range(FIRST_YEAR, last_year + 1):
        df = _fetch_gauge_year(station, year, "wassertemp")
        if not df.empty:
            frames.append(df)
            logger.info("%s %d: %d raw rows", name, year, len(df))
    if not frames:
        raise RuntimeError(f"No data at all for {name}")

    df = pd.concat(frames).dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    # The gauge publishes local wall-clock time; reuse the production DST handling.
    df["timestamp"] = localize_local_time(df["timestamp"], LOCAL_TIMEZONE)
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    series = df.set_index("timestamp")["wassertemp"].sort_index()
    series = series[~series.index.duplicated(keep="first")]
    # 15-minute samples -> hourly means. No interpolation here: a gap must stay a gap,
    # so the quality report and the model both see it.
    return series.resample("1h").mean().rename(name)


def fetch_weather(last_year: int) -> pd.DataFrame:
    """Hourly observed weather from Bright Sky, indexed in UTC."""
    frames = []
    for year in range(FIRST_YEAR, last_year + 1):
        start = datetime(year, 1, 1, tzinfo=UTC)
        end = min(datetime(year, 12, 31, 23, tzinfo=UTC), datetime.now(UTC))
        if start > end:
            continue
        raw = fetch_brightsky_data(start, end, WEATHER_STATION_ID)
        if raw is None or raw.empty:
            logger.warning("No weather for %d", year)
            continue
        frames.append(raw)
        logger.info("weather %d: %d rows", year, len(raw))

    df = pd.concat(frames)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    keep = [c for c in ("temperature", "pressure_msl", "sunshine", "cloud_cover",
                        "relative_humidity", "wind_speed", "dew_point", "precipitation")
            if c in df.columns]
    out = df[keep].rename(columns={"temperature": "airtemp", "pressure_msl": "pressure"})
    return out.resample("1h").mean()


def build(last_year: int | None = None) -> pd.DataFrame:
    last_year = last_year or datetime.now(UTC).year
    CACHE.mkdir(parents=True, exist_ok=True)

    parts = [fetch_gauge(name, station, last_year) for name, station in GAUGES.items()]
    weather = fetch_weather(last_year)

    df = pd.concat(parts + [weather], axis=1, sort=True).sort_index()
    # Trim to the span where the Eisbach itself has data; leading weather-only rows
    # would look like a gauge outage.
    valid = df["eisbach"].notna()
    df = df.loc[valid.idxmax(): valid[::-1].idxmax()]

    path = CACHE / "long_hourly.csv"
    df.to_csv(path)
    logger.info("Wrote %s: %d rows, %s .. %s", path, len(df), df.index.min(), df.index.max())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build()
