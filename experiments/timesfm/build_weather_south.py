"""Weather along the catchment, not just at the gauge.

Munich's own air temperature says what the last kilometre of the Eisbach is doing. It
says much less about the water arriving from the Isarwinkel and the Loisach valley,
which is where the river is actually made. Rain is the starker case: a thunderstorm over
Munich adds almost nothing to the Isar, while the same storm over Lenggries arrives as
real volume at a different temperature.

Five locations up the catchment plus Munich itself: air temperature, precipitation and
global radiation — the last being the energy that actually heats the water, where air
temperature is only its proxy.
Bright Sky stitches observations and the DWD forecast at a coordinate, so every one of
these is available **as past and as future** — a genuine known-future covariate, not an
oracle, once the pipeline asks for it at run time.

Writes ``data/experiments/weather_south.csv``, gitignored.
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

#: Ordered up the catchment. Munich is the gauge's own weather and is already a covariate
#: in the production model; the rest are new.
LOCATIONS = {
    "muenchen": (48.163, 11.543),
    "holzkirchen": (47.877, 11.700),
    "toelz": (47.761, 11.556),
    "lenggries": (47.683, 11.567),
    "kochel": (47.658, 11.366),
    "garmisch": (47.483, 11.062),
}

FIELDS = ("temperature", "precipitation", "solar")


def fetch_location(name: str, lat: float, lon: float, last_year: int) -> pd.DataFrame:
    frames = []
    for year in range(FIRST_YEAR, last_year + 1):
        start = datetime(year, 1, 1, tzinfo=UTC)
        end = min(datetime(year, 12, 31, 23, tzinfo=UTC), datetime.now(UTC))
        if start > end:
            continue
        for attempt in range(4):
            try:
                r = requests.get(BRIGHTSKY_URL, params={
                    "lat": lat, "lon": lon,
                    "date": start.isoformat(timespec="seconds"),
                    "last_date": end.isoformat(timespec="seconds")}, timeout=120)
                r.raise_for_status()
                rows = r.json().get("weather", [])
                break
            except requests.exceptions.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        if not rows:
            logger.warning("%s: nothing for %d", name, year)
            continue
        frames.append(pd.DataFrame(rows))
        time.sleep(0.2)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    out = df[[f for f in FIELDS if f in df.columns]].resample("1h").mean()
    return out.rename(columns={"temperature": f"t_{name}", "precipitation": f"rain_{name}",
                              "solar": f"solar_{name}"})


def build(last_year: int | None = None) -> pd.DataFrame:
    last_year = last_year or datetime.now(UTC).year
    CACHE.mkdir(parents=True, exist_ok=True)
    parts = []
    for name, (lat, lon) in LOCATIONS.items():
        part = fetch_location(name, lat, lon, last_year)
        parts.append(part)
        logger.info("%-13s %6d hours, %s .. %s, %s",
                    name, len(part), part.index.min().date(), part.index.max().date(),
                    ", ".join(f"{c} {part[c].notna().mean():.0%}" for c in part.columns))
    df = pd.concat(parts, axis=1, sort=True).sort_index()
    path = CACHE / "weather_south.csv"
    df.to_csv(path)
    logger.info("Wrote %s: %d rows x %d columns", path, len(df), df.shape[1])
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()
