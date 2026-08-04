"""Fetch the inputs: measured water temperature, and the DWD weather forecast.

Water temperature is scraped from GKD Bayern, which publishes it as an HTML table in
local wall-clock time. Weather comes from Bright Sky, a free API over DWD's open data,
in UTC.
"""

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

LOCAL_TIMEZONE = "Europe/Berlin"

#: DWD station whose forecast drives the model. Munich city.
WEATHER_STATION_ID = "03379"
BRIGHTSKY_URL = "https://api.brightsky.dev/weather"

#: How far ahead the weather forecast is fetched. Must exceed the model horizon plus
#: the covariate shift, or the last forecast hours would have no weather to look at.
FORECAST_DAYS = 8

#: The model's channels, in the order it was trained on. Must not be reordered.
CHANNELS = ("wassertemp", "airtemp_96", "pressure_96")


def fetch_brightsky_data(start_date: datetime, end_date: datetime,
                         station_id: str) -> pd.DataFrame | None:
    """Fetch hourly weather from Bright Sky, past and forecast alike.

    Returns an empty frame when the range holds no data, and ``None`` when the request
    itself failed — the caller needs to tell those apart.
    """
    start_utc = start_date.astimezone(UTC) if start_date.tzinfo else start_date.replace(tzinfo=UTC)
    end_utc = end_date.astimezone(UTC) if end_date.tzinfo else end_date.replace(tzinfo=UTC)
    params = {
        "dwd_station_id": station_id,
        "date": start_utc.isoformat(timespec="seconds"),
        "last_date": end_utc.isoformat(timespec="seconds"),
    }

    logger.info("Fetching weather for %s to %s (UTC)", params["date"], params["last_date"])
    try:
        response = requests.get(BRIGHTSKY_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get("weather", [])
    except requests.exceptions.RequestException:
        logger.exception("Bright Sky request failed")
        return None

    if not data:
        logger.warning("Bright Sky returned no weather for the requested range")
        return pd.DataFrame()

    logger.info("Loaded %d hourly weather points", len(data))
    return pd.DataFrame(data)


def get_prepared_weather_data() -> pd.DataFrame:
    """Return hourly air temperature, precipitation and pressure in local time.

    Precipitation is carried through and archived even though the model does not
    currently use it as a channel.
    """
    now_local = datetime.now().astimezone()
    df_raw = fetch_brightsky_data(
        now_local - timedelta(days=HISTORY_DAYS),
        now_local + timedelta(days=FORECAST_DAYS),
        WEATHER_STATION_ID,
    )
    if df_raw is None or df_raw.empty:
        raise RuntimeError("Could not fetch weather data; refusing to forecast without it")

    weather = df_raw[["timestamp", "temperature", "precipitation", "pressure_msl"]].copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather = weather.set_index("timestamp")

    # Bright Sky delivers UTC.
    weather.index = weather.index.tz_convert(LOCAL_TIMEZONE)
    weather = weather.sort_index()
    weather = weather[~weather.index.duplicated(keep="first")]
    weather = weather.rename(columns={
        "temperature": "lufttemperatur_c",
        "precipitation": "niederschlag_mm",
        "pressure_msl": "pressure",
    })

    weather["niederschlag_mm"] = weather["niederschlag_mm"].fillna(0)
    weather["lufttemperatur_c"] = weather["lufttemperatur_c"].interpolate(method="time")
    weather["pressure"] = weather["pressure"].interpolate(method="time")

    return weather.resample("1h").agg({
        "lufttemperatur_c": "mean",
        "niederschlag_mm": "sum",
        "pressure": "mean",
    }).round(2)


def fetch_data_from_url(url: str, column_name: str) -> pd.DataFrame:
    """Scrape one measurement series out of a GKD Bayern HTML table.

    GKD serves two table layouts with different header conventions, and numbers use a
    German decimal comma. Returns an empty frame if anything about the page is not what
    we expect, rather than guessing.
    """
    logger.info("Scraping %s", column_name)
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        html_content = response.content.decode("utf-8")
    except requests.exceptions.RequestException:
        logger.exception("Could not load %s", url)
        return pd.DataFrame()

    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table", class_="tblsort") or soup.find("table", class_="datentabelle")
    if not table:
        logger.warning("No recognisable data table at %s", url)
        return pd.DataFrame()

    headers = [h.get_text(strip=True) for h in table.find("thead").find_all("th")]
    # One layout splits date and time into separate columns, the other combines them.
    df_headers = headers if any("Uhrzeit" in s for s in headers) else ["Datum/Uhrzeit"] + headers[1:]

    rows = []
    for row in table.find("tbody").find_all("tr"):
        cells = row.find_all(["td", "th"])
        rows.append({
            df_headers[i]: cell.get_text(strip=True)
            for i, cell in enumerate(cells) if i < len(df_headers)
        })

    df = pd.DataFrame(rows)
    if "Datum/Uhrzeit" in df.columns:
        combined = df["Datum/Uhrzeit"].str.replace(" Uhr", "", regex=False).str.strip()
        df["timestamp"] = pd.to_datetime(combined, format="%d.%m.%Y %H:%M", errors="coerce")
    elif "Datum" in df.columns and "Uhrzeit" in df.columns:
        time_part = df["Uhrzeit"].str.replace(" Uhr", "", regex=False).str.strip()
        df["timestamp"] = pd.to_datetime(
            df["Datum"].astype(str).str.strip() + " " + time_part,
            format="%d.%m.%Y %H:%M", errors="coerce",
        )
    else:
        logger.warning("Table at %s has no recognisable date column", url)
        return pd.DataFrame()

    df = df.dropna(subset=["timestamp"])

    # The measurement column carries its unit in the header, so match on the prefix.
    prefix = column_name.split("_")[0].lower()
    matching = [c for c in df.columns if prefix in c.lower()]
    if not matching:
        logger.warning("No column matching %r in the table at %s", column_name, url)
        return pd.DataFrame()

    result = df[["timestamp", matching[0]]].copy()
    result = result.rename(columns={matching[0]: column_name})
    result[column_name] = pd.to_numeric(
        result[column_name].astype(str).str.replace(",", "."), errors="coerce",
    )
    return result


def localize_local_time(timestamps: pd.Series, timezone_name: str = LOCAL_TIMEZONE) -> pd.Series:
    """Attach the local timezone to naive wall-clock timestamps.

    The gauge publishes local wall-clock time, which is ambiguous for one hour every
    autumn and impossible for one hour every spring.

    ``ambiguous='infer'`` resolves the autumn fold correctly *when both repeats of the
    hour are present*. They frequently are not — a single dropped sample is enough — and
    pandas then raises, killing the run. Because the input window is 40 days wide, one
    missing sample would break every run for the following six weeks.

    So: infer when we can, and when we cannot, drop the one ambiguous hour rather than
    guess at it. Losing a single hour is invisible after resampling and interpolation;
    losing six weeks of forecasts is not.

    Callers must drop the resulting NaT rows.
    """
    naive = timestamps.sort_values()
    try:
        return naive.dt.tz_localize(timezone_name, ambiguous='infer', nonexistent='shift_forward')
    except ValueError as exc:  # pandas' AmbiguousTimeError is a ValueError subclass
        logger.warning(
            "Could not infer the DST fold (%s); dropping ambiguous timestamps instead.", exc,
        )
        return naive.dt.tz_localize(timezone_name, ambiguous='NaT', nonexistent='shift_forward')


HISTORY_DAYS = 40
GAUGE_URL = (
    "https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/"
    "muenchen-himmelreichbruecke-16515005/messwerte/tabelle"
)


def prepare_data():
    """Fetch everything the pipeline needs.

    Returns ``(df_long, df_weather, df_wt)``: the model's input frame, the hourly weather
    in UTC, and the raw hourly water temperature. The latter two are returned separately
    because a replay backtest has to reassemble the input frame from a *different*
    weather forecast, and cannot do that from ``df_long`` alone — by then the covariates
    are already shifted and the unshifted values are gone.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORY_DAYS)
    url = (
        f"{GAUGE_URL}?beginn={start_date.strftime('%d.%m.%Y')}"
        f"&ende={end_date.strftime('%d.%m.%Y')}"
    )

    df_wt = fetch_data_from_url(url, "wassertemp")
    if df_wt.empty:
        raise RuntimeError(f"No water temperature data returned by {GAUGE_URL}")

    # Localize to local time, surviving both DST transitions.
    df_wt = df_wt.sort_values('timestamp').reset_index(drop=True)
    df_wt['timestamp'] = localize_local_time(df_wt['timestamp'])
    df_wt = df_wt.dropna(subset=['timestamp'])

    # Resample only after localizing, or the DST hour lands in the wrong bucket.
    df_wt = df_wt.set_index('timestamp').resample('1h').first().reset_index()

    df_weather = get_prepared_weather_data()
    df_long = assemble_long_frame(df_wt, df_weather)

    df_wt_utc = df_wt.copy()
    df_wt_utc['timestamp'] = df_wt_utc['timestamp'].dt.tz_convert('UTC')
    return df_long, df_weather.tz_convert('UTC'), df_wt_utc


#: How far ahead the weather covariates are shifted, in hours. This is what lets the
#: model use a weather forecast: at any timestamp it sees the weather this far ahead,
#: which is exactly the known-future information a forecast provides.
COVARIATE_SHIFT_HOURS = 96


def assemble_long_frame(df_wt: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """Merge water temperature with weather and shape it into the model's input frame.

    Factored out of :func:`prepare_data` so that a replay backtest can rebuild the same
    frame from a *historical* weather forecast instead of the current one, which is the
    difference between an honest backtest and an oracle one.

    ``df_wt`` has a ``timestamp`` column and a ``wassertemp`` column; ``df_weather`` is
    indexed by timestamp with ``lufttemperatur_c`` and ``pressure`` columns. Both must be
    timezone-aware.
    """
    weather = df_weather.copy()
    weather.index.name = 'timestamp'
    df_merged = pd.merge(
        df_wt,
        weather.reset_index().rename(columns={'lufttemperatur_c': 'airtemp'}),
        on='timestamp',
        how='outer',
    )

    df_merged = df_merged.set_index('timestamp')
    df_merged = df_merged[df_merged.index.notna()].sort_index()

    # Fill the water temperature only up to its last real observation. Filling beyond it
    # would invent measurements for the window where only weather covariates exist, and
    # the model would learn to trust them.
    last_wt_time = df_wt['timestamp'].max()
    df_merged.loc[:last_wt_time, 'wassertemp'] = (
        df_merged.loc[:last_wt_time, 'wassertemp'].interpolate(method='time').ffill().bfill()
    )
    df_merged['airtemp'] = df_merged['airtemp'].interpolate(method='time').ffill().bfill()
    df_merged['pressure'] = df_merged['pressure'].interpolate(method='time').ffill().bfill()

    df_merged['airtemp_96'] = df_merged['airtemp'].shift(-COVARIATE_SHIFT_HOURS)
    df_merged['pressure_96'] = df_merged['pressure'].shift(-COVARIATE_SHIFT_HOURS)
    df_merged = df_merged.drop(columns=['airtemp', 'pressure'])

    # UTC from here on, so nothing downstream has to think about local time again.
    df_merged.index = df_merged.index.tz_convert('UTC')

    df_long = pd.melt(
        df_merged.reset_index(), id_vars=['timestamp'], value_vars=list(CHANNELS),
    )
    df_long.columns = ['date', 'cols', 'data']
    df_long['cols'] = pd.Categorical(df_long['cols'], categories=list(CHANNELS), ordered=True)
    return df_long.sort_values(by=['cols', 'date'])
