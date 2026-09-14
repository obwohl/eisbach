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

#: Bright Sky fields carried through and archived but deliberately *not* model channels,
#: mapped to how each is aggregated onto the hourly grid.
#:
#: A covariate that was never archived cannot be trained on later: Bright Sky serves
#: observations indefinitely, but a forecast is gone the moment it is superseded, and the
#: pre-August payload carried all of these before the normalisation dropped them. Solar
#: radiation in particular is the obvious physical driver of a shallow urban channel that
#: air temperature alone cannot explain, whereas pressure — the only covariate the
#: model's channel mask lets water attend to — moves the forecast by at most 0.018 °C
#: (see docs/model.md). So capture them now, and decide at the next training run.
#:
#: Adding one here archives it. Making it a *channel* would mean touching ``CHANNELS``,
#: which is the trained order and is load-bearing.
ARCHIVED_WEATHER_FIELDS = {
    "sunshine": "sum",           # minutes of sunshine within the hour
    "solar": "sum",              # kWh/m² within the hour
    "cloud_cover": "mean",       # percent
    "relative_humidity": "mean", # percent
    "wind_speed": "mean",        # km/h
    "dew_point": "mean",         # °C
}


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
        payload = response.json()
        data = payload.get("weather", [])
    except requests.exceptions.RequestException:
        logger.exception("Bright Sky request failed")
        return None

    if not data:
        logger.warning("Bright Sky returned no weather for the requested range")
        return pd.DataFrame()

    sources = {source["id"]: source for source in payload.get("sources", [])}
    for row in data:
        source = sources.get(row.get("source_id"))
        if source is None or str(source.get("dwd_station_id")).zfill(5) != station_id:
            raise ValueError(f"Bright Sky returned an unverified station; expected {station_id}")

    logger.info("Loaded %d hourly weather points", len(data))
    return pd.DataFrame(data)


def get_prepared_weather_data(*, start_date=None, end_date=None) -> pd.DataFrame:
    """Return the hourly DWD weather in local time.

    Air temperature and pressure are the model's covariates. Precipitation and everything
    in :data:`ARCHIVED_WEATHER_FIELDS` are carried through so they reach the archive; the
    model never sees them, and adding a column here cannot change its input frame, which
    is built from :data:`CHANNELS` alone.

    A field Bright Sky did not return for this range is simply absent rather than fatal —
    the forecast must not fail over a covariate nothing reads yet.
    """
    now_local = datetime.now().astimezone()
    df_raw = fetch_brightsky_data(
        start_date or now_local - timedelta(days=HISTORY_DAYS),
        end_date or now_local + timedelta(days=FORECAST_DAYS),
        WEATHER_STATION_ID,
    )
    if df_raw is None or df_raw.empty:
        raise RuntimeError("Could not fetch weather data; refusing to forecast without it")

    archived = [field for field in ARCHIVED_WEATHER_FIELDS if field in df_raw.columns]
    missing = [field for field in ARCHIVED_WEATHER_FIELDS if field not in df_raw.columns]
    if missing:
        logger.warning("Bright Sky returned no %s; archiving without them", ", ".join(missing))

    weather = df_raw[
        ["timestamp", "temperature", "precipitation", "pressure_msl", *archived]
    ].copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather = weather.set_index("timestamp")
    for field in archived:
        # An all-null field arrives as an object column, which would break the aggregation.
        weather[field] = pd.to_numeric(weather[field], errors="coerce")

    # Bright Sky delivers UTC.
    weather.index = weather.index.tz_convert(LOCAL_TIMEZONE)
    weather = weather.sort_index()
    weather = weather[~weather.index.duplicated(keep="first")]
    weather = weather.rename(columns={
        "temperature": "lufttemperatur_c",
        "precipitation": "niederschlag_mm",
        "pressure_msl": "pressure",
    })

    aggregation = {
        "lufttemperatur_c": "mean",
        "niederschlag_mm": lambda values: values.sum(min_count=1),
        "pressure": "mean",
    }
    for field in archived:
        # A sum over an hour DWD reported nothing for must stay missing: "no sunshine
        # value" is not "no sunshine", and this frame is archived as evidence.
        aggregation[field] = (
            (lambda values: values.sum(min_count=1))
            if ARCHIVED_WEATHER_FIELDS[field] == "sum"
            else ARCHIVED_WEATHER_FIELDS[field]
        )

    return weather.resample("1h").agg(aggregation).round(2)


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


class ImplausibleGaugeData(RuntimeError):
    """Raised when the gauge series is broken rather than merely spiky."""


#: Water temperatures outside this are an instrument fault, not weather. Sixteen years of
#: this gauge span 0.3 to 24.1 °C, so these bounds cannot reject a real reading; they
#: exist to catch a *sustained* fault, which the neighbourhood test below cannot see
#: because a sustained fault corrupts the neighbours too.
PLAUSIBLE_RANGE_C = (-5.0, 40.0)

#: How many hours either side form the neighbourhood a reading is judged against. The
#: statistic is their median, so with three on each side it survives two bad neighbours —
#: a spike up to three hours wide is still caught.
SPIKE_NEIGHBOURHOOD_HOURS = 3

#: How many robust standard deviations of the neighbourhood residual count as a spike.
SPIKE_SIGMAS = 12.0

#: The floor under that threshold, and it is load-bearing rather than defensive. The
#: gauge reports in steps of 0.1 °C and the river can sit still for days, so the residual
#: MAD of a calm winter window is exactly zero — in 11 % of 40-day windows across the
#: sixteen-year record. Without a floor the threshold would collapse to zero there and
#: reject most of the window.
#:
#: Calibrated against the whole record: across 124 303 hourly readings the largest
#: residual that was real is 1.33 °C, and the one instrument fault is 46.7 °C. At 2.0 °C
#: the gate sits 1.5x above anything genuine ever seen and 23x below the fault, and
#: flags exactly one reading in sixteen years.
SPIKE_FLOOR_C = 2.0

#: Above this share of rejected readings the gauge is not spiking, it is broken, and
#: interpolating over it would be inventing the input rather than repairing it.
MAX_REJECTED_FRACTION = 0.01

GAUGE_URL = (
    "https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/"
    "muenchen-himmelreichbruecke-16515005/messwerte/tabelle"
)


def _spikes(values: pd.Series, measured: pd.Series,
            *, exclude: pd.Series) -> tuple[pd.Series, float]:
    """Readings that disagree with the hours around them, and the threshold used.

    ``exclude`` names readings already known to be bad; they are kept out of every
    neighbourhood so they cannot condemn their neighbours, but they are still judged.
    """
    clean = values.mask(exclude)
    half = SPIKE_NEIGHBOURHOOD_HOURS
    neighbours = pd.concat(
        [clean.shift(k) for k in range(-half, half + 1) if k != 0], axis=1,
    ).median(axis=1)
    residual = values - neighbours
    mad = (residual - residual.median()).abs().median()
    threshold = max(SPIKE_SIGMAS * 1.4826 * float(mad), SPIKE_FLOOR_C)
    return measured & (residual.abs() > threshold), threshold


def reject_implausible_readings(df_wt: pd.DataFrame) -> pd.DataFrame:
    """Blank out gauge readings that no river produced, in place of the frame given.

    On 2026-09-09 the gauge reported 154.4 °C for one hour. Nothing stopped it: the
    plausibility gate in :mod:`eisbach.validate` judges the *forecast*, so the reading
    reached the model's input window, the observation archive, and from there 52 rows of
    the append-only verification store, where pooled live RMSE reads 2.903 with it and
    0.863 without.

    Two independent tests, because each covers the other's blind spot:

    * an absolute range, which catches a fault that lasts long enough to corrupt its own
      neighbourhood;
    * a **neighbourhood** test — the reading against the median of the hours around it,
      scaled by the robust spread of that same residual over the whole window. A river's
      temperature is smooth at hourly resolution, so this is sensitive to a single wrong
      hour without being fooled by the daily cycle, which moves the neighbourhood along
      with the reading.

    A rejected reading becomes ``NaN``, which is the whole repair: ``assemble_long_frame``
    already interpolates ``wassertemp`` onto the hourly grid for the model, and
    ``inference._observed_frame`` already drops NaN before archiving. So the model sees a
    repaired series and the archive records the hour as *never measured* rather than as an
    invented value — which is what it is, and which keeps the verification store's promise
    that only real readings are scored.

    Raises :class:`ImplausibleGaugeData` when too much of the window fails. At that point
    the series is not a good signal with a spike in it, and a run that interpolated over
    it would publish a forecast built mostly from guesses.
    """
    df = df_wt.copy()
    values = pd.to_numeric(df["wassertemp"], errors="coerce")
    measured = values.notna()
    if not measured.any():
        return df

    out_of_range = measured & ~values.between(*PLAUSIBLE_RANGE_C)

    # Two passes, because one is not enough. A spike several hours wide fills half of its
    # own neighbours' neighbourhoods, which drags their median far enough to condemn
    # perfectly good readings on either side of it — a three-hour fault takes five hours
    # down with it. The second pass rebuilds each neighbourhood without whatever the
    # first pass rejected, so the survivors are judged against real values only.
    spike, threshold = _spikes(values, measured, exclude=out_of_range)
    spike, threshold = _spikes(values, measured, exclude=out_of_range | spike)

    rejected = out_of_range | spike
    if not rejected.any():
        return df

    share = rejected.sum() / measured.sum()
    if share > MAX_REJECTED_FRACTION:
        raise ImplausibleGaugeData(
            f"{rejected.sum()} of {measured.sum()} gauge readings are implausible "
            f"({share:.1%} > {MAX_REJECTED_FRACTION:.1%}); the gauge is broken, not spiky"
        )

    for ts, value in zip(df.loc[rejected, "timestamp"], values[rejected], strict=True):
        logger.warning(
            "Rejecting implausible gauge reading %.1f °C at %s (threshold %.2f °C)",
            value, ts, threshold,
        )
    df.loc[rejected, "wassertemp"] = float("nan")
    return df


def prepare_data():
    """Legacy direct-fetch helper; production now uses covariates.prepare_live.

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

    # Before anything reads it: the model's input window, the observation archive and
    # every backtest replay all come off this one frame.
    df_wt = reject_implausible_readings(df_wt)

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
    timezone-aware. Any further weather column is carried along and then dropped by the
    melt, which takes :data:`CHANNELS` and nothing else — so archiving more weather can
    never change what the model is fed.
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

    from eisbach.covariates import fill_short_gaps

    # A whole one-hour internal gap may be interpolated for the model only. Longer
    # gaps, edges and drained-river episodes remain missing; the model refuses them.
    df_merged = df_merged.groupby(level=0).first().resample("1h").asfreq()
    last_wt_time = df_wt.loc[df_wt["wassertemp"].notna(), "timestamp"].max()
    df_merged.loc[:last_wt_time, "wassertemp"] = fill_short_gaps(
        df_merged.loc[:last_wt_time, "wassertemp"], "eisbach",
    )
    df_merged["airtemp"] = fill_short_gaps(df_merged["airtemp"], "airtemp")
    df_merged["pressure"] = fill_short_gaps(df_merged["pressure"], "pressure")

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
