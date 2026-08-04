import pandas as pd
from datetime import datetime, timedelta, timezone
import requests
from bs4 import BeautifulSoup
import logging

def fetch_brightsky_data(start_date: datetime, end_date: datetime, station_id: str) -> pd.DataFrame | None:
    TARGET_TIMEZONE = 'Europe/Berlin'
    start_utc = start_date.astimezone(timezone.utc) if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
    end_utc = end_date.astimezone(timezone.utc) if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)
    start_str = start_utc.isoformat(timespec='seconds')
    end_str = end_utc.isoformat(timespec='seconds')
    params = {'dwd_station_id': station_id, 'date': start_str, 'last_date': end_str}

    logging.info(f"Lade Wetterdaten von Bright Sky für den Zeitraum (in UTC): {start_str} bis {end_str}...")
    try:
        response = requests.get("https://api.brightsky.dev/weather", params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get('weather', [])
        if not data:
            logging.warning("Keine Wetterdaten für den angefragten Zeitraum gefunden.")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        logging.info(f"Erfolgreich {len(df)} stündliche Wetter-Datenpunkte geladen.")
        return df
    except requests.exceptions.RequestException as e:
        logging.exception(f"Netzwerk- oder API-Fehler beim Abrufen der Wetterdaten: {e}")
        return None

def get_prepared_weather_data():
    TARGET_TIMEZONE = 'Europe/Berlin'
    TAGE_VERGANGENHEIT = 40
    TAGE_ZUKUNFT = 8
    now_local = datetime.now().astimezone()
    start_date = now_local - timedelta(days=TAGE_VERGANGENHEIT)
    end_date = now_local + timedelta(days=TAGE_ZUKUNFT)

    df_raw = fetch_brightsky_data(start_date, end_date, "03379")
    if df_raw is None or df_raw.empty:
        logging.error("Download der Wetterdaten fehlgeschlagen. Überspringe Wetter-Integration.")
        return pd.DataFrame()

    logging.info(f"Verarbeite Wetterdaten und konvertiere zu Zeitzone '{TARGET_TIMEZONE}'...")
    wetter_df = df_raw[['timestamp', 'temperature', 'precipitation', 'pressure_msl']].copy()
    wetter_df['timestamp'] = pd.to_datetime(wetter_df['timestamp'])
    wetter_df.set_index('timestamp', inplace=True)

    # Bright Sky liefert UTC, hier konvertieren wir in die Zielzone
    wetter_df.index = wetter_df.index.tz_convert(TARGET_TIMEZONE)
    wetter_df.sort_index(inplace=True)
    wetter_df = wetter_df[~wetter_df.index.duplicated(keep='first')]
    wetter_df.rename(columns={'temperature': 'lufttemperatur_c', 'precipitation': 'niederschlag_mm', 'pressure_msl': 'pressure'}, inplace=True)

    wetter_df['niederschlag_mm'] = wetter_df['niederschlag_mm'].fillna(0)
    wetter_df['lufttemperatur_c'] = wetter_df['lufttemperatur_c'].interpolate(method='time')
    wetter_df['pressure'] = wetter_df['pressure'].interpolate(method='time')

    logging.info("Resample Wetterdaten auf 1-Stunden-Intervall...")
    wetter_1h = wetter_df.resample('1h').agg({
        'lufttemperatur_c': 'mean',
        'niederschlag_mm': 'sum',
        'pressure': 'mean'
    }).round(2)
    return wetter_1h

def fetch_data_from_url(url, column_name):
    logging.info(f"Processing URL for: {column_name}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        html_content = response.content.decode('utf-8')
    except Exception as e:
        logging.exception(f"Fehler beim Laden der URL: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find("table", class_="tblsort") or soup.find("table", class_="datentabelle")
    if not table: return pd.DataFrame()

    headers = [h.get_text(strip=True) for h in table.find('thead').find_all("th")]
    df_headers = headers if any('Uhrzeit' in s for s in headers) else ['Datum/Uhrzeit'] + headers[1:]
    rows = table.find('tbody').find_all("tr")
    data = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        data.append({df_headers[i]: cell.get_text(strip=True) for i, cell in enumerate(cells) if i < len(df_headers)})

    df = pd.DataFrame(data)
    if 'Datum/Uhrzeit' in df.columns:
        df['Datum/Uhrzeit'] = df['Datum/Uhrzeit'].str.replace(' Uhr', '', regex=False).str.strip()
        df['timestamp'] = pd.to_datetime(df['Datum/Uhrzeit'], format='%d.%m.%Y %H:%M', errors='coerce')
    elif 'Datum' in df.columns and 'Uhrzeit' in df.columns:
        df['Uhrzeit'] = df['Uhrzeit'].str.replace(' Uhr', '', regex=False).str.strip()
        df['timestamp'] = pd.to_datetime(df['Datum'].astype(str).str.strip() + ' ' + df['Uhrzeit'], format="%d.%m.%Y %H:%M", errors='coerce')

    df.dropna(subset=['timestamp'], inplace=True)

    # Spalte für Messwert finden
    matching_cols = [c for c in df.columns if column_name.split('_')[0].lower() in c.lower()]
    if not matching_cols:
        logging.warning(f"No matching column found for {column_name}")
        return pd.DataFrame()
    target_col = matching_cols[0]

    df_final = df[["timestamp", target_col]].copy()
    df_final.rename(columns={target_col: column_name}, inplace=True)
    df_final[column_name] = pd.to_numeric(df_final[column_name].astype(str).str.replace(",", "."), errors='coerce')

    return df_final


def localize_local_time(timestamps: pd.Series, timezone_name: str = 'Europe/Berlin') -> pd.Series:
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
        logging.warning(
            "Could not infer the DST fold (%s); dropping ambiguous timestamps instead.", exc,
        )
        return naive.dt.tz_localize(timezone_name, ambiguous='NaT', nonexistent='shift_forward')


def prepare_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=40)
    wassertemperatur_url = f"https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/muenchen-himmelreichbruecke-16515005/messwerte/tabelle?beginn={start_date.strftime('%d.%m.%Y')}&ende={end_date.strftime('%d.%m.%Y')}"

    # 1. Daten laden
    df_wt = fetch_data_from_url(wassertemperatur_url, "wassertemp")

    if df_wt.empty:
        logging.error("Water temperature data is empty. Aborting preparation.")
        return pd.DataFrame(), pd.DataFrame()

    # 2. Localize to local time, surviving both DST transitions.
    df_wt = df_wt.sort_values('timestamp').reset_index(drop=True)
    df_wt['timestamp'] = localize_local_time(df_wt['timestamp'])
    df_wt = df_wt.dropna(subset=['timestamp'])

    # 3. Resampling erst NACH der Lokalisierung
    df_wt = df_wt.set_index('timestamp').resample('1h').first().reset_index()

    # 4. Wetterdaten holen
    df_wetter = get_prepared_weather_data()

    # 5. Mergen (beide sind nun Europe/Berlin aware)
    df_merged = pd.merge(
        df_wt,
        df_wetter.reset_index().rename(columns={'lufttemperatur_c': 'airtemp'}),
        on='timestamp',
        how='outer'
    )

    df_merged.set_index('timestamp', inplace=True)
    df_merged = df_merged[df_merged.index.notna()].sort_index()

    # We must only interpolate/fill the water temperature up to its actual last available timestamp.
    # Otherwise, we leak data into the future where only weather covariates exist.
    last_wt_time = df_wt['timestamp'].max()
    df_merged.loc[:last_wt_time, 'wassertemp'] = df_merged.loc[:last_wt_time, 'wassertemp'].interpolate(method='time').ffill().bfill()
    df_merged['airtemp'] = df_merged['airtemp'].interpolate(method='time').ffill().bfill()
    df_merged['pressure'] = df_merged['pressure'].interpolate(method='time').ffill().bfill()

    # 6. Feature Shifting (Vorschau-Werte)
    df_merged['airtemp_96'] = df_merged['airtemp'].shift(-96)
    df_merged['pressure_96'] = df_merged['pressure'].shift(-96)
    df_merged.drop(columns=['airtemp', 'pressure'], inplace=True)

    # 7. Finalisierung: Zurück zu UTC für konsistente Speicherung/Verarbeitung
    df_merged.index = df_merged.index.tz_convert('UTC')

    # 8. Melting für Output
    df_long = pd.melt(df_merged.reset_index(), id_vars=['timestamp'], value_vars=['wassertemp', 'airtemp_96', 'pressure_96'])
    df_long.columns = ['date', 'cols', 'data']
    df_long['cols'] = pd.Categorical(df_long['cols'], categories=['wassertemp', 'airtemp_96', 'pressure_96'], ordered=True)
    df_long = df_long.sort_values(by=['cols', 'date'])

    return df_long, df_wetter.tz_convert('UTC')
