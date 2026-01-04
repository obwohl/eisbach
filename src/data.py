import pandas as pd
from datetime import datetime, timedelta, timezone
import requests
from bs4 import BeautifulSoup
import time

TARGET_TIMEZONE = 'Europe/Berlin'
TAGE_VERGANGENHEIT = 40
TAGE_ZUKUNFT = 8

def fetch_brightsky_data(start_date: datetime, end_date: datetime, station_id: str) -> pd.DataFrame | None:
    start_utc = start_date.astimezone(timezone.utc) if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
    end_utc = end_date.astimezone(timezone.utc) if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)
    start_str = start_utc.isoformat(timespec='seconds')
    end_str = end_utc.isoformat(timespec='seconds')
    params = {'dwd_station_id': station_id, 'date': start_str, 'last_date': end_str}

    print(f"Lade Wetterdaten von Bright Sky für den Zeitraum (in UTC): {start_str} bis {end_str}...")
    try:
        response = requests.get("https://api.brightsky.dev/weather", params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get('weather', [])
        if not data:
            print("Keine Wetterdaten für den angefragten Zeitraum gefunden.")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        print(f"Erfolgreich {len(df)} stündliche Wetter-Datenpunkte geladen.")
        return df
    except requests.exceptions.RequestException as e:
        print(f"Netzwerk- oder API-Fehler beim Abrufen der Wetterdaten: {e}")
        return None

def get_prepared_weather_data() -> pd.DataFrame:
    now_local = datetime.now().astimezone()
    start_date = now_local - timedelta(days=TAGE_VERGANGENHEIT)
    end_date = now_local + timedelta(days=TAGE_ZUKUNFT)

    df_raw = fetch_brightsky_data(start_date, end_date, "03379")
    if df_raw is None or df_raw.empty:
        print("Download der Wetterdaten fehlgeschlagen. Überspringe Wetter-Integration.")
        return pd.DataFrame()

    print(f"\nVerarbeite Wetterdaten und konvertiere zu Zeitzone '{TARGET_TIMEZONE}'...")
    wetter_df = df_raw[['timestamp', 'temperature', 'precipitation', 'pressure_msl']].copy()
    wetter_df['timestamp'] = pd.to_datetime(wetter_df['timestamp'])
    wetter_df.set_index('timestamp', inplace=True)
    wetter_df.index = wetter_df.index.tz_convert(TARGET_TIMEZONE)
    wetter_df.sort_index(inplace=True)
    wetter_df = wetter_df[~wetter_df.index.duplicated(keep='first')]
    wetter_df.rename(columns={'temperature': 'lufttemperatur_c', 'precipitation': 'niederschlag_mm', 'pressure_msl': 'pressure'}, inplace=True)
    wetter_df['niederschlag_mm'] = wetter_df['niederschlag_mm'].fillna(0)
    wetter_df['lufttemperatur_c'] = wetter_df['lufttemperatur_c'].interpolate(method='time')
    wetter_df['pressure'] = wetter_df['pressure'].interpolate(method='time')

    print("Resample Wetterdaten auf 1-Stunden-Intervall...")
    wetter_1h = wetter_df.resample('1h').agg({
        'lufttemperatur_c': 'mean',
        'niederschlag_mm': 'sum',
        'pressure': 'mean'
    }).round(2)
    return wetter_1h

def fetch_data_from_url(url, column_name) -> pd.DataFrame:
    print(f"-> Processing URL for: {column_name}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        html_content = response.content.decode('utf-8')
    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] Could not fetch URL {url}: {e}")
        return pd.DataFrame()
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find("table", class_="tblsort")
    if not table:
        table = soup.find("table", class_="datentabelle")
        if not table:
            print(f"   [WARN] No table with class 'tblsort' or 'datentabelle' found on URL: {url}")
            return pd.DataFrame()

    # Header Parsing
    try:
        headers = [header.get_text(strip=True) for header in table.find('thead').find_all("th")]
    except AttributeError:
        print(f"   [WARN] Could not find a 'thead' section in the table for URL: {url}")
        return pd.DataFrame()

    # Sometimes headers in 'Aktuelle Werte' are 'Datum', 'Uhrzeit', 'Wert'
    # Sometimes merged.
    # The snippet showed 'Datum', 'Wassertemperatur...' but values had time.

    # We will use what headers we found.
    df_headers = headers

    rows = table.find('tbody').find_all("tr")
    if not rows:
        print(f"   [WARN] No data rows (tr) found in table body for URL: {url}")
        return pd.DataFrame()
    data = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        # Handle colspan or mismatch? Usually just text.
        row_data = {df_headers[i]: cell.get_text(strip=True) for i, cell in enumerate(cells) if i < len(df_headers)}
        data.append(row_data)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)

    # Flexible Date Parsing
    # Case 1: 'Datum/Uhrzeit' column
    if 'Datum/Uhrzeit' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Datum/Uhrzeit'], format='%d.%m.%Y %H:%M', errors='coerce')

    # Case 2: Separate 'Datum' and 'Uhrzeit'
    elif 'Datum' in df.columns and 'Uhrzeit' in df.columns:
        df['timestamp_str'] = df['Datum'] + ' ' + df['Uhrzeit']
        df["timestamp"] = pd.to_datetime(df['timestamp_str'], format="%d.%m.%Y %H:%M", errors='coerce')

    # Case 3: Only 'Datum' but it might contain time? Or we check if 'Uhrzeit' is implicit?
    # Sometimes GKD puts 'Datum' header but cells have 'dd.mm.yyyy HH:MM' or just date.
    elif 'Datum' in df.columns:
        # Check first non-empty value length or format
        sample = df['Datum'].iloc[0] if not df.empty else ""
        if len(sample) > 10: # "dd.mm.yyyy" is 10 chars. "dd.mm.yyyy HH:MM" is 16.
             df["timestamp"] = pd.to_datetime(df['Datum'], format="%d.%m.%Y %H:%M", errors='coerce')
             # Fallback if that fails
             if df['timestamp'].isna().all():
                 df["timestamp"] = pd.to_datetime(df['Datum'], format="%d.%m.%Y", errors='coerce')
        else:
             df["timestamp"] = pd.to_datetime(df['Datum'], format="%d.%m.%Y", errors='coerce')

    else:
        # Try to find any column that looks like a date
        print(f"   [WARN] 'Datum' column not found. Headers: {df.columns.tolist()}")
        return pd.DataFrame()

    df.dropna(subset=['timestamp'], inplace=True)
    if df.empty:
        print(f"   [WARN] No valid timestamps could be parsed.")
        return pd.DataFrame()

    # Column name extraction
    # Look for 'Wassertemperatur' or the specific name passed
    target_col_candidates = [col for col in df.columns if column_name.split('_')[0].lower() in col.lower()]
    if not target_col_candidates:
        target_col_candidates = [col for col in df.columns if "wassertemperatur" in col.lower()]

    if not target_col_candidates:
        print(f"   [ERROR] Expected header containing '{column_name}' not found. Available headers: {df.columns.tolist()}")
        return pd.DataFrame()
    target_header = target_col_candidates[0]

    df_final = df[["timestamp", target_header]].copy()
    df_final.rename(columns={target_header: column_name}, inplace=True)

    # Numeric conversion
    val_series = df_final[column_name].astype(str).str.replace("--", "", regex=False).str.replace(",", ".", regex=False)
    df_final[column_name] = pd.to_numeric(val_series, errors='coerce')

    # Resample to hourly frequency
    df_final.set_index('timestamp', inplace=True)

    if len(df_final) > 1:
        # Check freq
        time_diff = df_final.index[1] - df_final.index[0]
        # If daily (>= 20h), interpolate. If hourly, resample/mean/first.
        if time_diff >= timedelta(hours=20):
            df_final = df_final.resample('1h').interpolate(method='time')
        else:
            df_final = df_final.resample('1h').first()
    else:
        df_final = df_final.resample('1h').first()

    df_final.reset_index(inplace=True)

    print(f"   [SUCCESS] Found and processed {len(df_final)} rows.")
    return df_final

def get_eisbach_data() -> pd.DataFrame:
    """
    Fetches, cleans, and merges water temperature and weather data.
    Returns a dataframe ready for TimeSeriesPredictor.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=TAGE_VERGANGENHEIT)
    end_date_str = end_date.strftime("%d.%m.%Y")
    start_date_str = start_date.strftime("%d.%m.%Y")

    print(f"--- Starting data fetch for period: {start_date_str} to {end_date_str} ---\n")

    wassertemperatur_url = f"https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/muenchen-himmelreichbruecke-16515005/messwerte/tabelle?beginn={start_date_str}&ende={end_date_str}"
    df_wassertemperatur = fetch_data_from_url(wassertemperatur_url, "wassertemp")

    if df_wassertemperatur.empty:
        print("Warning: Water temperature data is empty.")

    if not df_wassertemperatur.empty:
        df_wassertemperatur['timestamp'] = df_wassertemperatur['timestamp'].dt.tz_localize('Europe/Berlin', ambiguous='NaT', nonexistent='NaT')

    # Fetch Weather Data
    df_wetter = get_prepared_weather_data()
    # Note: df_wetter index is timestamp (datetime with TZ)

    # Merge
    df_merged = pd.DataFrame()
    if not df_wassertemperatur.empty:
        df_merged = df_wassertemperatur.copy()
        if 'timestamp' in df_merged.columns:
            df_merged.set_index('timestamp', inplace=True)

    if not df_wetter.empty:
        # df_wetter has index timestamp
        if df_merged.empty:
            df_merged = df_wetter.copy()
        else:
             # Merge carefully to preserve all timestamps
             df_merged = df_merged.join(df_wetter, how='outer')

    # Sort
    if not df_merged.empty:
        df_merged.sort_index(inplace=True)

        # Determine the last timestamp where we had valid water temperature data.
        # We assume df_wassertemperatur provides 'wassertemp'.
        last_valid_target_ts = None
        if 'wassertemp' in df_merged.columns:
            last_valid_target_ts = df_merged['wassertemp'].last_valid_index()

        # Interpolate missing values
        # We can interpolate weather columns freely.
        # But 'wassertemp' should NOT be filled into the future (beyond last valid measurement).
        # However, small gaps within the history should be filled.

        cols_to_interp = [c for c in df_merged.columns if c != 'wassertemp']
        df_merged[cols_to_interp] = df_merged[cols_to_interp].interpolate(method='time')
        df_merged[cols_to_interp] = df_merged[cols_to_interp].ffill().bfill()

        if 'wassertemp' in df_merged.columns:
            # Interpolate only up to the last valid index to fill gaps
            if last_valid_target_ts is not None:
                # We temporarily take the slice up to last valid
                series = df_merged.loc[:last_valid_target_ts, 'wassertemp']
                series_filled = series.interpolate(method='time').ffill().bfill()

                # Update original
                df_merged.loc[:last_valid_target_ts, 'wassertemp'] = series_filled

                # Ensure everything AFTER last_valid_target_ts is NaN for wassertemp
                # (Should be already, but just in case)
                df_merged.loc[last_valid_target_ts + timedelta(seconds=1):, 'wassertemp'] = None

        # Reset index to make timestamp a column again for TimeSeriesDataFrame
        df_merged.reset_index(inplace=True)

    # Clean up column names for AutoGluon
    df_merged = df_merged.loc[:,~df_merged.columns.duplicated()]

    return df_merged
