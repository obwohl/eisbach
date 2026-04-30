import pandas as pd
from datetime import datetime, timedelta, timezone
import requests
from bs4 import BeautifulSoup

def fetch_brightsky_data(start_date: datetime, end_date: datetime, station_id: str) -> pd.DataFrame | None:
    TARGET_TIMEZONE = 'Europe/Berlin'
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

def get_prepared_weather_data():
    TARGET_TIMEZONE = 'Europe/Berlin'
    TAGE_VERGANGENHEIT = 40
    TAGE_ZUKUNFT = 8
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

    # Bright Sky liefert UTC, hier konvertieren wir in die Zielzone
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

def fetch_data_from_url(url, column_name):
    print(f"-> Processing URL for: {column_name}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        html_content = response.content.decode('utf-8')
    except Exception as e:
        print(f"Fehler beim Laden der URL: {e}")
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
        df['timestamp'] = pd.to_datetime(df['Datum/Uhrzeit'], format='%d.%m.%Y %H:%M', errors='coerce')
    elif 'Datum' in df.columns and 'Uhrzeit' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Datum'] + ' ' + df['Uhrzeit'], format="%d.%m.%Y %H:%M", errors='coerce')

    df.dropna(subset=['timestamp'], inplace=True)

    # Spalte für Messwert finden
    matching_cols = [c for c in df.columns if column_name.split('_')[0].lower() in c.lower()]
    if not matching_cols:
        print(f"Warning: No matching column found for {column_name}")
        return pd.DataFrame()
    target_col = matching_cols[0]

    df_final = df[["timestamp", target_col]].copy()
    df_final.rename(columns={target_col: column_name}, inplace=True)
    df_final[column_name] = pd.to_numeric(df_final[column_name].astype(str).str.replace(",", "."), errors='coerce')

    return df_final


def prepare_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=40)
    wassertemperatur_url = f"https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/muenchen-himmelreichbruecke-16515005/messwerte/tabelle?beginn={start_date.strftime('%d.%m.%Y')}&ende={end_date.strftime('%d.%m.%Y')}"

    # 1. Daten laden
    df_wt = fetch_data_from_url(wassertemperatur_url, "wassertemp")

    if df_wt.empty:
        print("Error: Water temperature data is empty. Aborting preparation.")
        return pd.DataFrame(), pd.DataFrame()

    # 2. KEY FIX: Zeitumstellung robust handhaben (Frühling & Herbst)
    # 'nonexistent' fängt den 29.03.2026 02:00 Uhr ab, 'ambiguous' den Herbst.
    df_wt['timestamp'] = df_wt['timestamp'].dt.tz_localize(
        'Europe/Berlin',
        ambiguous='infer',
        nonexistent='shift_forward'
    )

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

    return df_long, df_wetter
