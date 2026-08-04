import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import logging
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data_from_url(url, column_name):
    logging.info(f"Processing URL for: {column_name}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=40, headers=headers)
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
        df['timestamp'] = pd.to_datetime(df['Datum/Uhrzeit'], format='%d.%m.%Y %H:%M', errors='coerce')
    elif 'Datum' in df.columns and 'Uhrzeit' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Datum'] + ' ' + df['Uhrzeit'], format="%d.%m.%Y %H:%M", errors='coerce')

    df.dropna(subset=['timestamp'], inplace=True)

    matching_cols = [c for c in df.columns if column_name.split('_')[0].lower() in c.lower()]
    if not matching_cols:
        logging.warning(f"No matching column found for {column_name}")
        return pd.DataFrame()
    target_col = matching_cols[0]

    df_final = df[["timestamp", target_col]].copy()
    df_final.rename(columns={target_col: column_name}, inplace=True)
    df_final[column_name] = pd.to_numeric(df_final[column_name].astype(str).str.replace(",", "."), errors='coerce')

    return df_final

def download_chunk(i):
    end_date = datetime.now()
    start = end_date - timedelta(days=365*(i+1))
    end = end_date - timedelta(days=365*i)

    start_str = start.strftime('%d.%m.%Y')
    end_str = end.strftime('%d.%m.%Y')

    url_eisbach = f"https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/muenchen-himmelreichbruecke-16515005/messwerte/tabelle?beginn={start_str}&ende={end_str}"
    url_isar = f"https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/muenchen-16005701/messwerte/tabelle?beginn={start_str}&ende={end_str}"

    print(f"Downloading data from {start_str} to {end_str}")
    df_e = fetch_data_from_url(url_eisbach, "wassertemp_eisbach")
    df_i = fetch_data_from_url(url_isar, "wassertemp_isar")
    return df_e, df_i

def download_long_term_data(years_back=10):
    eisbach_dfs = []
    isar_dfs = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(download_chunk, range(years_back)))

    for e, i in results:
        if not e.empty: eisbach_dfs.append(e)
        if not i.empty: isar_dfs.append(i)

    if not eisbach_dfs or not isar_dfs:
        raise Exception("No data could be downloaded.")

    df_eisbach = pd.concat(eisbach_dfs).drop_duplicates('timestamp')
    df_isar = pd.concat(isar_dfs).drop_duplicates('timestamp')

    # Sort and remove duplicates from exact timestamp match before localizing
    df_eisbach = df_eisbach.sort_values('timestamp').drop_duplicates('timestamp')
    df_isar = df_isar.sort_values('timestamp').drop_duplicates('timestamp')

    # Resolve pytz issue: localize on unique sorted index
    df_eisbach = df_eisbach.set_index('timestamp')
    df_isar = df_isar.set_index('timestamp')

    # Remove any potential timezone info if it exists
    if df_eisbach.index.tzinfo is not None:
        df_eisbach.index = df_eisbach.index.tz_localize(None)
    if df_isar.index.tzinfo is not None:
        df_isar.index = df_isar.index.tz_localize(None)

    # Localize safely, drop ambiguous times if they still cause issues
    try:
        df_eisbach.index = df_eisbach.index.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')
    except Exception as e:
        print(f"Timezone fix for Eisbach: {e}")
        # Force NaT on ambiguous, then drop
        df_eisbach.index = df_eisbach.index.tz_localize('Europe/Berlin', ambiguous='NaT', nonexistent='shift_forward')
        df_eisbach = df_eisbach[df_eisbach.index.notna()]

    try:
        df_isar.index = df_isar.index.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')
    except Exception as e:
        print(f"Timezone fix for Isar: {e}")
        df_isar.index = df_isar.index.tz_localize('Europe/Berlin', ambiguous='NaT', nonexistent='shift_forward')
        df_isar = df_isar[df_isar.index.notna()]

    df_eisbach = df_eisbach.resample('1h').first()
    df_isar = df_isar.resample('1h').first()

    df_combined = pd.merge(df_eisbach, df_isar, left_index=True, right_index=True, how='inner')
    df_combined['diff'] = df_combined['wassertemp_eisbach'] - df_combined['wassertemp_isar']

    return df_combined

def run():
    print("Fetching long term data (10 years)...")
    csv_path = "isar_eisbach_comparison/isar_eisbach_10_years.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        df = download_long_term_data(10)
        df.to_csv(csv_path)

    print("Generating plots...")
    # 1. Timeseries Diff Plot
    plt.figure(figsize=(15, 6))
    df['diff'].plot(alpha=0.3, color='gray', label='Hourly Difference')
    df['diff'].rolling(24*30, min_periods=1).mean().plot(color='red', label='30-Day Rolling Average')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Temperature Difference over 10 Years (Eisbach - Isar)')
    plt.ylabel('Difference (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('isar_eisbach_comparison/diff_timeseries.png')

    # 2. Daily Pattern (Diurnal Cycle)
    plt.figure(figsize=(12, 6))
    daily_pattern = df.groupby(df.index.hour)['diff'].agg(['mean', 'std'])
    plt.plot(daily_pattern.index, daily_pattern['mean'], marker='o', color='blue')
    plt.fill_between(daily_pattern.index,
                     daily_pattern['mean'] - daily_pattern['std'],
                     daily_pattern['mean'] + daily_pattern['std'],
                     alpha=0.2, color='blue')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Average Daily Pattern of Difference (10 Years)')
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Average Difference (°C)')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('isar_eisbach_comparison/diff_daily_pattern.png')

    # 3. Yearly Pattern (Seasonal Cycle)
    plt.figure(figsize=(12, 6))
    monthly_pattern = df.groupby(df.index.month)['diff'].agg(['mean', 'std'])
    plt.plot(monthly_pattern.index, monthly_pattern['mean'], marker='s', color='green')
    plt.fill_between(monthly_pattern.index,
                     monthly_pattern['mean'] - monthly_pattern['std'],
                     monthly_pattern['mean'] + monthly_pattern['std'],
                     alpha=0.2, color='green')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Average Seasonal Pattern of Difference (10 Years)')
    plt.xlabel('Month of Year (1-12)')
    plt.ylabel('Average Difference (°C)')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('isar_eisbach_comparison/diff_yearly_pattern.png')

    print("Done. Plots saved.")

if __name__ == "__main__":
    run()
