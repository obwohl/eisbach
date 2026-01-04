import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from src.data import fetch_data_from_url, fetch_brightsky_data, TARGET_TIMEZONE

def get_long_historical_data(days_back=1095): # ~3 years
    """
    Fetches historical data for a long period to find high variance windows.
    Fetches in chunks to avoid timeouts or daily aggregation if possible.
    """
    end_date_total = datetime.now()
    start_date_total = end_date_total - timedelta(days=days_back)

    print(f"--- Fetching Long History: {start_date_total.date()} to {end_date_total.date()} ---")

    # 1. Water Temp
    # Chunk size reduced to 35 days to ensure hourly resolution from GKD
    chunk_size_days = 35
    current_start = start_date_total

    water_dfs = []

    while current_start < end_date_total:
        current_end = min(current_start + timedelta(days=chunk_size_days), end_date_total)

        s_str = current_start.strftime("%d.%m.%Y")
        e_str = current_end.strftime("%d.%m.%Y")

        url = f"https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/muenchen-himmelreichbruecke-16515005/messwerte/tabelle?beginn={s_str}&ende={e_str}"
        print(f"  Fetching water chunk: {s_str} - {e_str}")

        df_chunk = fetch_data_from_url(url, "wassertemp")
        if not df_chunk.empty:
            water_dfs.append(df_chunk)

        current_start = current_end + timedelta(days=1) # Next day start
        time.sleep(0.5) # Be nice

    if not water_dfs:
        print("Error: No water temp data found.")
        return pd.DataFrame()

    df_water = pd.concat(water_dfs).drop_duplicates(subset='timestamp').sort_values('timestamp')
    df_water['timestamp'] = df_water['timestamp'].dt.tz_localize(TARGET_TIMEZONE, ambiguous='NaT', nonexistent='NaT')

    # 2. Weather
    print("Fetching weather data in chunks...")
    weather_dfs = []
    current_start = start_date_total

    # Brightsky can handle larger chunks usually, but we match for consistency/safety
    while current_start < end_date_total:
        current_end = min(current_start + timedelta(days=chunk_size_days), end_date_total)

        # fetch_brightsky_data
        df_w_chunk = fetch_brightsky_data(current_start, current_end, "03379")
        if df_w_chunk is not None and not df_w_chunk.empty:
            weather_dfs.append(df_w_chunk)

        current_start = current_end + timedelta(days=1)
        time.sleep(0.2)

    if not weather_dfs:
        print("Error: No weather data found.")
        return pd.DataFrame()

    df_weather_raw = pd.concat(weather_dfs).drop_duplicates(subset='timestamp').sort_values('timestamp')

    # Process weather
    wetter_df = df_weather_raw[['timestamp', 'temperature', 'precipitation', 'pressure_msl']].copy()
    wetter_df['timestamp'] = pd.to_datetime(wetter_df['timestamp'])
    wetter_df.set_index('timestamp', inplace=True)
    wetter_df.index = wetter_df.index.tz_convert(TARGET_TIMEZONE)
    wetter_df.sort_index(inplace=True)
    wetter_df = wetter_df[~wetter_df.index.duplicated(keep='first')]
    wetter_df.rename(columns={'temperature': 'lufttemperatur_c', 'precipitation': 'niederschlag_mm', 'pressure_msl': 'pressure'}, inplace=True)

    # Resample to hourly
    wetter_1h = wetter_df.resample('1h').agg({
        'lufttemperatur_c': 'mean',
        'niederschlag_mm': 'sum',
        'pressure': 'mean'
    }).round(2)

    # Merge
    df_water.set_index('timestamp', inplace=True)
    df_merged = df_water.join(wetter_1h, how='inner')

    # Interpolate small gaps
    df_merged = df_merged.interpolate(method='time', limit=24)
    df_merged = df_merged.dropna()

    df_merged.reset_index(inplace=True)
    return df_merged

def find_high_variance_windows(df: pd.DataFrame, window_size=96, top_n=10):
    """
    Finds top N windows of size `window_size` (hours) where:
    1. Min water temp in window > 10.0
    2. Variance of water temp is maximized.
    """
    print("Searching for high variance windows...")
    results = []

    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculate rolling metrics
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)

    rolling_var = df['wassertemp'].rolling(window=indexer).var()
    rolling_min = df['wassertemp'].rolling(window=indexer).min()

    # Create a DataFrame for candidates
    candidates = pd.DataFrame({
        'start_idx': df.index,
        'timestamp': df['timestamp'],
        'variance': rolling_var,
        'min_temp': rolling_min
    })

    # Filter
    valid_candidates = candidates[candidates['min_temp'] > 10.0].copy()

    if valid_candidates.empty:
        print("No windows found with temp > 10.0°C. Relaxing constraint to > 5.0°C for testing.")
        valid_candidates = candidates[candidates['min_temp'] > 5.0].copy()

    # Sort by variance descending
    valid_candidates = valid_candidates.sort_values('variance', ascending=False)

    # Select top N non-overlapping windows
    final_windows = []
    taken_indices = set()

    for _, row in valid_candidates.iterrows():
        if len(final_windows) >= top_n:
            break

        start_idx = int(row['start_idx'])
        end_idx = start_idx + window_size

        # Check overlap (any shared index)
        # Using a set for indices check might be slow if many windows.
        # But top_n is small.
        indices = range(start_idx, end_idx)
        if any(idx in taken_indices for idx in indices):
            continue

        # Add to taken
        for idx in indices:
            taken_indices.add(idx)

        start_time = row['timestamp']
        end_time = df.iloc[end_idx-1]['timestamp'] if end_idx < len(df) else df.iloc[-1]['timestamp']

        final_windows.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': start_time,
            'end_time': end_time,
            'variance': row['variance'],
            'min_temp': row['min_temp']
        })

    print(f"Found {len(final_windows)} windows.")
    return final_windows, df
