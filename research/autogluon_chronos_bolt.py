"""Archived: AutoGluon / Chronos-Bolt forecasting pipeline (Kaggle notebook, 2025).

ARCHIVED, UNMAINTAINED RESEARCH CODE. NOT PART OF THE PRODUCTION PIPELINE.

Provenance
----------
Extracted verbatim from cell 7 of the former ``eisbach_2025_backup.ipynb`` (since
deleted). The code is preserved exactly as it was written; it has deliberately NOT
been fixed, modernised, or adapted. Do not treat it as a working script.

Why it will not run here
------------------------
* It was written for a Kaggle notebook environment: it imports ``kaggle_secrets``
  (``UserSecretsClient``) and ``boto3``/S3, neither of which exists or is configured
  outside Kaggle. It will fail at import time in this repository.
* It requires ``autogluon`` (``autogluon.timeseries``, ``autogluon.common``), which is
  NOT listed in ``requirements.txt`` and is not installed by the production setup. It
  also pulls in ``bokeh`` and ``scipy``, likewise undeclared.
* It writes ``eisbach_predictions.csv`` into the current working directory and fetches
  live data from Bright Sky and gkd.bayern.de at run time.

What it does
------------
Fetches ~370 days of Bright Sky weather and ~364 days of Eisbach water temperature
(Himmelreichbrücke), then fits an AutoGluon ``TimeSeriesPredictor`` restricted to
``hyperparameters={"Chronos": [{"model_path": "bolt_base"}]}`` for a 64-hour horizon,
and merges the resulting quantiles with the weather frame.

Status: parked, not abandoned
-----------------------------
Benchmarking AutoGluon / Chronos-Bolt against the production DUET-Prob model remains a
possible future direction. This file is kept so that the pipeline does not have to be
reconstructed from scratch if that comparison is ever picked back up. It is simply not
work that is currently in flight.
"""

from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.timeseries.splitter import ExpandingWindowSplitter
from autogluon.common import space
from contextlib import redirect_stdout
from scipy.stats import randint, uniform
from bokeh.plotting import figure, show
from bokeh.palettes import Category10

from bokeh.models import HoverTool, ColumnDataSource, Legend, Segment, Text, VBar, LinearAxis, Range1d
from bokeh.layouts import column

import logging
from collections import deque
import time
from bokeh.io import save
import re
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from kaggle_secrets import UserSecretsClient

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
        return df
    except requests.exceptions.RequestException as e:
        print(f"Netzwerk- oder API-Fehler beim Abrufen der Wetterdaten: {e}")
        return None

def get_prepared_weather_data():
    TARGET_TIMEZONE = 'Europe/Berlin'
    TAGE_VERGANGENHEIT = 370
    TAGE_ZUKUNFT = 8
    now_local = datetime.now().astimezone()
    start_date = now_local - timedelta(days=TAGE_VERGANGENHEIT)
    end_date = now_local + timedelta(days=TAGE_ZUKUNFT)

    df_raw = fetch_brightsky_data(start_date, end_date, "03379")
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    wetter_df = df_raw[['timestamp', 'temperature', 'precipitation']].copy()
    wetter_df['timestamp'] = pd.to_datetime(wetter_df['timestamp'])
    wetter_df.set_index('timestamp', inplace=True)
    wetter_df.index = wetter_df.index.tz_convert(TARGET_TIMEZONE)
    wetter_df.sort_index(inplace=True)
    wetter_df = wetter_df[~wetter_df.index.duplicated(keep='first')]
    wetter_df.rename(columns={'temperature': 'lufttemperatur_c', 'precipitation': 'niederschlag_mm'}, inplace=True)
    wetter_df['niederschlag_mm'] = wetter_df['niederschlag_mm'].fillna(0)
    wetter_df['lufttemperatur_c'] = wetter_df['lufttemperatur_c'].interpolate(method='time')

    wetter_1h = wetter_df.resample('1h').agg({'lufttemperatur_c': 'mean', 'niederschlag_mm': 'sum'}).round(2)
    return wetter_1h

def fetch_data_from_url(url, column_name):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        html_content = response.content.decode('utf-8')
    except Exception: return pd.DataFrame()
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
    target_header = column_name.split('_')[0]
    df_final = df[["timestamp", target_header]].copy()
    df_final.rename(columns={target_header: column_name}, inplace=True)
    df_final[column_name] = pd.to_numeric(df_final[column_name].astype(str).str.replace(",", "."), errors='coerce')
    return df_final

def main():
    wetter_data_1h = get_prepared_weather_data()
    end_date = datetime.now() - timedelta(hours=1)
    start_date = end_date - timedelta(days=364)
    
    urls_and_columns = {f"https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/muenchen-himmelreichbruecke-16515005/messwerte/tabelle?beginn={start_date.strftime('%d.%m.%Y')}&ende={end_date.strftime('%d.%m.%Y')}": "Wassertemperatur [°C]_München Himmelreichbruecke"}
    all_dfs = [fetch_data_from_url(url, col) for url, col in urls_and_columns.items()]
    merged_data = all_dfs[0].sort_values('timestamp')
    
    merged_data.rename(columns={merged_data.columns[1]: "wassertemp"}, inplace=True)
    water_hourly = merged_data.set_index("timestamp").resample("1h").median().reset_index()
    water_hourly["item_id"] = "eisbach_temp"
    
    data = TimeSeriesDataFrame.from_data_frame(water_hourly)
    PREDICTION_LENGTH = 64
    predictor = TimeSeriesPredictor(prediction_length=PREDICTION_LENGTH, target="wassertemp", eval_metric="SQL", verbosity=0)
    predictor.fit(data, hyperparameters={"Chronos": [{"model_path": "bolt_base", "ag_args": {"name_suffix": "bolt_base"}}]}, time_limit=300)
    
    future_pred = predictor.predict(data)
    preds_for_csv = future_pred.copy().reset_index(level='item_id', drop=True)

    # FIXED: Handhabung von Duplikaten vor dem Merge
    if not wetter_data_1h.empty:
        # Sicherstellen, dass der Index keine Duplikate hat
        preds_for_csv = preds_for_csv[~preds_for_csv.index.duplicated(keep='first')]
        wetter_data_1h = wetter_data_1h[~wetter_data_1h.index.duplicated(keep='first')]
        
        # Zeitzonen-Angleichung
        if preds_for_csv.index.tz is None:
            preds_for_csv.index = preds_for_csv.index.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')
        
        preds_for_csv = preds_for_csv.merge(wetter_data_1h, left_index=True, right_index=True, how='left')

    preds_for_csv[["0.1", "0.5", "0.9"]].to_csv("eisbach_predictions.csv", float_format="%.1f")
    print("Erfolgreich gespeichert.")

if __name__ == "__main__":
    main()