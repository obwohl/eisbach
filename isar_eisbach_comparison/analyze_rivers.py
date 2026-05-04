import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, timezone
import requests
from chronos import BaseChronosPipeline

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def crps(y_true, y_pred_quantiles, quantiles):
    crps_vals = []
    for t in range(len(y_true)):
        y_t = y_true[t]
        q_preds = y_pred_quantiles[:, t]
        sort_idx = np.argsort(quantiles)
        q_preds = q_preds[sort_idx]
        qs = quantiles[sort_idx]
        val = 0
        for i in range(1, len(qs)):
            x_m = (q_preds[i] + q_preds[i-1]) / 2
            p_m = (qs[i] + qs[i-1]) / 2
            if y_t < x_m:
                val += (p_m**2) * (q_preds[i] - q_preds[i-1])
            else:
                val += ((1-p_m)**2) * (q_preds[i] - q_preds[i-1])
        crps_vals.append(val)
    return np.mean(crps_vals)

def get_brightsky_data_longterm(start_date: datetime, end_date: datetime, station_id: str) -> pd.DataFrame:
    dfs = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=365), end_date)
        start_utc = current_start.astimezone(timezone.utc) if current_start.tzinfo else current_start.replace(tzinfo=timezone.utc)
        end_utc = current_end.astimezone(timezone.utc) if current_end.tzinfo else current_end.replace(tzinfo=timezone.utc)
        params = {'dwd_station_id': station_id, 'date': start_utc.isoformat(timespec='seconds'), 'last_date': end_utc.isoformat(timespec='seconds')}
        try:
            response = requests.get("https://api.brightsky.dev/weather", params=params, timeout=30)
            if response.status_code == 200:
                data = response.json().get('weather', [])
                if data: dfs.append(pd.DataFrame(data))
        except:
            pass
        current_start = current_end

    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_convert('Europe/Berlin')
    df = df[~df.index.duplicated(keep='first')]
    return df.resample('1h').agg({'temperature': 'mean'})

def run():
    csv_path = "isar_eisbach_comparison/isar_eisbach_10_years.csv"
    if not os.path.exists(csv_path):
        print("Data not found.")
        return

    df_combined = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    weather_csv = "isar_eisbach_comparison/weather_10_years.csv"
    if os.path.exists(weather_csv):
        df_weather = pd.read_csv(weather_csv, index_col=0, parse_dates=True)
    else:
        print("Fetching 10 years of weather data...")
        start_dt = df_combined.index.min()
        end_dt = df_combined.index.max()
        df_weather = get_brightsky_data_longterm(start_dt, end_dt, "03379")
        df_weather.to_csv(weather_csv)

    df_ffill = pd.merge(df_combined, df_weather, left_index=True, right_index=True, how='inner')
    df_ffill = df_ffill.ffill().bfill()

    print("\n--- Explorative Datenanalyse (10 Jahre) ---")
    print(df_ffill.describe())
    df_diff = df_ffill['wassertemp_eisbach'] - df_ffill['wassertemp_isar']
    print("\nDifferenz (Eisbach - Isar):")
    print(df_diff.describe())

    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    results = []
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for horizons in [24, 96]:
        print(f"\nEvaluating Horizon: {horizons}h")
        train_end = len(df_ffill) - horizons
        if train_end <= 0: continue

        target_eisbach = df_ffill['wassertemp_eisbach'].iloc[train_end:train_end+horizons].values
        target_isar = df_ffill['wassertemp_isar'].iloc[train_end:train_end+horizons].values

        # 1. Univariate
        context_eisbach_uni = torch.tensor(df_ffill['wassertemp_eisbach'].iloc[:train_end].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        quantiles_tensor_eisbach, _ = pipeline.predict_quantiles(context_eisbach_uni, prediction_length=horizons, quantile_levels=quantiles)
        q_preds = quantiles_tensor_eisbach[0][0].numpy().T
        mae_eisbach_uni = mean_absolute_error(target_eisbach, q_preds[4, :])
        crps_eisbach_uni = crps(target_eisbach, q_preds, np.array(quantiles))

        context_isar_uni = torch.tensor(df_ffill['wassertemp_isar'].iloc[:train_end].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        quantiles_tensor_isar, _ = pipeline.predict_quantiles(context_isar_uni, prediction_length=horizons, quantile_levels=quantiles)
        q_preds = quantiles_tensor_isar[0][0].numpy().T
        mae_isar_uni = mean_absolute_error(target_isar, q_preds[4, :])
        crps_isar_uni = crps(target_isar, q_preds, np.array(quantiles))

        results.append({'Horizon': horizons, 'River': 'Eisbach', 'Type': 'Univariate', 'MAE': mae_eisbach_uni, 'CRPS': crps_eisbach_uni})
        results.append({'Horizon': horizons, 'River': 'Isar', 'Type': 'Univariate', 'MAE': mae_isar_uni, 'CRPS': crps_isar_uni})

        # 2. Multivariate (Coupled Rivers + Air Temp)
        # We define target as Isar, and past/future covariates as Eisbach + AirTemp.
        # Alternatively, target as Eisbach, and Isar + AirTemp as covariates.

        inputs_eisbach_mult = [{
            "target": df_ffill['wassertemp_eisbach'].iloc[:train_end].values,
            "past_covariates": {
                "temperature": df_ffill['temperature'].iloc[:train_end].values,
                "isar": df_ffill['wassertemp_isar'].iloc[:train_end].values
            },
            "future_covariates": {
                "temperature": df_ffill['temperature'].iloc[train_end:train_end+horizons].values,
                "isar": df_ffill['wassertemp_isar'].iloc[train_end:train_end+horizons].values
            }
        }]

        try:
            quantiles_tensor_eisbach_mult, _ = pipeline.predict_quantiles(inputs_eisbach_mult, prediction_length=horizons, quantile_levels=quantiles)
            q_preds_mult = quantiles_tensor_eisbach_mult[0][0].numpy().T
            mae_eisbach_mult = mean_absolute_error(target_eisbach, q_preds_mult[4, :])
            crps_eisbach_mult = crps(target_eisbach, q_preds_mult, np.array(quantiles))
        except Exception as e:
            print(f"Multivariate error for Eisbach: {e}")
            mae_eisbach_mult, crps_eisbach_mult = np.nan, np.nan

        inputs_isar_mult = [{
            "target": df_ffill['wassertemp_isar'].iloc[:train_end].values,
            "past_covariates": {
                "temperature": df_ffill['temperature'].iloc[:train_end].values,
                "eisbach": df_ffill['wassertemp_eisbach'].iloc[:train_end].values
            },
            "future_covariates": {
                "temperature": df_ffill['temperature'].iloc[train_end:train_end+horizons].values,
                "eisbach": df_ffill['wassertemp_eisbach'].iloc[train_end:train_end+horizons].values
            }
        }]

        try:
            quantiles_tensor_isar_mult, _ = pipeline.predict_quantiles(inputs_isar_mult, prediction_length=horizons, quantile_levels=quantiles)
            q_preds_mult = quantiles_tensor_isar_mult[0][0].numpy().T
            mae_isar_mult = mean_absolute_error(target_isar, q_preds_mult[4, :])
            crps_isar_mult = crps(target_isar, q_preds_mult, np.array(quantiles))
        except Exception as e:
            print(f"Multivariate error for Isar: {e}")
            mae_isar_mult, crps_isar_mult = np.nan, np.nan

        results.append({'Horizon': horizons, 'River': 'Eisbach', 'Type': 'Multivariate (Coupled)', 'MAE': mae_eisbach_mult, 'CRPS': crps_eisbach_mult})
        results.append({'Horizon': horizons, 'River': 'Isar', 'Type': 'Multivariate (Coupled)', 'MAE': mae_isar_mult, 'CRPS': crps_isar_mult})

    df_results = pd.DataFrame(results)
    df_results.to_csv('isar_eisbach_comparison/forecasting_results_10y.csv', index=False)
    print("\nResults saved to isar_eisbach_comparison/forecasting_results_10y.csv")
    print(df_results)

if __name__ == "__main__":
    run()
