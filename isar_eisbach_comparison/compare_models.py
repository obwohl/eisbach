import pandas as pd
import numpy as np
import torch
import os
import subprocess
import sys
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

def evaluate_baseline_model(train_end, horizon, df_ffill):
    max_idx = train_end + horizon - 1

    df_trunc = df_ffill.iloc[:max_idx + 1].copy()

    try:
        df_trunc.index = df_trunc.index.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')
    except:
        pass

    try:
        df_trunc.index = df_trunc.index.tz_convert('UTC')
    except:
        df_trunc.index = pd.to_datetime(df_trunc.index, utc=True)

    df_trunc.index = df_trunc.index.tz_localize(None)

    df_trunc['airtemp_96'] = df_trunc['temperature'].shift(-horizon)
    df_trunc['pressure_96'] = df_trunc['pressure_msl'].shift(-horizon)

    df_baseline = df_trunc.iloc[:train_end].copy()

    df_long = pd.melt(df_baseline.reset_index(), id_vars=['timestamp'], value_vars=['wassertemp_eisbach', 'airtemp_96', 'pressure_96'])
    df_long.columns = ['date', 'cols', 'data']
    df_long['cols'] = df_long['cols'].replace({'wassertemp_eisbach': 'wassertemp'})
    df_long['cols'] = pd.Categorical(df_long['cols'], categories=['wassertemp', 'airtemp_96', 'pressure_96'], ordered=True)
    df_long = df_long.sort_values(by=['cols', 'date'])

    os.makedirs('data', exist_ok=True)
    df_long.to_csv('data/df_long_showdown.csv', index=False)

    subprocess.run([
        sys.executable, 'ts_proba_cuda/run_single_forecast.py',
        '--checkpoint', 'ts_proba_cuda/checkpoints/best_model.pt',
        '--data-file', 'data/df_long_showdown.csv',
        '--output-csv', 'data/inference_showdown.csv'
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    df_inf = pd.read_csv('data/inference_showdown.csv', parse_dates=[0], index_col=0)

    target_true = df_ffill['wassertemp_eisbach'].iloc[train_end:train_end+horizon].values

    quantiles_to_check = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    q_preds = np.zeros((len(quantiles_to_check), horizon))
    for i, q in enumerate(quantiles_to_check):
        col_name = f"wassertemp_q{q}"
        if col_name in df_inf.columns:
            # If multiple targets predicted, just take the first column matching (it's 1D)
            q_preds[i, :] = df_inf[col_name].values[:horizon]
        else:
            q_preds[i, :] = np.nan

    median_pred = df_inf["wassertemp_q0.5"].values[:horizon]
    mae = mean_absolute_error(target_true, median_pred)
    crps_val = crps(target_true, q_preds, np.array(quantiles_to_check))

    return mae, crps_val

def run():
    csv_path = "isar_eisbach_comparison/isar_eisbach_10_years.csv"
    weather_csv = "isar_eisbach_comparison/weather_10_years_pressure.csv"
    if not os.path.exists(csv_path) or not os.path.exists(weather_csv):
        print("Data not found.")
        return

    df_combined = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_weather = pd.read_csv(weather_csv, index_col=0, parse_dates=True)

    df_ffill = pd.merge(df_combined, df_weather, left_index=True, right_index=True, how='inner')
    df_ffill = df_ffill.ffill().bfill()

    horizon = 96
    end_idx = len(df_ffill) - horizon - 1

    windows = []
    for i in range(10):
        windows.append(end_idx - (i * 24 * 30))

    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    chronos_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []
    print("Starting Model Showdown across 10 windows...")

    agg_chronos_mae = []
    agg_chronos_crps = []
    agg_base_mae = []
    agg_base_crps = []

    for i, train_end in enumerate(windows):
        print(f"Evaluating Window {i+1}/10...")

        target_true = df_ffill['wassertemp_eisbach'].iloc[train_end:train_end+horizon].values

        mae_base, crps_base = evaluate_baseline_model(train_end, horizon, df_ffill)
        agg_base_mae.append(mae_base)
        agg_base_crps.append(crps_base)

        ctx_len = min(512, train_end)
        train_start = train_end - ctx_len

        inputs_eisbach_mult = [{
            "target": df_ffill['wassertemp_eisbach'].iloc[train_start:train_end].values,
            "past_covariates": {
                "temperature": df_ffill['temperature'].iloc[train_start:train_end].values,
                "isar": df_ffill['wassertemp_isar'].iloc[train_start:train_end].values
            },
            "future_covariates": {
                "temperature": df_ffill['temperature'].iloc[train_end:train_end+horizon].values
            }
        }]

        try:
            quantiles_tensor_eisbach_mult, _ = pipeline.predict_quantiles(inputs_eisbach_mult, prediction_length=horizon, quantile_levels=chronos_quantiles)
            q_preds_mult = quantiles_tensor_eisbach_mult[0][0].numpy().T
            mae_chronos = mean_absolute_error(target_true, q_preds_mult[4, :])
            crps_chronos = crps(target_true, q_preds_mult, np.array(chronos_quantiles))
        except Exception as e:
            print(f"Chronos error: {e}")
            mae_chronos, crps_chronos = np.nan, np.nan

        agg_chronos_mae.append(mae_chronos)
        agg_chronos_crps.append(crps_chronos)

    print("\n--- Showdown Results ---")
    results.append({
        'Model': 'Baseline (Custom Eisbach Model)',
        'MAE (Mean)': np.nanmean(agg_base_mae),
        'CRPS (Mean)': np.nanmean(agg_base_crps)
    })
    results.append({
        'Model': 'Chronos-2 (Multivariate Coupled)',
        'MAE (Mean)': np.nanmean(agg_chronos_mae),
        'CRPS (Mean)': np.nanmean(agg_chronos_crps)
    })

    df_res = pd.DataFrame(results)
    df_res.to_csv("isar_eisbach_comparison/model_showdown.csv", index=False)
    print(df_res)

if __name__ == "__main__":
    run()
