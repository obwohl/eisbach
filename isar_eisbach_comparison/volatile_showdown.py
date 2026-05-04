import pandas as pd
import numpy as np
import torch
import os
import subprocess
import sys
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Baseline uses a hardcoded 96h shift to simulate future weather
    df_trunc['airtemp_96'] = df_trunc['temperature'].shift(-96)
    df_trunc['pressure_96'] = df_trunc['pressure_msl'].shift(-96)

    df_baseline = df_trunc.iloc[:train_end].copy()

    df_long = pd.melt(df_baseline.reset_index(), id_vars=['timestamp'], value_vars=['wassertemp_eisbach', 'airtemp_96', 'pressure_96'])
    df_long.columns = ['date', 'cols', 'data']
    df_long['cols'] = df_long['cols'].replace({'wassertemp_eisbach': 'wassertemp'})
    df_long['cols'] = pd.Categorical(df_long['cols'], categories=['wassertemp', 'airtemp_96', 'pressure_96'], ordered=True)
    df_long = df_long.sort_values(by=['cols', 'date'])

    os.makedirs('data', exist_ok=True)
    df_long.to_csv('data/df_long_volatile.csv', index=False)

    subprocess.run([
        sys.executable, 'ts_proba_cuda/run_single_forecast.py',
        '--checkpoint', 'ts_proba_cuda/checkpoints/best_model.pt',
        '--data-file', 'data/df_long_volatile.csv',
        '--output-csv', 'data/inference_volatile.csv'
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    df_inf = pd.read_csv('data/inference_volatile.csv', parse_dates=[0], index_col=0)

    target_true = df_ffill['wassertemp_eisbach'].iloc[train_end:train_end+horizon].values
    quantiles_to_check = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

    # Baseline model outputs quantiles, but might have duplicate columns if multiple series were exported
    # The fix is to match exact float names and just take the first matching Series
    clean_cols = []
    for c in df_inf.columns:
        if 'q' in str(c):
            clean_cols.append(float(str(c).split('q')[-1]))
        else:
            try:
                clean_cols.append(float(c))
            except:
                clean_cols.append(c)
    df_inf.columns = clean_cols

    q_preds = np.zeros((len(quantiles_to_check), horizon))
    for i, q in enumerate(quantiles_to_check):
        if q in df_inf.columns:
            # Handle if there are multiple columns with the same name
            col_data = df_inf[q]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            q_preds[i, :] = col_data.values[:horizon]
        else:
            q_preds[i, :] = np.nan

    median_col = df_inf[0.5]
    if isinstance(median_col, pd.DataFrame):
        median_col = median_col.iloc[:, 0]
    median_pred = median_col.values[:horizon]
    mae = mean_absolute_error(target_true, median_pred)
    crps_val = crps(target_true, q_preds, np.array(quantiles_to_check))

    return mae, crps_val, q_preds

def find_volatile_windows(df, num_24=5, num_96=4):
    target = df['wassertemp_eisbach'].values
    n = len(target)

    # 1. Find 5 windows of 24h with highest variance
    var_24 = []
    for i in range(1000, n - max(96, 24) - 1):
        var_24.append((i, np.var(target[i:i+24])))
    var_24.sort(key=lambda x: x[1], reverse=True)

    chosen_24 = []
    for idx, var in var_24:
        # Check overlap
        overlap = False
        for c in chosen_24:
            if abs(idx - c) < 24:
                overlap = True
                break
        if not overlap:
            chosen_24.append(idx)
        if len(chosen_24) == num_24:
            break

    # 2. Find 4 windows of 96h (4 days) that are "interesting for swimmers"
    # Temp between 12 and 18 degrees, high variance, strong downward or upward trends
    scores_96 = []
    for i in range(1000, n - 96 - 1):
        window_data = target[i:i+96]
        mean_t = np.mean(window_data)
        if 13 <= mean_t <= 16:
            var = np.var(window_data)
            trend = abs(window_data[0] - window_data[-1])
            score = var * trend
            scores_96.append((i, score))

    scores_96.sort(key=lambda x: x[1], reverse=True)

    chosen_96 = []
    for idx, score in scores_96:
        overlap = False
        for c in chosen_96:
            if abs(idx - c) < 96:
                overlap = True
                break
        if not overlap:
            chosen_96.append(idx)
        if len(chosen_96) == num_96:
            break

    return chosen_24, chosen_96

def run():
    csv_path = "isar_eisbach_comparison/isar_eisbach_10_years.csv"
    weather_csv = "isar_eisbach_comparison/weather_10_years_pressure.csv"
    if not os.path.exists(csv_path) or not os.path.exists(weather_csv):
        print("Data not found.")
        return

    df_combined = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_weather = pd.read_csv(weather_csv, index_col=0, parse_dates=True)

    df_ffill = pd.merge(df_combined, df_weather, left_index=True, right_index=True, how='inner').ffill().bfill()

    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.bfloat16)
    chronos_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    chosen_24, chosen_96 = find_volatile_windows(df_ffill)

    os.makedirs('isar_eisbach_comparison/plots', exist_ok=True)

    for horizons, windows, prefix in [(24, chosen_24, "volatile_24h"), (96, chosen_96, "volatile_96h")]:
        print(f"\n--- Evaluating {len(windows)} windows for {horizons}h ---")
        for idx_w, train_end in enumerate(windows):
            target_true = df_ffill['wassertemp_eisbach'].iloc[train_end:train_end+horizons].values
            date_range = df_ffill.index[train_end:train_end+horizons]

            # Baseline
            mae_base, crps_base, q_preds_base = evaluate_baseline_model(train_end, horizons, df_ffill)

            # Chronos-2 Multivariate
            ctx_len = min(512, train_end)
            train_start = train_end - ctx_len

            inputs_eisbach_mult = [{
                "target": df_ffill['wassertemp_eisbach'].iloc[train_start:train_end].values,
                "past_covariates": {
                    "temperature": df_ffill['temperature'].iloc[train_start:train_end].values,
                    "isar": df_ffill['wassertemp_isar'].iloc[train_start:train_end].values
                },
                "future_covariates": {
                    "temperature": df_ffill['temperature'].iloc[train_end:train_end+horizons].values
                }
            }]

            try:
                quantiles_tensor_eisbach_mult, _ = pipeline.predict_quantiles(inputs_eisbach_mult, prediction_length=horizons, quantile_levels=chronos_quantiles)
                q_preds_chronos = quantiles_tensor_eisbach_mult[0][0].numpy().T
                mae_chronos = mean_absolute_error(target_true, q_preds_chronos[4, :])
                crps_chronos = crps(target_true, q_preds_chronos, np.array(chronos_quantiles))
            except Exception as e:
                print(f"Chronos error: {e}")
                continue

            # Plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
            fig.suptitle(f"Eisbach Volatility: {horizons}h starting {date_range[0].strftime('%Y-%m-%d %H:%00')}", fontsize=16)

            # Subplot 1: Baseline
            ax1.plot(date_range, target_true, color='black', label='True Temp', linewidth=2)
            ax1.plot(date_range, q_preds_base[3, :], color='red', label='Median Forecast', linestyle='--')
            ax1.fill_between(date_range, q_preds_base[1, :], q_preds_base[5, :], color='red', alpha=0.3, label='5%-95% Quantile')
            ax1.fill_between(date_range, q_preds_base[2, :], q_preds_base[4, :], color='red', alpha=0.5, label='25%-75% Quantile')
            ax1.set_title(f"Baseline (Custom Model)\nMAE: {mae_base:.3f} | CRPS: {crps_base:.3f}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)

            # Subplot 2: Chronos-2
            ax2.plot(date_range, target_true, color='black', label='True Temp', linewidth=2)
            ax2.plot(date_range, q_preds_chronos[4, :], color='blue', label='Median Forecast', linestyle='--')
            ax2.fill_between(date_range, q_preds_chronos[0, :], q_preds_chronos[8, :], color='blue', alpha=0.3, label='10%-90% Quantile')
            ax2.fill_between(date_range, q_preds_chronos[2, :], q_preds_chronos[6, :], color='blue', alpha=0.5, label='30%-70% Quantile')
            ax2.set_title(f"Chronos-2 (Multivariate)\nMAE: {mae_chronos:.3f} | CRPS: {crps_chronos:.3f}")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(f"isar_eisbach_comparison/plots/{prefix}_{idx_w+1}.png")
            plt.close()
            print(f"Generated plot for {prefix} #{idx_w+1}")

if __name__ == "__main__":
    run()
