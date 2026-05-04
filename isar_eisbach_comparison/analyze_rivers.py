import pandas as pd
import numpy as np
import torch
import os
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

def run():
    csv_path = "isar_eisbach_comparison/isar_eisbach_data.csv"
    if not os.path.exists(csv_path):
        print("Data not found.")
        return

    df_combined = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_ffill = df_combined.ffill().bfill()

    print("\n--- Explorative Datenanalyse ---")
    print(df_combined.describe())
    df_diff = df_combined['wassertemp_eisbach'] - df_combined['wassertemp_isar']
    print("\nDifferenz (Eisbach - Isar):")
    print(df_diff.describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(df_diff.dropna(), kde=True)
    plt.title('Verteilung der Temperaturdifferenz (Eisbach - Isar)')
    plt.xlabel('Temperaturdifferenz (°C)')
    plt.savefig('isar_eisbach_comparison/diff_dist.png')

    plt.figure(figsize=(10, 6))
    plt.scatter(df_combined['wassertemp_isar'], df_combined['wassertemp_eisbach'], alpha=0.5)
    plt.plot([0, 20], [0, 20], 'r--')
    plt.title('Scatterplot Isar vs Eisbach Wassertemperatur')
    plt.xlabel('Isar (°C)')
    plt.ylabel('Eisbach (°C)')
    plt.savefig('isar_eisbach_comparison/scatter.png')

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
        # For chronos 2, shape: (batch, n_variates, history_length). Univariate means n_variates=1
        context_eisbach_uni = torch.tensor(df_ffill['wassertemp_eisbach'].iloc[:train_end].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        quantiles_tensor_eisbach, _ = pipeline.predict_quantiles(context_eisbach_uni, prediction_length=horizons, quantile_levels=quantiles)
        # quantiles_tensor returns a list of tensors for chronos2 (one per batch item)
        # Each tensor has shape (n_variates, prediction_length, len(quantile_levels))
        q_preds = quantiles_tensor_eisbach[0][0].numpy().T # Shape: (len(quantiles), pred_length)
        mae_eisbach_uni = mean_absolute_error(target_eisbach, q_preds[4, :])
        crps_eisbach_uni = crps(target_eisbach, q_preds, np.array(quantiles))

        context_isar_uni = torch.tensor(df_ffill['wassertemp_isar'].iloc[:train_end].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        quantiles_tensor_isar, _ = pipeline.predict_quantiles(context_isar_uni, prediction_length=horizons, quantile_levels=quantiles)
        q_preds = quantiles_tensor_isar[0][0].numpy().T
        mae_isar_uni = mean_absolute_error(target_isar, q_preds[4, :])
        crps_isar_uni = crps(target_isar, q_preds, np.array(quantiles))

        results.append({'Horizon': horizons, 'River': 'Eisbach', 'Type': 'Univariate', 'MAE': mae_eisbach_uni, 'CRPS': crps_eisbach_uni})
        results.append({'Horizon': horizons, 'River': 'Isar', 'Type': 'Univariate', 'MAE': mae_isar_uni, 'CRPS': crps_isar_uni})

        # 2. Multivariate (with future known covariates via the list of dicts approach supported by Chronos 2)
        # We need "target" (history_length,)
        # "past_covariates": {"temperature": (history_length,)}
        # "future_covariates": {"temperature": (prediction_length,)}

        inputs_eisbach_mult = [{
            "target": df_ffill['wassertemp_eisbach'].iloc[:train_end].values,
            "past_covariates": {"temperature": df_ffill['temperature'].iloc[:train_end].values},
            "future_covariates": {"temperature": df_ffill['temperature'].iloc[train_end:train_end+horizons].values}
        }]

        try:
            quantiles_tensor_eisbach_mult, _ = pipeline.predict_quantiles(inputs_eisbach_mult, prediction_length=horizons, quantile_levels=quantiles)
            # Result is a list, we take index 0, variate 0
            q_preds_mult = quantiles_tensor_eisbach_mult[0][0].numpy().T
            mae_eisbach_mult = mean_absolute_error(target_eisbach, q_preds_mult[4, :])
            crps_eisbach_mult = crps(target_eisbach, q_preds_mult, np.array(quantiles))
        except Exception as e:
            print(f"Multivariate error for Eisbach: {e}")
            mae_eisbach_mult, crps_eisbach_mult = np.nan, np.nan

        inputs_isar_mult = [{
            "target": df_ffill['wassertemp_isar'].iloc[:train_end].values,
            "past_covariates": {"temperature": df_ffill['temperature'].iloc[:train_end].values},
            "future_covariates": {"temperature": df_ffill['temperature'].iloc[train_end:train_end+horizons].values}
        }]

        try:
            quantiles_tensor_isar_mult, _ = pipeline.predict_quantiles(inputs_isar_mult, prediction_length=horizons, quantile_levels=quantiles)
            q_preds_mult = quantiles_tensor_isar_mult[0][0].numpy().T
            mae_isar_mult = mean_absolute_error(target_isar, q_preds_mult[4, :])
            crps_isar_mult = crps(target_isar, q_preds_mult, np.array(quantiles))
        except Exception as e:
            print(f"Multivariate error for Isar: {e}")
            mae_isar_mult, crps_isar_mult = np.nan, np.nan

        results.append({'Horizon': horizons, 'River': 'Eisbach', 'Type': 'Multivariate', 'MAE': mae_eisbach_mult, 'CRPS': crps_eisbach_mult})
        results.append({'Horizon': horizons, 'River': 'Isar', 'Type': 'Multivariate', 'MAE': mae_isar_mult, 'CRPS': crps_isar_mult})

    df_results = pd.DataFrame(results)
    df_results.to_csv('isar_eisbach_comparison/forecasting_results.csv', index=False)
    print("\nResults saved to isar_eisbach_comparison/forecasting_results.csv")
    print(df_results)

if __name__ == "__main__":
    run()
