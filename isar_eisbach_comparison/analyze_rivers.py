import pandas as pd
import numpy as np
import torch
import os
from chronos import BaseChronosPipeline
import matplotlib.pyplot as plt
import seaborn as sns

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

def evaluate_chronos(pipeline, context, target, prediction_length, num_samples=20):
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    try:
        quantiles_tensor, _ = pipeline.predict_quantiles(
            context_tensor,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        # For univariate, shape is (1, pred_len, num_quantiles)
        q_preds = quantiles_tensor[0].numpy().T
        median_pred = q_preds[4, :]
        mae = mean_absolute_error(target, median_pred)
        crps_val = crps(target, q_preds, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        return mae, crps_val
    except AttributeError:
        pass

def run():
    csv_path = "isar_eisbach_comparison/isar_eisbach_data.csv"
    if not os.path.exists(csv_path):
        print("Data not found.")
        return

    df_combined = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_ffill = df_combined.ffill().bfill()

    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    results = []
    for horizons in [24, 96]:
        print(f"\nEvaluating Horizon: {horizons}h")
        train_end = len(df_ffill) - horizons
        if train_end <= 0: continue

        context_eisbach = df_ffill['wassertemp_eisbach'].iloc[:train_end].values
        target_eisbach = df_ffill['wassertemp_eisbach'].iloc[train_end:train_end+horizons].values
        mae_eisbach_uni, crps_eisbach_uni = evaluate_chronos(pipeline, context_eisbach, target_eisbach, horizons)

        context_isar = df_ffill['wassertemp_isar'].iloc[:train_end].values
        target_isar = df_ffill['wassertemp_isar'].iloc[train_end:train_end+horizons].values
        mae_isar_uni, crps_isar_uni = evaluate_chronos(pipeline, context_isar, target_isar, horizons)

        results.append({'Horizon': horizons, 'River': 'Eisbach', 'Type': 'Univariate', 'MAE': mae_eisbach_uni, 'CRPS': crps_eisbach_uni})
        results.append({'Horizon': horizons, 'River': 'Isar', 'Type': 'Univariate', 'MAE': mae_isar_uni, 'CRPS': crps_isar_uni})

        # Multivariate approach in chronos-2 by flattening covariates into the sequence, or we just leave as NaN because it's not supported natively for one target.
        # But wait, earlier the error was "operands could not be broadcast together with shapes (24,) (24,2)"
        # This is because the output is (2, 24, 9) representing two predictions! We just need to select index 0 for the target!
        context_eisbach_mult = [
            torch.tensor(df_ffill['wassertemp_eisbach'].iloc[:train_end].values, dtype=torch.float32),
            torch.tensor(df_ffill['temperature'].iloc[:train_end].values, dtype=torch.float32)
        ]

        try:
            quantiles_tensor, _ = pipeline.predict_quantiles(
                context_eisbach_mult,
                prediction_length=horizons,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
            # Two series predicted independently. quantiles_tensor is (2, pred_len, num_quantiles)
            q_preds_mult = quantiles_tensor[0].numpy().T
            mae_eisbach_mult = mean_absolute_error(target_eisbach, q_preds_mult[4, :])
            crps_eisbach_mult = crps(target_eisbach, q_preds_mult, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        except Exception as e:
            print(f"Multivariate error for Eisbach: {e}")
            mae_eisbach_mult, crps_eisbach_mult = np.nan, np.nan

        context_isar_mult = [
            torch.tensor(df_ffill['wassertemp_isar'].iloc[:train_end].values, dtype=torch.float32),
            torch.tensor(df_ffill['temperature'].iloc[:train_end].values, dtype=torch.float32)
        ]
        try:
            quantiles_tensor, _ = pipeline.predict_quantiles(
                context_isar_mult,
                prediction_length=horizons,
                quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
            q_preds_mult = quantiles_tensor[0].numpy().T
            mae_isar_mult = mean_absolute_error(target_isar, q_preds_mult[4, :])
            crps_isar_mult = crps(target_isar, q_preds_mult, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
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
