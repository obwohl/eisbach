import pandas as pd
import numpy as np
import torch
import os
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

def evaluate_combination(pipeline, inputs_list, target_true, horizons, quantiles):
    try:
        quantiles_tensor, _ = pipeline.predict_quantiles(inputs_list, prediction_length=horizons, quantile_levels=quantiles)
        # Handle shape
        if isinstance(quantiles_tensor, list):
            q_preds_mult = quantiles_tensor[0].numpy().T
        else:
            q_preds_mult = quantiles_tensor[0, 0].numpy().T
        mae_val = mean_absolute_error(target_true, q_preds_mult[4, :])
        crps_val = crps(target_true, q_preds_mult, np.array(quantiles))
        return mae_val, crps_val
    except Exception as e:
        print(f"Error evaluating: {e}")
        return np.nan, np.nan

def run():
    csv_path = "isar_eisbach_comparison/isar_eisbach_10_years.csv"
    weather_csv = "isar_eisbach_comparison/weather_10_years.csv"
    if not os.path.exists(csv_path) or not os.path.exists(weather_csv):
        print("Data missing.")
        return

    df_combined = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_weather = pd.read_csv(weather_csv, index_col=0, parse_dates=True)
    df_ffill = pd.merge(df_combined, df_weather, left_index=True, right_index=True, how='inner').ffill().bfill()

    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    horizons_to_test = [24, 96]

    end_idx_1 = len(df_ffill) - max(horizons_to_test)
    end_idx_2 = end_idx_1 - 240 # -10 days
    end_idx_3 = end_idx_1 - 480 # -20 days
    windows = [end_idx_1, end_idx_2, end_idx_3]

    combinations = [
        "Univariate",
        "+ Luft (Past)",
        "+ Luft (Past+Future)",
        "+ Fluss (Past)",
        "+ Fluss+Luft (Past)",
        "+ Fluss+Luft (Past+Future)"
    ]

    results_dict = []

    for river in ['Eisbach', 'Isar']:
        target_col = 'wassertemp_eisbach' if river == 'Eisbach' else 'wassertemp_isar'
        other_col = 'wassertemp_isar' if river == 'Eisbach' else 'wassertemp_eisbach'

        for horizons in horizons_to_test:
            print(f"\n--- Evaluating {river} for {horizons}h ---")

            agg_mae = {c: [] for c in combinations}
            agg_crps = {c: [] for c in combinations}

            for w, train_end in enumerate(windows):
                if train_end <= 0: continue

                print(f"Window {w+1}/3...")

                target_true = df_ffill[target_col].iloc[train_end:train_end+horizons].values
                # We need to shorten history length for speed (Chronos handles arbitrary, but 512 is max standard context usually, or we just pass the last 1000)
                # To speed up and avoid 14+ min hangs from 87k context length:
                ctx_len = min(512, train_end)
                train_start = train_end - ctx_len

                target_hist = df_ffill[target_col].iloc[train_start:train_end].values

                luft_hist = df_ffill['temperature'].iloc[train_start:train_end].values
                luft_fut = df_ffill['temperature'].iloc[train_end:train_end+horizons].values

                other_hist = df_ffill[other_col].iloc[train_start:train_end].values

                # 1. Univariate
                mae, crps_val = evaluate_combination(pipeline, [{
                    "target": target_hist
                }], target_true, horizons, quantiles)
                agg_mae["Univariate"].append(mae); agg_crps["Univariate"].append(crps_val)

                # 2. + Luft (Past)
                mae, crps_val = evaluate_combination(pipeline, [{
                    "target": target_hist,
                    "past_covariates": {"luft": luft_hist}
                }], target_true, horizons, quantiles)
                agg_mae["+ Luft (Past)"].append(mae); agg_crps["+ Luft (Past)"].append(crps_val)

                # 3. + Luft (Past+Future)
                mae, crps_val = evaluate_combination(pipeline, [{
                    "target": target_hist,
                    "past_covariates": {"luft": luft_hist},
                    "future_covariates": {"luft": luft_fut}
                }], target_true, horizons, quantiles)
                agg_mae["+ Luft (Past+Future)"].append(mae); agg_crps["+ Luft (Past+Future)"].append(crps_val)

                # 4. + Fluss (Past)
                mae, crps_val = evaluate_combination(pipeline, [{
                    "target": target_hist,
                    "past_covariates": {"fluss": other_hist}
                }], target_true, horizons, quantiles)
                agg_mae["+ Fluss (Past)"].append(mae); agg_crps["+ Fluss (Past)"].append(crps_val)

                # 5. + Fluss+Luft (Past)
                mae, crps_val = evaluate_combination(pipeline, [{
                    "target": target_hist,
                    "past_covariates": {"fluss": other_hist, "luft": luft_hist}
                }], target_true, horizons, quantiles)
                agg_mae["+ Fluss+Luft (Past)"].append(mae); agg_crps["+ Fluss+Luft (Past)"].append(crps_val)

                # 6. + Fluss+Luft (Past+Future)
                mae, crps_val = evaluate_combination(pipeline, [{
                    "target": target_hist,
                    "past_covariates": {"fluss": other_hist, "luft": luft_hist},
                    "future_covariates": {"luft": luft_fut}
                }], target_true, horizons, quantiles)
                agg_mae["+ Fluss+Luft (Past+Future)"].append(mae); agg_crps["+ Fluss+Luft (Past+Future)"].append(crps_val)

            # Average results
            for c in combinations:
                results_dict.append({
                    'River': river,
                    'Horizon': horizons,
                    'Configuration': c,
                    'MAE (Mean)': np.nanmean(agg_mae[c]),
                    'CRPS (Mean)': np.nanmean(agg_crps[c])
                })

    df_results = pd.DataFrame(results_dict)
    df_results.to_csv('isar_eisbach_comparison/forecasting_combinations.csv', index=False)
    print("\nResults saved to isar_eisbach_comparison/forecasting_combinations.csv")
    print(df_results)

if __name__ == "__main__":
    run()
