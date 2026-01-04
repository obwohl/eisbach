import pandas as pd
from src.analysis import get_long_historical_data, find_high_variance_windows
from src.inference import run_chronos_inference
from src.evaluation import calculate_mean_pinball_loss
from src.plotting_experiment import plot_experiment_window
import logging

def run_experiment():
    # 1. Fetch Data
    # 3 years history
    full_data = get_long_historical_data(days_back=1095)

    if full_data.empty:
        print("Experiment Aborted: No data.")
        return

    # 2. Find Windows
    windows, df_sorted = find_high_variance_windows(full_data, window_size=96, top_n=10)

    if not windows:
        print("No valid windows found.")
        return

    print(f"--- Starting Experiment on {len(windows)} Windows ---")

    results = []

    for i, window in enumerate(windows):
        print(f"\nProcessing Window {i+1}/{len(windows)}: {window['start_time']} (Var: {window['variance']:.2f})")

        cutoff_idx = window['start_idx']
        end_idx = window['end_idx']

        # Context start
        context_start_idx = max(0, cutoff_idx - (24 * 60))

        # Slice for the "Run"
        experiment_slice = df_sorted.iloc[context_start_idx : end_idx].copy()

        rel_cutoff = cutoff_idx - context_start_idx

        # Store Truth
        truth_window = experiment_slice.iloc[rel_cutoff:].copy()

        # Fix TZ for truth_window to match prediction (which will be naive)
        if pd.api.types.is_datetime64tz_dtype(truth_window['timestamp']):
            truth_window['timestamp'] = truth_window['timestamp'].dt.tz_localize(None)

        # Mask Target
        experiment_slice.iloc[rel_cutoff:, experiment_slice.columns.get_loc('wassertemp')] = None

        # 4. Run Inference (Model A: Covariates)
        print("  Running Covariate Model...")
        pred_cov, _, _ = run_chronos_inference(experiment_slice, prediction_length=96, num_test_windows=0, use_covariates=True)

        # 5. Run Inference (Model B: Naive)
        print("  Running Naive Model...")
        pred_naive, _, _ = run_chronos_inference(experiment_slice, prediction_length=96, num_test_windows=0, use_covariates=False)

        # 6. Evaluation
        loss_cov = calculate_mean_pinball_loss(pred_cov, truth_window)
        loss_naive = calculate_mean_pinball_loss(pred_naive, truth_window)

        print(f"  -> Loss Cov: {loss_cov:.4f} | Loss Naive: {loss_naive:.4f}")

        # 7. Plotting
        # Pass history context (last 48h before cutoff)
        history_context = df_sorted.iloc[max(0, cutoff_idx - 48) : cutoff_idx].copy()
        # Fix history TZ for plotting consistency if needed (matplotlib handles it, but mixing might be bad)
        # Actually plot_experiment_window handles timestamps.
        # But predictions are Naive. Truth is now Naive. History is likely TZ-aware.
        # Let's make history naive too for plotting alignment.
        if pd.api.types.is_datetime64tz_dtype(history_context['timestamp']):
            history_context['timestamp'] = history_context['timestamp'].dt.tz_localize(None)

        output_filename = f"variance_experiment_window_{i+1}.png"
        plot_experiment_window(history_context, truth_window,
                               pred_cov, pred_naive,
                               loss_cov, loss_naive,
                               window, output_filename)

        results.append({
            'window': i+1,
            'start': window['start_time'],
            'variance': window['variance'],
            'loss_cov': loss_cov,
            'loss_naive': loss_naive
        })

    # Summary
    print("\n--- Experiment Summary ---")
    res_df = pd.DataFrame(results)
    print(res_df)
    res_df.to_csv("experiment_results.csv", index=False)

if __name__ == "__main__":
    run_experiment()
