from src.data import get_eisbach_data
from src.inference import run_chronos_inference
from src.plotting import save_static_plot
from src.upload import run_upload
from src.evaluation import calculate_mean_pinball_loss
import pandas as pd
import numpy as np

def main():
    print("--- 1. Fetching Data ---")
    data = get_eisbach_data()
    print(f"Data fetched: {len(data)} rows.")
    if data.empty:
        print("No data available. Exiting.")
        return

    # --- Model A: With Covariates ---
    print("\n--- 2a. Running Inference (WITH Covariates) ---")
    future_pred_cov, backtest_preds_cov, _ = run_chronos_inference(data, prediction_length=96, num_test_windows=3, use_covariates=True)

    # --- Model B: Naive (No Covariates) ---
    print("\n--- 2b. Running Inference (Naive / NO Covariates) ---")
    future_pred_naive, backtest_preds_naive, _ = run_chronos_inference(data, prediction_length=96, num_test_windows=3, use_covariates=False)

    if future_pred_cov is None or future_pred_naive is None:
        print("Inference failed.")
        return

    # --- Evaluation ---
    print("\n--- 3. Evaluation (Mean Pinball Loss) ---")

    def evaluate_backtests(backtests, label):
        scores = []
        for i, bt_df in enumerate(backtests):
            loss = calculate_mean_pinball_loss(bt_df, data)
            scores.append(loss)
            print(f"  {label} Backtest {i+1}: {loss:.4f}")
        avg_score = np.nanmean(scores)
        print(f"  -> {label} Average Loss: {avg_score:.4f}")
        return avg_score

    avg_loss_cov = evaluate_backtests(backtest_preds_cov, "Covariates")
    avg_loss_naive = evaluate_backtests(backtest_preds_naive, "Naive")

    print(f"\nSummary:")
    print(f"  Covariates Avg Loss: {avg_loss_cov:.4f}")
    print(f"  Naive Avg Loss:      {avg_loss_naive:.4f}")

    if avg_loss_cov < avg_loss_naive:
        print("  -> Covariate model performed better.")
    else:
        print("  -> Naive model performed better.")

    print("\n--- 4. Saving Predictions ---")
    # Save main prediction to CSV (Covariate model is primary)
    future_pred_cov.to_csv("eisbach_predictions.csv")
    print("Predictions saved to eisbach_predictions.csv")

    print("\n--- 5. Generating Plot ---")
    # Note: save_static_plot doesn't take loss values in arguments directly to display in title,
    # but the user requested loss display for *Experiment* plots primarily.
    # For the main plot, showing 3 backtests with losses is crowded.
    # The requirement "im Fenster auch gleich der Pinball-Loss angezeigt wird" likely referred to the experiment windows
    # ("Und dann... von all diesen zehn Fenstern... möchte ich...").
    # But for completeness, we can print it to console (done above).

    save_static_plot(data,
                     future_pred_cov, backtest_preds_cov,
                     future_pred_naive, backtest_preds_naive,
                     output_filename="eisbach_new.png")

    print("\n--- 6. Uploading Results ---")
    run_upload()

    print("\n--- Done ---")

if __name__ == "__main__":
    main()
