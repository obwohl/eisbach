from src.data import get_eisbach_data
from src.inference import run_chronos_inference
from src.plotting import plot_forecasts
from src.upload import run_upload
import pandas as pd
def main():
    print("--- 1. Fetching Data ---")
    data = get_eisbach_data()
    print(f"Data fetched: {len(data)} rows.")
    if data.empty:
        print("No data available. Exiting.")
        return

    print("--- 2. Running Inference ---")
    future_pred, backtest_preds, _ = run_chronos_inference(data, prediction_length=64, num_test_windows=3)

    if future_pred is None:
        print("Inference failed.")
        return

    print("--- 3. Saving Predictions ---")
    # Save main prediction to CSV
    future_pred.to_csv("eisbach_predictions.csv")
    print("Predictions saved to eisbach_predictions.csv")

    print("--- 4. Generating Plot ---")
    # We need the full historical data for plotting, plus the predictions
    plot_forecasts(data, future_pred, backtest_preds)

    print("--- 5. Uploading Results ---")
    run_upload()

    print("--- Done ---")

if __name__ == "__main__":
    main()
