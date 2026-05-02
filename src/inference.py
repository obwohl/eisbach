import pandas as pd
import subprocess
import os
import sys
import shutil

def run_inference(df_long, timestamp_str=""):
    # Ensure the 'date' column is in datetime format before we begin
    df_long['date'] = pd.to_datetime(df_long['date'])

    # Calculate the last timestamp based on the actual target data, not the future covariates
    df_wassertemp = df_long[df_long['cols'] == 'wassertemp'].dropna(subset=['data'])
    last_timestamp = df_wassertemp['date'].max()

    print(f"--- Starting Forecast and Backtests (anchored at {last_timestamp}) ---")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # --- Main Forecast ---
    # We truncate df_long EXACTLY at last_timestamp.
    # The covariates (airtemp_96, pressure_96) are ALREADY shifted by -96 hours.
    # This means the covariate values AT `last_timestamp` are actually the future weather at `last_timestamp + 96h`.
    # Therefore, the model has all information required to forecast the next 96 hours just from the data up to `last_timestamp`.
    df_long_main = df_long[df_long['date'] <= last_timestamp].copy()
    df_long_main.to_csv('data/df_long.csv', index=False)
    subprocess.run([
        sys.executable, 'ts_proba_cuda/run_single_forecast.py',
        '--checkpoint', 'ts_proba_cuda/checkpoints/best_model.pt',
        '--data-file', 'data/df_long.csv',
        '--output-csv', 'data/inference.csv'
    ], check=True)

    # --- Prepare and run backtests ---
    backtest_offsets = [96, 192, 288]
    for i, offset in enumerate(backtest_offsets, 1):
        print(f"\n[{i}/{len(backtest_offsets)}] Preparing and running -{offset}h backtest...")
        backtest_end_date = last_timestamp - pd.Timedelta(hours=offset)

        # For backtests, we also truncate exactly at backtest_end_date.
        df_long_backtest = df_long[df_long['date'] <= backtest_end_date].copy()

        data_file = f'data/df_long_backtest_{offset}_corrected.csv'
        output_csv = f'data/inference_backtest_{offset}_corrected.csv'

        df_long_backtest.to_csv(data_file, index=False)
        subprocess.run([
            sys.executable, 'ts_proba_cuda/run_single_forecast.py',
            '--checkpoint', 'ts_proba_cuda/checkpoints/best_model.pt',
            '--data-file', data_file,
            '--output-csv', output_csv
        ], check=True)

    print("\n--- Backtests Complete. Loading data for plotting. ---")

    # --- Load All Forecast Data ---
    df_inference = pd.read_csv('data/inference.csv', parse_dates=[0], index_col=0)
    df_inference_backtest_96_corr = pd.read_csv('data/inference_backtest_96_corrected.csv', parse_dates=[0], index_col=0)
    df_inference_backtest_192_corr = pd.read_csv('data/inference_backtest_192_corrected.csv', parse_dates=[0], index_col=0)
    df_inference_backtest_288_corr = pd.read_csv('data/inference_backtest_288_corrected.csv', parse_dates=[0], index_col=0)

    # Copy the main inference CSV to the root directory with the timestamp
    main_csv_name = f"Prediction_{timestamp_str}.csv" if timestamp_str else "Prediction.csv"
    shutil.copy('data/inference.csv', main_csv_name)
    print(f"Saved main readable prediction data to {main_csv_name}")

    return df_inference, df_inference_backtest_96_corr, df_inference_backtest_192_corr, df_inference_backtest_288_corr
