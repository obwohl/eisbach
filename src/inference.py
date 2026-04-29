import pandas as pd
import subprocess
import os

def run_inference(df_long):
    # Ensure the 'date' column is in datetime format before we begin
    df_long['date'] = pd.to_datetime(df_long['date'])
    last_timestamp = df_long['date'].max()

    print("--- Starting Forecast and Backtests ---")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # --- Main Forecast ---
    df_long.to_csv('data/df_long.csv', index=False)
    subprocess.run([
        'python', 'ts_proba_cuda/run_single_forecast.py',
        '--checkpoint', 'ts_proba_cuda/checkpoints/best_model.pt',
        '--data-file', 'data/df_long.csv',
        '--output-csv', 'data/inference.csv'
    ], check=True)

    # --- Prepare and run -96h backtest ---
    print("\n[1/3] Preparing and running -96h backtest...")
    backtest_96_end_date = last_timestamp - pd.Timedelta(hours=96)
    df_long_backtest_96_corrected = df_long[df_long['date'] <= backtest_96_end_date]
    df_long_backtest_96_corrected.to_csv('data/df_long_backtest_96_corrected.csv', index=False)
    subprocess.run([
        'python', 'ts_proba_cuda/run_single_forecast.py',
        '--checkpoint', 'ts_proba_cuda/checkpoints/best_model.pt',
        '--data-file', 'data/df_long_backtest_96_corrected.csv',
        '--output-csv', 'data/inference_backtest_96_corrected.csv'
    ], check=True)

    # --- Prepare and run -192h backtest ---
    print("\n[2/3] Preparing and running -192h backtest...")
    backtest_192_end_date = last_timestamp - pd.Timedelta(hours=192)
    df_long_backtest_192_corrected = df_long[df_long['date'] <= backtest_192_end_date]
    df_long_backtest_192_corrected.to_csv('data/df_long_backtest_192_corrected.csv', index=False)
    subprocess.run([
        'python', 'ts_proba_cuda/run_single_forecast.py',
        '--checkpoint', 'ts_proba_cuda/checkpoints/best_model.pt',
        '--data-file', 'data/df_long_backtest_192_corrected.csv',
        '--output-csv', 'data/inference_backtest_192_corrected.csv'
    ], check=True)

    # --- Prepare and run -288h backtest ---
    print("\n[3/3] Preparing and running -288h backtest...")
    backtest_288_end_date = last_timestamp - pd.Timedelta(hours=288)
    df_long_backtest_288_corrected = df_long[df_long['date'] <= backtest_288_end_date]
    df_long_backtest_288_corrected.to_csv('data/df_long_backtest_288_corrected.csv', index=False)
    subprocess.run([
        'python', 'ts_proba_cuda/run_single_forecast.py',
        '--checkpoint', 'ts_proba_cuda/checkpoints/best_model.pt',
        '--data-file', 'data/df_long_backtest_288_corrected.csv',
        '--output-csv', 'data/inference_backtest_288_corrected.csv'
    ], check=True)

    print("\n--- Backtests Complete. Loading data for plotting. ---")

    # --- Load All Forecast Data ---
    df_inference = pd.read_csv('data/inference.csv', parse_dates=[0], index_col=0)
    df_inference_backtest_96_corr = pd.read_csv('data/inference_backtest_96_corrected.csv', parse_dates=[0], index_col=0)
    df_inference_backtest_192_corr = pd.read_csv('data/inference_backtest_192_corrected.csv', parse_dates=[0], index_col=0)
    df_inference_backtest_288_corr = pd.read_csv('data/inference_backtest_288_corrected.csv', parse_dates=[0], index_col=0)

    return df_inference, df_inference_backtest_96_corr, df_inference_backtest_192_corr, df_inference_backtest_288_corr
