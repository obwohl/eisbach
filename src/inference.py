import pandas as pd
import subprocess
import os
import sys
import shutil

def save_forecast_to_archive(df_forecast, reference_time, archive_path='data/forecast_archive/water_temp_predictions_archive.csv'):
    """
    Saves the forecast to an archive CSV file. It prevents duplicates by overwriting
    existing forecasts with the same reference_time.
    """
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)

    # Create a copy to avoid modifying the original dataframe
    df_to_save = df_forecast.copy()

    # Ensure index is standard column if it is the target time
    if df_to_save.index.name != 'target_time':
        df_to_save.index.name = 'target_time'
    df_to_save = df_to_save.reset_index()

    df_to_save['reference_time'] = reference_time

    # Ensure order of columns: reference_time, target_time, then the rest
    cols = ['reference_time', 'target_time'] + [c for c in df_to_save.columns if c not in ('reference_time', 'target_time')]
    df_to_save = df_to_save[cols]

    if os.path.exists(archive_path):
        # We only need to check if reference_time already exists to decide if we need
        # a full read-modify-write or a fast append
        df_existing_refs = pd.read_csv(archive_path, usecols=['reference_time'])
        # Convert to strict UTC datetimes for robust comparison
        existing_refs_dt = pd.to_datetime(df_existing_refs['reference_time'], utc=True)
        ref_time_dt = pd.to_datetime(reference_time, utc=True)

        if (existing_refs_dt == ref_time_dt).any():
            # Duplicate found, must read fully, filter, and overwrite
            df_archive = pd.read_csv(archive_path, parse_dates=['reference_time', 'target_time'])
            df_archive_refs_dt = pd.to_datetime(df_archive['reference_time'], utc=True)
            df_archive = df_archive[df_archive_refs_dt != ref_time_dt]
            df_archive = pd.concat([df_archive, df_to_save], ignore_index=True)
            df_archive.to_csv(archive_path, index=False)
        else:
            # Fast append mode
            df_to_save.to_csv(archive_path, mode='a', header=False, index=False)
    else:
        # First time writing the file
        df_to_save.to_csv(archive_path, index=False)
    print(f"Archived forecast for reference_time {reference_time} to {archive_path}")

def load_forecast_from_archive(reference_time, archive_path='data/forecast_archive/water_temp_predictions_archive.csv', tolerance_hours=2):
    """
    Loads a forecast from the archive that was made closest to the requested reference_time.
    Returns None if no forecast is found within the specified tolerance.
    """
    if not os.path.exists(archive_path):
        return None

    df_archive = pd.read_csv(archive_path, parse_dates=['reference_time', 'target_time'])
    if df_archive.empty:
        return None

    # Find all unique reference times in the archive
    unique_refs = df_archive['reference_time'].unique()

    # Calculate absolute differences using vectorized operations and forcing UTC
    # to avoid TypeError from timezone-aware vs naive comparisons
    unique_refs_ts = pd.to_datetime(pd.Series(unique_refs), utc=True)
    reference_time_utc = pd.to_datetime(reference_time, utc=True)

    diffs = (unique_refs_ts - reference_time_utc).abs()
    min_diff_idx = diffs.argmin()
    min_diff = diffs.iloc[min_diff_idx]
    closest_ref = unique_refs[min_diff_idx]

    # Check if within tolerance
    if min_diff <= pd.Timedelta(hours=tolerance_hours):
        print(f"Found honest historical forecast in archive (ref_time: {closest_ref}, diff: {min_diff}).")
        # Filter the archive for this closest reference time
        df_forecast = df_archive[df_archive['reference_time'] == closest_ref].copy()

        # Prepare the dataframe to look like `df_inference` (target_time as index)
        df_forecast = df_forecast.set_index('target_time')
        # Drop the reference_time column as it's not needed for plotting
        df_forecast = df_forecast.drop(columns=['reference_time'])

        return df_forecast
    else:
        print(f"No honest historical forecast found in archive within {tolerance_hours}h tolerance (closest was {min_diff} away).")
        return None

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


    # --- Load Main Forecast Data ---
    df_inference = pd.read_csv('data/inference.csv', parse_dates=[0], index_col=0)

    # Save the main forecast to our archive using last_timestamp as the reference_time
    save_forecast_to_archive(df_inference, last_timestamp)

    # --- Prepare and run backtests ---
    backtest_offsets = [96, 192, 288]
    backtest_dfs = {}

    for i, offset in enumerate(backtest_offsets, 1):
        print(f"\n[{i}/{len(backtest_offsets)}] Preparing and running -{offset}h backtest...")
        backtest_end_date = last_timestamp - pd.Timedelta(hours=offset)

        output_csv = f'data/inference_backtest_{offset}_corrected.csv'

        # Check archive first
        archived_df = load_forecast_from_archive(backtest_end_date)
        if archived_df is not None:
            # Save the loaded archive to CSV to keep flow consistent
            archived_df.to_csv(output_csv)
            backtest_dfs[offset] = archived_df
            continue

        # Fallback to computing the backtest if not in archive
        # For backtests, we also truncate exactly at backtest_end_date.
        df_long_backtest = df_long[df_long['date'] <= backtest_end_date].copy()

        data_file = f'data/df_long_backtest_{offset}_corrected.csv'

        df_long_backtest.to_csv(data_file, index=False)
        subprocess.run([
            sys.executable, 'ts_proba_cuda/run_single_forecast.py',
            '--checkpoint', 'ts_proba_cuda/checkpoints/best_model.pt',
            '--data-file', data_file,
            '--output-csv', output_csv
        ], check=True)

        backtest_dfs[offset] = pd.read_csv(output_csv, parse_dates=[0], index_col=0)

    print("\n--- Backtests Complete. ---")

    # Copy the main inference CSV to the root directory with the timestamp
    main_csv_name = f"Prediction_{timestamp_str}.csv" if timestamp_str else "Prediction.csv"

    # User-facing CSV should be in local time (Europe/Berlin) and formatted cleanly
    df_inference_local = df_inference.copy()
    # The index is likely naive or UTC here. If naive, assume UTC.
    if df_inference_local.index.tzinfo is None:
        df_inference_local.index = df_inference_local.index.tz_localize('UTC')
    df_inference_local.index = df_inference_local.index.tz_convert('Europe/Berlin')

    # Format the index as string so it looks normal (e.g. "2026-05-02 12:00") without the offset
    df_inference_local.index = df_inference_local.index.strftime('%Y-%m-%d %H:%M')

    df_inference_local.to_csv(main_csv_name)
    print(f"Saved main readable prediction data to {main_csv_name}")

    return df_inference, backtest_dfs[96], backtest_dfs[192], backtest_dfs[288]
