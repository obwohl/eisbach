from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd
import logging

def run_chronos_inference(data: pd.DataFrame, prediction_length=64, num_test_windows=3):
    """
    Runs Chronos-2 inference on the provided data.
    """
    if data.empty:
        print("No data provided for inference.")
        return None, None, None

    # Prepare TimeSeriesDataFrame
    # We need 'item_id' and 'timestamp'
    if 'item_id' not in data.columns:
        data['item_id'] = 'eisbach_temp'

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Fix for AutoGluon: Remove timezone information (make it naive)
    if pd.api.types.is_datetime64tz_dtype(data['timestamp']):
        print("Removing timezone information from timestamp for AutoGluon compatibility.")
        data['timestamp'] = data['timestamp'].dt.tz_localize(None)

    ts_data = TimeSeriesDataFrame.from_data_frame(
        data,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    # Define Predictor
    known_covariates_names = []
    if 'lufttemperatur_c' in ts_data.columns:
        known_covariates_names.append('lufttemperatur_c')
    if 'niederschlag_mm' in ts_data.columns:
        known_covariates_names.append('niederschlag_mm')
    if 'pressure' in ts_data.columns:
        known_covariates_names.append('pressure')

    print(f"Using known covariates: {known_covariates_names}")

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target="wassertemp",
        known_covariates_names=known_covariates_names,
        eval_metric="MASE"
    )

    # "Fit" (setup) the predictor
    predictor.fit(
        ts_data,
        presets="chronos2",
        time_limit=300
    )

    # 1. Main Forecast (Future)

    # Get the series for the item
    item_df = ts_data.loc['eisbach_temp']
    last_valid_target_idx = item_df['wassertemp'].last_valid_index()

    if last_valid_target_idx is None:
        print("Error: No valid target data found.")
        return None, None, None

    print(f"Last valid target timestamp: {last_valid_target_idx}")

    start_timestamp = ts_data.index.get_level_values('timestamp').min()
    history_data = ts_data.slice_by_time(start_timestamp, last_valid_target_idx)

    # Use make_future_data_frame to get the exact expected structure for known_covariates
    # make_future_data_frame(data, periods=prediction_length)
    # It does not take known_covariates_names. It just extends the index.
    future_covariates = predictor.make_future_data_frame(history_data)

    # Now we need to fill this dataframe with actual values from ts_data
    # ts_data has the values (merged weather data)

    # We can use update or merge.
    # Reset index for easier merging
    future_cov_df = future_covariates.reset_index()
    ts_data_df = ts_data.reset_index()

    # We only want the relevant columns
    cols_to_merge = ['item_id', 'timestamp'] + known_covariates_names
    source_df = ts_data_df[cols_to_merge]

    # Merge
    # We want to keep future_cov_df rows (the expected structure) and fill values from source_df
    # Note: future_covariates from make_future_data_frame contains columns for known covariates but they are filled with NaNs?
    # Or just index? It seems it returns a dataframe with index extended.
    # Let's check columns.

    # Actually, make_future_data_frame creates a DataFrame with future timestamps and valid item_ids.
    # It doesn't automatically pull values.

    merged = pd.merge(future_cov_df[['item_id', 'timestamp']], source_df, on=['item_id', 'timestamp'], how='left')

    # Check for missing values and fill
    if merged[known_covariates_names].isnull().any().any():
        print("Warning: Missing values in future known covariates. Filling with ffill and then backfill from history.")
        # Fill from history (last known value)
        last_known = history_data.tail(1).reset_index()

        for col in known_covariates_names:
            merged[col] = merged[col].ffill()
            if merged[col].isnull().any():
                val = last_known[col].values[0]
                merged[col] = merged[col].fillna(val)

    # Re-create TimeSeriesDataFrame
    future_covariates = TimeSeriesDataFrame.from_data_frame(
        merged,
        id_column='item_id',
        timestamp_column='timestamp'
    )

    print(f"Predicting with constructed known covariates.")

    predictions = predictor.predict(history_data, known_covariates=future_covariates)

    # 2. Backtesting (Historical Forecasts)
    backtest_predictions = []

    for i in range(1, num_test_windows + 1):
        offset = i * prediction_length
        cutoff_timestamp = last_valid_target_idx - pd.Timedelta(hours=offset)

        print(f"Running backtest {i}: Cutoff {cutoff_timestamp}")

        history_slice = ts_data.slice_by_time(start_timestamp, cutoff_timestamp)

        # Prepare known covariates for this slice using make_future_data_frame
        covariates_slice = predictor.make_future_data_frame(history_slice)

        cov_df = covariates_slice.reset_index()
        # Merge with source data (ts_data contains all history including weather)
        merged_cov = pd.merge(cov_df[['item_id', 'timestamp']], ts_data_df[cols_to_merge], on=['item_id', 'timestamp'], how='left')

        # Fill missing
        if merged_cov[known_covariates_names].isnull().any().any():
             last_known = history_slice.tail(1).reset_index()
             for col in known_covariates_names:
                 merged_cov[col] = merged_cov[col].ffill()
                 if merged_cov[col].isnull().any():
                     val = last_known[col].values[0]
                     merged_cov[col] = merged_cov[col].fillna(val)

        covariates_slice = TimeSeriesDataFrame.from_data_frame(
            merged_cov,
            id_column='item_id',
            timestamp_column='timestamp'
        )

        try:
            pred = predictor.predict(history_slice, known_covariates=covariates_slice)

            if isinstance(pred, TimeSeriesDataFrame):
                pred = pred.reset_index()

            pred['start_timestamp'] = cutoff_timestamp
            pred['type'] = 'backtest'
            backtest_predictions.append(pred)
        except Exception as e:
            logging.exception(f"Error in backtest {i}: {e}")

    return predictions, backtest_predictions, predictor
