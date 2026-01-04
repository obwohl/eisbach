from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd
import logging

def run_chronos_inference(data: pd.DataFrame, prediction_length=96, num_test_windows=3, use_covariates=True):
    """
    Runs Chronos-2 inference on the provided data with 3 backtest windows.
    prediction_length defaults to 96 (4 days).
    use_covariates: If True, uses 'lufttemperatur_c' and 'niederschlag_mm' as known future covariates.
                    If False, runs univariate forecasting (naive).
    """
    if data.empty:
        print("No data provided for inference.")
        return None, None, None

    # Prepare TimeSeriesDataFrame
    if 'item_id' not in data.columns:
        data['item_id'] = 'eisbach_temp'

    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Fix for AutoGluon: Remove timezone information
    if pd.api.types.is_datetime64tz_dtype(data['timestamp']):
        print("Removing timezone information from timestamp for AutoGluon compatibility.")
        data['timestamp'] = data['timestamp'].dt.tz_localize(None)

    # Ensure regularity for Chronos
    # We resample to hourly frequency to avoid "irregular frequency" errors.
    # We leave gaps as NaNs (user requested no interpolation/invention of data).
    # But we must have the rows.
    data = data.set_index('timestamp').resample('h').first().reset_index()
    # If item_id became NaN/dropped during resample (unlikely if grouping, but here we just resample), restore it.
    data['item_id'] = 'eisbach_temp'

    ts_data = TimeSeriesDataFrame.from_data_frame(
        data,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    # Define Known Covariates
    known_covariates_names = []
    if use_covariates:
        if 'lufttemperatur_c' in ts_data.columns:
            known_covariates_names.append('lufttemperatur_c')
        if 'niederschlag_mm' in ts_data.columns:
            known_covariates_names.append('niederschlag_mm')
        print(f"Using known covariates: {known_covariates_names}")
    else:
        print("Running without covariates (Naive Mode).")

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target="wassertemp",
        known_covariates_names=known_covariates_names,
        eval_metric="MASE",
        freq='h' # Explicitly set frequency
    )

    # Fit (Zero-Shot Chronos-2)
    predictor.fit(
        ts_data,
        presets="chronos2",
        time_limit=300
    )

    # Helper to prepare known covariates DataFrame for a given history
    def get_known_covariates_for_prediction(history_df, full_data_df):
        # Even if we don't have covariates, make_future_data_frame creates the index structure
        future_frame = predictor.make_future_data_frame(history_df)

        if not known_covariates_names:
            return future_frame

        future_frame_reset = future_frame.reset_index()
        full_data_reset = full_data_df.reset_index()

        cols_to_merge = ['item_id', 'timestamp'] + known_covariates_names
        # Merge to get actual values (weather forecast or historical weather)
        merged = pd.merge(
            future_frame_reset[['item_id', 'timestamp']],
            full_data_reset[cols_to_merge],
            on=['item_id', 'timestamp'],
            how='left'
        )

        # Fill missing values if any (simple ffill/bfill fallback)
        if merged[known_covariates_names].isnull().any().any():
            last_known = history_df.tail(1).reset_index()
            for col in known_covariates_names:
                merged[col] = merged[col].ffill()
                if merged[col].isnull().any():
                     val = last_known[col].values[0] if col in last_known else 0
                     merged[col] = merged[col].fillna(val)

        return TimeSeriesDataFrame.from_data_frame(
            merged,
            id_column='item_id',
            timestamp_column='timestamp'
        )

    # 1. Main Forecast (Future)
    # If num_test_windows is 0, we assume the caller wants a forecast starting from the END of the provided data
    # (masked or not).

    # Check if we have valid target data at all
    item_df = ts_data.loc['eisbach_temp']
    last_valid_target_idx = item_df['wassertemp'].last_valid_index()

    if last_valid_target_idx is None:
        print("Error: No valid target data found.")
        return None, None, None

    start_timestamp = ts_data.index.get_level_values('timestamp').min()

    # For main forecast, history is everything up to the last valid target (wassertemp)
    history_data = ts_data.slice_by_time(start_timestamp, last_valid_target_idx)

    # Covariates come from the full dataset (which includes future weather)
    future_covariates = get_known_covariates_for_prediction(history_data, ts_data)

    future_predictions = predictor.predict(history_data, known_covariates=future_covariates)

    if isinstance(future_predictions, TimeSeriesDataFrame):
         future_predictions = future_predictions.reset_index() # Return as standard DF

    # 2. Backtesting
    backtest_predictions = []

    if num_test_windows > 0:
        for i in range(1, num_test_windows + 1):
            offset = i * 96 # Fixed 96h steps
            cutoff_timestamp = last_valid_target_idx - pd.Timedelta(hours=offset)

            print(f"--- Running Backtest {i}: Cutoff {cutoff_timestamp} ---")

            history_slice = ts_data.slice_by_time(start_timestamp, cutoff_timestamp)
            covariates_slice = get_known_covariates_for_prediction(history_slice, ts_data)

            try:
                pred = predictor.predict(history_slice, known_covariates=covariates_slice)

                if isinstance(pred, TimeSeriesDataFrame):
                    pred = pred.reset_index()

                pred['start_timestamp'] = cutoff_timestamp
                pred['type'] = f'backtest_{offset}h'
                backtest_predictions.append(pred)

            except Exception as e:
                logging.exception(f"Error in backtest {i}: {e}")

    return future_predictions, backtest_predictions, predictor
