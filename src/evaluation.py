import pandas as pd
import numpy as np

def calculate_mean_pinball_loss(forecast_df: pd.DataFrame, actual_df: pd.DataFrame, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], target_col='wassertemp'):
    """
    Calculates the Mean Pinball Loss for a forecast against actual values.

    forecast_df: DataFrame containing quantile columns (e.g., '0.1', '0.5', '0.9') and 'timestamp'.
    actual_df: DataFrame containing the actual target values and 'timestamp'.
    quantiles: List of quantiles to evaluate.
    target_col: Name of the target column in actual_df.
    """
    # Ensure timestamps are datetime for merging
    if not pd.api.types.is_datetime64_any_dtype(forecast_df['timestamp']):
        forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
    if not pd.api.types.is_datetime64_any_dtype(actual_df['timestamp']):
        actual_df['timestamp'] = pd.to_datetime(actual_df['timestamp'])

    # Merge forecast with actuals
    merged = pd.merge(forecast_df, actual_df[['timestamp', target_col]], on='timestamp', how='inner')

    if merged.empty:
        return np.nan

    total_loss = 0
    valid_quantiles = 0

    for q in quantiles:
        q_str = str(q)
        if q_str not in merged.columns:
            # Try float to string formatting just in case "0.1" vs "0.10" issues
            # But usually AutoGluon uses "0.1" etc.
            continue

        y_true = merged[target_col].values
        y_pred = merged[q_str].values

        # Pinball loss formula: (y - y_hat) * q if y >= y_hat else (y_hat - y) * (1 - q)
        # diff = y - y_hat
        # loss = max(q * diff, (q - 1) * diff)

        diff = y_true - y_pred
        loss = np.maximum(q * diff, (q - 1) * diff)

        total_loss += np.mean(loss)
        valid_quantiles += 1

    if valid_quantiles == 0:
        return np.nan

    return total_loss / valid_quantiles
