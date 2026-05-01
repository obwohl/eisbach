import pandas as pd
import pytest
import os

# Data path configurations
DATA_DIR = 'data'
INF_CSV = os.path.join(DATA_DIR, 'inference.csv')
BT_96 = os.path.join(DATA_DIR, 'inference_backtest_96_corrected.csv')
BT_192 = os.path.join(DATA_DIR, 'inference_backtest_192_corrected.csv')
BT_288 = os.path.join(DATA_DIR, 'inference_backtest_288_corrected.csv')
DF_LONG = os.path.join(DATA_DIR, 'df_long.csv')

def load_data():
    df_inf = pd.read_csv(INF_CSV, parse_dates=[0], index_col=0)
    df_bt_96 = pd.read_csv(BT_96, parse_dates=[0], index_col=0)
    df_bt_192 = pd.read_csv(BT_192, parse_dates=[0], index_col=0)
    df_bt_288 = pd.read_csv(BT_288, parse_dates=[0], index_col=0)
    df_long = pd.read_csv(DF_LONG, parse_dates=['date'])
    return df_inf, df_bt_96, df_bt_192, df_bt_288, df_long

@pytest.fixture(scope="module")
def data():
    # Make sure we have the files before testing
    assert os.path.exists(INF_CSV), f"Missing {INF_CSV}, please run the pipeline first."
    return load_data()

def check_plausibility(df, col_pattern, min_val, max_val):
    cols = [c for c in df.columns if col_pattern in c]
    for col in cols:
        assert df[col].min() >= min_val, f"Value in {col} is below {min_val}"
        assert df[col].max() <= max_val, f"Value in {col} is above {max_val}"

def test_backtest_windows(data):
    df_inf, df_bt_96, df_bt_192, df_bt_288, df_long = data

    backtests = [df_bt_96, df_bt_192, df_bt_288]

    for idx, df_bt in enumerate(backtests):
        # 1. Check existence of forecasts (water temperature) and air temperature
        assert 'wassertemp_q0.5' in df_bt.columns, "wassertemp median forecast missing"
        assert 'wassertemp_q0.01' in df_bt.columns, "wassertemp q0.01 forecast missing"
        assert 'wassertemp_q0.99' in df_bt.columns, "wassertemp q0.99 forecast missing"
        assert 'airtemp_96_q0.5' in df_bt.columns, "airtemp forecast missing"

        # Ensure no NaNs in important forecast columns
        assert not df_bt['wassertemp_q0.5'].isna().any(), "NaNs found in wassertemp forecast"
        assert not df_bt['airtemp_96_q0.5'].isna().any(), "NaNs found in airtemp forecast"

        # 2. Plausibility checks
        # Air temp between -5 and 30
        check_plausibility(df_bt, 'airtemp_96', -5.0, 45.0) # Using 45 to be generous with summer/quantiles
        # Water temp prediction between -10 and 30
        check_plausibility(df_bt, 'wassertemp', -10.0, 35.0)

        # 3. Statistical calibration check (approx 80% inside the 1%-99% interval)
        actual_water = df_long[(df_long['cols'] == 'wassertemp')].set_index('date')

        # Intersect actuals with the backtest window
        overlap = df_bt.index.intersection(actual_water.index)
        assert len(overlap) > 0, "No overlapping actual data found for backtest validation"

        actual_vals = actual_water.loc[overlap, 'data']
        lower_bound = df_bt.loc[overlap, 'wassertemp_q0.01']
        upper_bound = df_bt.loc[overlap, 'wassertemp_q0.99']

        # Calculate how many fall inside the bounds
        inside = ((actual_vals >= lower_bound) & (actual_vals <= upper_bound)).sum()
        total = len(overlap)
        percentage_inside = inside / total

        # The user requested that we check if at least ~80% of data is within the 1-99% bounds
        assert percentage_inside >= 0.80, f"Statistical check failed for backtest {idx+1}: Only {percentage_inside*100:.1f}% inside 1-99 bounds."

def test_future_forecast_window(data):
    df_inf, _, _, _, df_long = data

    # 1. Existence of predictions and airtemp
    assert 'wassertemp_q0.5' in df_inf.columns, "wassertemp median forecast missing"
    assert 'airtemp_96_q0.5' in df_inf.columns, "airtemp forecast missing"

    assert not df_inf['wassertemp_q0.5'].isna().any(), "NaNs found in future wassertemp forecast"
    assert not df_inf['airtemp_96_q0.5'].isna().any(), "NaNs found in future airtemp forecast"

    # 2. Plausibility bounds (generous: -10 to 45)
    check_plausibility(df_inf, 'wassertemp', -10.0, 35.0)
    check_plausibility(df_inf, 'airtemp_96', -5.0, 45.0)

    # 3. Check that actual water temperature does NOT exist for this future horizon
    actual_water = df_long[(df_long['cols'] == 'wassertemp')].set_index('date')
    overlap = df_inf.index.intersection(actual_water.index)

    if len(overlap) > 0:
        # If there's an overlap in dates, ensure the actual values are NaN in the future
        actual_vals_in_future = actual_water.loc[overlap, 'data']
        assert actual_vals_in_future.isna().all(), "Actual water temperatures found in the future forecast window! Data leakage."
