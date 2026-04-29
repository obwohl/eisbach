import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from src.data import prepare_data

def test_prepare_data_equivalence(mocker):
    # Mock date so time-based offsets are consistent
    mock_now_naive = datetime(2026, 4, 1, 12, 0)
    mock_now_aware = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc).astimezone()

    mocker.patch('src.data.datetime')
    import src.data as data_mod
    data_mod.datetime.now.return_value = mock_now_aware

    # Mock fetch_data_from_url to return a predefined dataframe (Naive)
    mock_wt = pd.DataFrame({
        'timestamp': pd.date_range(start=mock_now_naive - timedelta(days=40), end=mock_now_naive, freq='1h'),
        'wassertemp': np.random.rand(40*24 + 1) * 15 + 5
    })

    mocker.patch('src.data.fetch_data_from_url', return_value=mock_wt)

    # Mock get_prepared_weather_data (Aware in Europe/Berlin)
    mock_wetter = pd.DataFrame({
        'timestamp': pd.date_range(start=mock_now_naive - timedelta(days=40), end=mock_now_naive + timedelta(days=8), freq='1h').tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward'),
        'lufttemperatur_c': np.random.rand((40+8)*24 + 1) * 20,
        'niederschlag_mm': np.zeros((40+8)*24 + 1),
        'pressure': np.random.rand((40+8)*24 + 1) * 20 + 1000
    }).set_index('timestamp')
    mocker.patch('src.data.get_prepared_weather_data', return_value=mock_wetter)

    # Run the function
    df_long, df_wetter = prepare_data()

    # Equivalence assertions based on the original notebook structure
    # 1. Returned df_long should have columns: 'date', 'cols', 'data'
    assert list(df_long.columns) == ['date', 'cols', 'data']

    # 2. cols should be categorical with specific categories
    assert isinstance(df_long['cols'].dtype, pd.CategoricalDtype)
    assert list(df_long['cols'].cat.categories) == ['wassertemp', 'airtemp_96', 'pressure_96']

    # 3. Dates should be in UTC
    assert df_long['date'].dt.tz is not None
    assert str(df_long['date'].dt.tz) == 'UTC'

    # The NaN values are expected because we shift by -96, meaning the last 96 rows of airtemp_96 and pressure_96 will be NaN!
    # So we should just verify that wassertemp does not have NaNs in the overlapping region
    assert not df_long[df_long['cols'] == 'wassertemp']['data'].isna().any()
