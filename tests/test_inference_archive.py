import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import os
import shutil

from src.inference import save_forecast_to_archive, load_forecast_from_archive

@pytest.fixture
def temp_archive_path(tmp_path):
    archive_dir = tmp_path / "forecast_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "test_archive.csv"
    yield str(archive_path)

@pytest.fixture
def sample_forecast_df():
    # Create a simple forecast dataframe
    base_time = pd.Timestamp("2023-10-24 12:00:00", tz=timezone.utc)
    times = [base_time + pd.Timedelta(hours=i) for i in range(1, 5)]
    data = {
        'target_time': times,
        'wassertemp_q0.5': [15.0, 15.1, 15.2, 15.3],
        'wassertemp_q0.1': [14.0, 14.1, 14.2, 14.3],
        'wassertemp_q0.9': [16.0, 16.1, 16.2, 16.3]
    }
    df = pd.DataFrame(data).set_index('target_time')
    return df, base_time

def test_save_and_load_archive(temp_archive_path, sample_forecast_df):
    df_forecast, reference_time = sample_forecast_df

    # Save to archive
    save_forecast_to_archive(df_forecast, reference_time, archive_path=temp_archive_path)

    assert os.path.exists(temp_archive_path)

    # Load from archive (exact match)
    df_loaded = load_forecast_from_archive(reference_time, archive_path=temp_archive_path)

    assert df_loaded is not None
    # Index should be target_time and columns should match (excluding reference_time which is dropped on load)
    assert df_loaded.index.name == 'target_time'
    assert 'wassertemp_q0.5' in df_loaded.columns
    # Check length
    assert len(df_loaded) == 4

    # Assert values are identical
    pd.testing.assert_frame_equal(df_loaded, df_forecast)

def test_hybrid_logic_tolerance(temp_archive_path, sample_forecast_df):
    df_forecast, reference_time = sample_forecast_df

    # Save the base forecast
    save_forecast_to_archive(df_forecast, reference_time, archive_path=temp_archive_path)

    # 1. Test finding within tolerance (e.g. asking for 1 hour later)
    query_time_within = reference_time + pd.Timedelta(hours=1)
    df_loaded = load_forecast_from_archive(query_time_within, archive_path=temp_archive_path, tolerance_hours=2)
    assert df_loaded is not None

    # 2. Test NOT finding outside tolerance (e.g. asking for 3 hours later)
    query_time_outside = reference_time + pd.Timedelta(hours=3)
    df_not_loaded = load_forecast_from_archive(query_time_outside, archive_path=temp_archive_path, tolerance_hours=2)
    assert df_not_loaded is None

def test_save_deduplication(temp_archive_path, sample_forecast_df):
    df_forecast, reference_time = sample_forecast_df

    # Save twice for the exact same reference_time
    save_forecast_to_archive(df_forecast, reference_time, archive_path=temp_archive_path)
    save_forecast_to_archive(df_forecast, reference_time, archive_path=temp_archive_path)

    # Load the actual CSV to check length manually
    df_archive = pd.read_csv(temp_archive_path)

    # It should only have 4 rows (the ones we inserted once), not 8
    assert len(df_archive) == 4
