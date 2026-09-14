"""Regression checks for experiment-only legacy and availability guards."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_audit import field, legacy_reader, select  # noqa: E402

from eisbach import archive  # noqa: E402


def test_legacy_reader_expands_fallback_without_mutation(tmp_path):
    fetched = '2026-05-01T12:30:00Z'
    raw = pd.DataFrame({'timestamp': pd.date_range('2026-05-01 13:00Z', periods=4, freq='h'),
                        'archive_timestamp': [fetched] * 4, 'temperature': [1., 2., 3., 4.]})
    with patch.object(archive, '_read_partition', return_value=raw):
        with legacy_reader():
            snap, anchor = archive.load_weather_snapshot(pd.Timestamp('2026-05-01 13:00Z'), root=tmp_path)
            assert len(snap) == 4
            assert anchor == pd.Timestamp(fetched)
    assert 'reference_time' not in raw


def test_strict_selector_rejects_future_fetch_even_for_exact_anchor():
    idx = pd.date_range('2026-05-01 00:00Z', periods=8, freq='h')
    df = pd.DataFrame(1., index=idx, columns=['eisbach', 'airtemp', 't_catchment'])
    snap = pd.DataFrame({'timestamp': idx, 'temperature': np.ones(8),
                         'archive_timestamp': ['2026-05-02T00:00:00Z'] * 8})
    with patch.object(archive, 'load_weather_snapshot', side_effect=lambda ts: (snap, ts)):
        usable, audit = select(df, context=2, horizon=2)
    assert not usable
    assert audit.late.all()
    assert audit.air.all()


def test_production_rejects_late_older_anchor():
    ts = pd.Timestamp('2026-05-01 14:00Z')
    raw = pd.DataFrame({'timestamp': [ts], 'reference_time': [ts - pd.Timedelta(hours=2)],
                        'archive_timestamp': [ts + pd.Timedelta(hours=2)]})
    with patch.object(archive, '_read_partition', return_value=raw):
        assert archive.load_weather_snapshot(ts) is None


def test_coalesces_legacy_field_in_mixed_schema():
    idx = pd.date_range('2026-05-01', periods=2, tz='UTC', freq='h')
    raw = pd.DataFrame({'timestamp': idx, 'lufttemperatur_c': [np.nan, 2.],
                        'temperature': [1., np.nan]})
    np.testing.assert_array_equal(field(raw, idx, ('lufttemperatur_c', 'temperature')), [1., 2.])
    assert field(raw, idx.append(pd.DatetimeIndex([idx[-1] + pd.Timedelta(hours=1)])),
                 ('temperature',)) is None


def test_legacy_reader_preserves_both_iso_timestamp_formats(tmp_path):
    weather = tmp_path / 'weather'
    weather.mkdir()
    path = weather / '2026-05.csv'
    pd.DataFrame({'timestamp': ['2026-05-01 13:00:00+00:00', '2026-05-01T14:00:00+00:00'],
                  'archive_timestamp': ['2026-05-01T12:30:00Z'] * 2,
                  'temperature': [1., 2.]}).to_csv(path, index=False)
    before = path.read_bytes()
    with legacy_reader():
        snap, _ = archive.load_weather_snapshot(pd.Timestamp('2026-05-01 13:00Z'), root=tmp_path)
    assert snap.timestamp.notna().all()
    assert len(snap) == 2
    assert path.read_bytes() == before
