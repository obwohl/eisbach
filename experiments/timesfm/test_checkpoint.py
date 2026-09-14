"""Checkpoint integrity is essential when runs take hours."""
import numpy as np
import pandas as pd
import pytest
from checkpoint import resume, save


def records():
    return [{'label': 'candidate', 'reference_time': pd.Timestamp('2025-01-01', tz='UTC'),
             'lead_lo': lo, 'mae': 1.0, 'crps': 0.5, 'rmse': 1.1}
            for lo in (0, 6, 12, 18, 24, 48, 72)]


def test_resume_keeps_last_complete_variant(tmp_path):
    path = tmp_path / 'scores.csv'
    rows = records()
    save(path, rows)
    assert resume(path, ['candidate'], [rows[0]['reference_time']]) == rows


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'nan'])
def test_resume_rejects_corruption(tmp_path, corruption):
    rows = records()
    if corruption == 'missing':
        rows.pop()
    elif corruption == 'duplicate':
        rows[-1] = rows[0].copy()
    else:
        rows[0]['mae'] = np.nan
    path = tmp_path / 'scores.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError):
        resume(path, ['candidate'], [rows[0]['reference_time']])


def test_bad_save_preserves_previous_checkpoint(tmp_path):
    path = tmp_path / 'scores.csv'
    rows = records()
    save(path, rows)
    before = path.read_bytes()
    rows[0]['crps'] = np.nan
    with pytest.raises(ValueError):
        save(path, rows)
    assert path.read_bytes() == before
