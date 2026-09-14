"""Atomic, provenance-bound checkpoints for exp9 and exp14."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def bind(path: Path, df: pd.DataFrame, anchors, context: int, variants) -> None:
    """Refuse stale results after data, windows, backend or variant changes."""
    from importlib.metadata import PackageNotFoundError, version

    def installed(package: str) -> str:
        # MLX exists only on Apple silicon. Recording its absence keeps a Linux run from
        # crashing here, and still binds the result to a different environment than a
        # MacBook run, which is the point of the manifest.
        try:
            return version(package)
        except PackageNotFoundError:
            return "absent"

    spec = {
        'data': hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest(),
        'columns': list(df.columns), 'anchors': [str(t) for t in anchors],
        'context': context, 'variants': variants,
        'backend': os.environ.get('TIMESFM_BACKEND', 'torch'),
        'timesfm': installed('timesfm'), 'mlx': installed('mlx'),
    }
    encoded = json.dumps(spec, sort_keys=True, indent=2)
    manifest = path.with_suffix('.json')
    if manifest.exists():
        if manifest.read_text() != encoded:
            raise ValueError(f'{path}: checkpoint provenance changed; use a new cache path')
    elif path.exists():
        raise ValueError(f'{path}: checkpoint has no provenance')
    else:
        manifest.write_text(encoded)


def resume(path: Path, wanted: list[str], anchors) -> list[dict]:
    if not path.exists():
        return []
    scores = pd.read_csv(path, parse_dates=['reference_time'])
    expected = {(pd.Timestamp(t), lo) for t in anchors for lo in (0, 6, 12, 18, 24, 48, 72)}
    for label, group in scores.groupby('label'):
        keys = list(zip(group.reference_time, group.lead_lo, strict=True))
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f'{path}: incomplete or duplicate windows for {label}')
        if not np.isfinite(group[['mae', 'crps', 'rmse']].to_numpy()).all():
            raise ValueError(f'{path}: nonfinite scores for {label}')
    return scores[scores.label.isin(wanted)].to_dict('records')


def save(path: Path, rows: list[dict]) -> None:
    scores = pd.DataFrame(rows)
    if not np.isfinite(scores[['mae', 'crps', 'rmse']].to_numpy()).all():
        raise ValueError('Refusing to checkpoint nonfinite scores')
    tmp = path.with_suffix('.tmp')
    scores.to_csv(tmp, index=False)
    tmp.replace(path)
