"""Read-only experiment adapter and audit of the production snapshot selector."""
from contextlib import contextmanager
from functools import lru_cache
from unittest.mock import patch

import numpy as np
import pandas as pd

from eisbach import archive


@contextmanager
def legacy_reader():
    """Make the documented legacy anchor fallback explicit on in-memory copies.

    Production's empty Series for a missing column does not expand on fillna;
    consequently old partitions otherwise return only row zero. No archive writes.
    """
    read = archive._read_partition

    @lru_cache(None)
    def cached(path):
        frame = read(path).copy()
        # Production infers a single timestamp format and coerces the other
        # valid ISO representation to NaT. Reparse original text, never rewrite it.
        if path.exists():
            text = pd.read_csv(path)
            for col in ('timestamp', 'reference_time'):
                if col in text:
                    frame[col] = pd.to_datetime(text[col], utc=True, errors='coerce', format='mixed')
        if not frame.empty and 'reference_time' not in frame:
            frame['reference_time'] = pd.to_datetime(
                frame.archive_timestamp, utc=True, format='mixed')
        return frame

    with patch.object(archive, '_read_partition', cached):
        yield


def field(snapshot, targets, candidates):
    s = snapshot.copy()
    s['timestamp'] = pd.to_datetime(s.timestamp, utc=True, errors='coerce', format='mixed')
    s = s.dropna(subset=['timestamp']).drop_duplicates('timestamp').set_index('timestamp')
    # Coalesce rowwise: a concatenated partition can contain both schema columns.
    out = pd.Series(np.nan, index=targets)
    for col in candidates:
        if col in s:
            out = out.fillna(pd.to_numeric(s[col], errors='coerce').reindex(targets))
    a = out.to_numpy(dtype=float)
    return a if np.isfinite(a).all() else None


def select(df, context, horizon):
    usable, audit = [], []
    idx = df.index
    with legacy_reader():
        for pos in range(context, len(idx) - horizon):
            ts = idx[pos]
            if ts < pd.Timestamp('2026-05-01', tz='UTC'):
                continue
            found = archive.load_weather_snapshot(ts)
            if found is None:
                continue
            snap, anchor = found
            fetched = pd.to_datetime(snap.archive_timestamp, utc=True, format='mixed').max()
            targets = idx[pos + 1:pos + 1 + horizon]
            air = field(snap, targets, ('lufttemperatur_c', 'temperature'))
            solar = field(snap, targets, ('solar',))
            complete = np.isfinite(df[['eisbach', 'airtemp', 't_catchment']]
                                   .reindex(targets).to_numpy()).all()
            late = fetched > ts
            audit.append(dict(reference_time=ts, anchor=anchor, fetched=fetched,
                              late=late, air=air is not None, solar=solar is not None,
                              complete=bool(complete)))
            if not late and air is not None and complete:
                assert anchor <= ts and fetched <= ts
                usable.append((ts, {'airtemp': air, '_solar': solar}))
    return usable, pd.DataFrame(audit)
