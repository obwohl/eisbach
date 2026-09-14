"""Independent origins for the unresolved Puppling increment beside Beuerberg."""
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import bench
import numpy as np
import pandas as pd
from checkpoint import bind, resume, save
from exp12_pairs import evaluate
from exp17_selection import MODEL_REVISION, WEATHER, anchors_for, comparisons, load, subset
from huggingface_hub import snapshot_download


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if os.environ.get('TIMESFM_BACKEND') != 'mlx':
        raise ValueError('MLX required')
    df = load()
    original = anchors_for(df)
    # Same eligible months, up to two new dates per month. Avoid all original
    # scored horizons; earliest then latest valid date, at least 96h apart.
    months = {(t.year, t.month) for t in original}
    eligible = {}
    for pos in range(8759, len(df) - 96, 24):
        ts = df.index[pos]
        key = (ts.year, ts.month)
        if key not in months or min(abs(ts - t) for t in original) < pd.Timedelta(hours=96):
            continue
        past = df.iloc[pos - 8759:pos + 1]
        future = df.iloc[pos + 1:pos + 97]
        if past.notna().mean().min() < .95 or past.iloc[[0, -1]].isna().any().any():
            continue
        if future[['eisbach', *WEATHER]].isna().any().any():
            continue
        eligible.setdefault(key, []).append(ts)
    anchors = []
    for dates in eligible.values():
        for ts in [dates[0], dates[-1]]:
            if not anchors or min(abs(ts - t) for t in anchors) >= pd.Timedelta(hours=96):
                anchors.append(ts)
    anchors.sort()
    LOG = logging.getLogger(__name__)
    LOG.info('%d additional, disjoint origins', len(anchors))
    path = bench.CACHE / 'exp17_puppling.csv'
    bind(path, df, anchors, 8760, [[str(m), WEATHER, subset(m)] for m in [4, 6]])
    provenance = {'source': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'shared_source': hashlib.sha256(Path(__file__).with_name('exp17_selection.py')
                                                  .read_bytes()).hexdigest(),
                  'revision': MODEL_REVISION}
    meta = path.with_name('exp17_puppling_runtime.json')
    if meta.exists() and json.loads(meta.read_text()) != provenance:
        raise ValueError('Changed provenance')
    meta.write_text(json.dumps(provenance, indent=2))
    rows = resume(path, [4, 6], anchors)
    for r in rows:
        r['label'] = str(r['label'])
    model_path = snapshot_download('google/timesfm-3.0-pytorch', revision=MODEL_REVISION,
                                   local_files_only=True)
    fc = bench.load_forecaster(model_path)
    for mask in [4, 6]:
        if str(mask) in {r['label'] for r in rows}:
            continue
        arm = []
        t0 = time.monotonic()
        for start in range(0, len(anchors), 8):
            arm += evaluate(fc, df, anchors[start:start + 8], df.eisbach,
                            label=str(mask), past_only=subset(mask), context=8760, future=WEATHER)
            LOG.info('mask=%d %d/%d %.1fs', mask, min(start + 8, len(anchors)),
                     len(anchors), time.monotonic() - t0)
        rows += arm
        save(path, rows)
    scores = pd.DataFrame(rows)
    # Cluster into calendar months before bootstrap; new origins are not treated
    # as independent months. Each month has equal weight, matching stage one.
    scores['reference_time'] = pd.to_datetime(scores.reference_time).dt.strftime('%Y-%m-01')
    scores['reference_time'] = pd.to_datetime(scores.reference_time, utc=True)
    result = {f'block_{b}': comparisons(scores, reference='4', block=b) for b in [1, 3, 6]}
    if not np.isfinite(pd.DataFrame(rows)[['mae', 'crps']].to_numpy()).all():
        raise ValueError('Nonfinite scores')
    path.with_name('exp17_puppling_summary.json').write_text(json.dumps(result, indent=2))
    LOG.info('%s', result['block_3'])


if __name__ == '__main__':
    main()
