"""Paired subset ablation on the fixed exp17 configuration (oracle weather)."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from importlib.metadata import version
from pathlib import Path

import bench
import numpy as np
import pandas as pd
from checkpoint import bind, resume, save
from exp8_catchment_weather import load_all
from exp12_pairs import evaluate

LOG = logging.getLogger(__name__)
WEATHER = ['airtemp', 't_catchment', 'rain_toelz', 'rain_lenggries',
           'rain_kochel', 'rain_garmisch', 'solar_hohenpeissenberg']
CANDIDATES = ['isar_lenggries', 'isar_puppling', 'loisach_beuerberg']
BASE = ['isar_toelz']
MODEL_REVISION = '43046b85ec22d584a13f8098c2ed39c889e129c2'


def load():
    df = load_all()[['eisbach', *WEATHER[:-1], *BASE, *CANDIDATES]].copy()
    solar = pd.read_csv(bench.CACHE / 'exp17_solar_hp.csv', index_col=0, parse_dates=[0])
    solar.index = pd.to_datetime(solar.index, utc=True)
    df = df.join(solar)
    for col in ['eisbach', *BASE, *CANDIDATES]:
        df.loc[~df[col].between(*bench.PLAUSIBLE_RANGE), col] = np.nan
    return df


def anchors_for(df):
    """One origin per calendar month, selected by availability alone, >=96h apart.

    All arms use the same full-year-eligible origins. Future water never enters
    eligibility; truth and future weather must be fully observed, without filling.
    """
    choices = {}
    for pos in range(8759, len(df) - 96, 24):
        ts = df.index[pos]
        history = df.iloc[pos - 8759:pos + 1]
        future = df.iloc[pos + 1:pos + 97]
        if history.notna().mean().min() < .95:
            continue
        if history.iloc[[0, -1]].isna().any().any():
            continue
        if future[['eisbach', *WEATHER]].isna().any().any():
            continue
        choices.setdefault((ts.year, ts.month), []).append(ts)
    anchors = [g[len(g) // 2] for g in choices.values()]
    return [t for i, t in enumerate(anchors)
            if i == 0 or t - anchors[i - 1] >= pd.Timedelta(hours=96)]


def subset(mask):
    return [*BASE, *[c for i, c in enumerate(CANDIDATES) if mask & (1 << i)]]


def comparisons(scores, reference='0', block=3):
    """Paired circular moving-block bootstrap over chronological monthly origins.

    Resample numerator and denominator together; never resample hours as independent.
    """
    out = []
    for metric in ['mae', 'crps']:
        values = bench.per_run(scores, metric).sort_index()
        values.columns = values.columns.astype(str)
        if values.isna().any().any():
            raise ValueError('Unequal paired coverage')
        base = values[reference].to_numpy()
        n = len(base)
        rng = np.random.default_rng(1701)
        starts = rng.integers(n, size=(10000, int(np.ceil(n / block))))
        ix = ((starts[..., None] + np.arange(block)) % n).reshape(10000, -1)[:, :n]
        for label in values:
            if label == reference:
                continue
            delta = values[label].to_numpy() - base
            boot = 100 * delta[ix].mean(axis=1) / base[ix].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            years = {}
            for year in values.index.year.unique():
                sel = values.index.year == year
                years[str(year)] = float(100 * delta[sel].mean() / base[sel].mean())
            out.append(dict(label=label, reference=reference, metric=metric, n=n,
                            baseline=float(base.mean()), value=float(values[label].mean()),
                            delta=float(delta.mean()), pct=float(100 * delta.mean() / base.mean()),
                            ci=[float(lo), float(hi)], better_in=float((delta < 0).mean()), years=years))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', type=int, choices=[1024, 8760], default=1024)
    parser.add_argument('--masks', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if os.environ.get('TIMESFM_BACKEND') != 'mlx':
        raise ValueError('This run requires TIMESFM_BACKEND=mlx')
    df = load()
    anchors = anchors_for(df)
    if not anchors:
        raise ValueError('No complete eligible origins')
    LOG.info('%d monthly nonoverlapping origins: %s to %s', len(anchors), anchors[0], anchors[-1])
    path = bench.CACHE / f'exp17_fixed_{args.context}.csv'
    variants = [[str(m), WEATHER, subset(m)] for m in range(8)]
    bind(path, df, anchors, args.context, variants)
    provenance = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      timesfm=version('timesfm'), mlx=version('mlx'), model_revision=MODEL_REVISION,
                      data_sha256=hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest(),
                      weather=WEATHER, candidates=CANDIDATES, context=args.context,
                      horizon=96, origins=[str(t) for t in anchors])
    meta = path.with_name(path.stem + '_runtime.json')
    if not args.analyze:
        if meta.exists() and json.loads(meta.read_text()) != provenance:
            raise ValueError('Runtime provenance changed')
        meta.write_text(json.dumps(provenance, indent=2))
    # resume reads numeric labels as integers: normalize after validating coverage.
    rows = resume(path, list(range(8)), anchors)
    for r in rows:
        r['label'] = str(r['label'])
    done = {r['label'] for r in rows}
    if not args.analyze:
        fc = None
        if any(str(m) not in done for m in args.masks):
            from huggingface_hub import snapshot_download

            model_path = snapshot_download('google/timesfm-3.0-pytorch',
                                           revision=MODEL_REVISION, local_files_only=True)
            fc = bench.load_forecaster(model_path)
        for mask in args.masks:
            label = str(mask)
            if label in done:
                continue
            t0 = time.monotonic()
            arm = []
            for start in range(0, len(anchors), 8):
                arm += evaluate(fc, df, anchors[start:start + 8], df.eisbach,
                                label=label, past_only=subset(mask),
                                context=args.context, future=WEATHER)
                LOG.info('ctx=%d mask=%d %d/%d (%.1fs)', args.context, mask,
                         min(start + 8, len(anchors)), len(anchors), time.monotonic() - t0)
            rows += arm
            save(path, rows)
    scores = pd.DataFrame(rows)
    result = {f'block_{b}': comparisons(scores, block=b) for b in [1, 3, 6]}
    result['buckets'] = {str(lo): comparisons(g) for lo, g in scores.groupby('lead_lo')}
    # All measured conditional increments expose redundancy and interactions.
    result['conditional'] = {ref: comparisons(scores, reference=ref)
                             for ref in sorted(scores.label.unique())}
    path.with_name(path.stem + '_summary.json').write_text(json.dumps(result, indent=2))
    for rec in result['block_3']:
        LOG.info('%s', rec)


if __name__ == '__main__':
    main()
