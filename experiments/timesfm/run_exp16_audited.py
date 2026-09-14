"""Audited exp16: strict availability, paired controls and repeated simulations."""
import hashlib
import json
import logging
import time
from pathlib import Path

import bench
import numpy as np
import pandas as pd
from checkpoint import bind, resume, save
from exp8_catchment_weather import load_all
from exp16_honesty import CONTEXT, DRAWS, degrade, error_pool, score_variant
from replay_audit import select

LOG = logging.getLogger(__name__)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(fc, df, items, variants, name, provenance):
    path = bench.CACHE / f'{name}.csv'
    anchors = [ts for ts, _ in items]
    bind(path, df, anchors, CONTEXT,
         [[label, future, past, provenance] for label, future, past, _ in variants])
    rows = resume(path, [v[0] for v in variants], anchors)
    done = {r['label'] for r in rows}
    for label, future, past, overrides in variants:
        if label in done:
            continue
        t0 = time.time()
        for i in range(0, len(items), 32):
            chunk = overrides[i:i + 32]
            rows += score_variant(fc, df, df.eisbach, chunk, label=label,
                                  future=dict.fromkeys(future), past_only=past)
            LOG.info('%s: %d/%d windows', label, min(i + 32, len(items)), len(items))
        save(path, rows)
        LOG.info('%s complete, %.1fs', label, time.time() - t0)
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.getLogger('eisbach.archive').setLevel(logging.WARNING)
    df = load_all()
    items, audit = select(df, CONTEXT, bench.HORIZON)
    audit.to_csv(bench.CACHE / 'exp16_lookup_audit.csv', index=False)
    if not items:
        raise RuntimeError('No strictly available, fully observed replay windows')
    LOG.info('%d strictly replayable windows, %s .. %s', len(items), items[0][0], items[-1][0])
    errors = np.stack([o['airtemp'] - df.airtemp.loc[
        ts + pd.Timedelta(hours=1):ts + pd.Timedelta(hours=96)].to_numpy()
        for ts, o in items])
    error_frame = pd.DataFrame(errors, index=[ts for ts, _ in items], columns=range(1, 97))
    error_frame.index.name = 'reference_time'
    error_frame.to_csv(bench.CACHE / 'exp16_errors.csv')
    provenance = {
        'sources': {p.name: digest(p) for p in [
            Path(__file__), Path(__file__).with_name('exp16_honesty.py'),
            Path(__file__).with_name('replay_audit.py'),
            Path(__file__).with_name('bench.py'), Path(__file__).with_name('checkpoint.py'),
            Path(archive_path())]},
        'archive': {p.name: digest(p) for p in sorted(bench.ARCHIVE.glob('weather/*.csv'))},
        'error_hash': hashlib.sha256(errors.tobytes()).hexdigest(),
        'draws': DRAWS, 'simulation_seed': 0,
    }
    (bench.CACHE / 'exp16_provenance.json').write_text(json.dumps(provenance, indent=2))
    oracle = [(ts, {}) for ts, _ in items]
    replay = [(ts, {'airtemp': o['airtemp']}) for ts, o in items]
    past = ['isar_toelz']
    fc = bench.load_forecaster()
    variants = [
        ('Ohne Wetter', [], past, oracle),
        ('Muenchen Orakel', ['airtemp'], past, oracle),
        ('Muenchen Replay', ['airtemp'], past, replay),
        ('Zwei Luft Orakel', ['airtemp', 't_catchment'], past, oracle),
        ('Muenchen Replay + Sued Orakel', ['airtemp', 't_catchment'], past, replay),
    ]
    run(fc, df, items, variants, 'exp16_audited', provenance)
    solar_items = [(ts, o) for ts, o in items if o['_solar'] is not None
                   and np.isfinite(df.solar_muenchen.loc[
                       ts + pd.Timedelta(hours=1):ts + pd.Timedelta(hours=96)]).all()]
    if solar_items:
        so = [(ts, {}) for ts, _ in solar_items]
        sr = [(ts, {'airtemp': o['airtemp'], 'solar_muenchen': o['_solar']})
              for ts, o in solar_items]
        sa = [(ts, {'airtemp': o['airtemp']}) for ts, o in solar_items]
        run(fc, df, solar_items, [
            ('Luft Solar Orakel', ['airtemp', 'solar_muenchen'], past, so),
            ('Luft Solar Replay', ['airtemp', 'solar_muenchen'], past, sr),
            ('Luft Replay Solar Orakel', ['airtemp', 'solar_muenchen'], past, sa),
        ], 'exp16_solar', provenance)
    # A fixed non-overlapping subset limits repeated simulation cost and avoids
    # pretending 20 Monte Carlo draws are 20 independent observed seasons.
    subset = []
    for item in items:
        if not subset or item[0] - subset[-1][0] >= pd.Timedelta(hours=96):
            subset.append(item)
    long = error_frame.stack().rename('error').reset_index()
    long.columns = ['reference_time', 'lead', 'error']
    pool = error_pool(long)
    rng = np.random.default_rng(0)
    for draw in range(DRAWS):
        sim_variants = []
        for method in ('block24', 'trace96'):
            simulated = []
            for ts, o in subset:
                obs = df.t_catchment.loc[ts + pd.Timedelta(hours=1):
                                        ts + pd.Timedelta(hours=96)].to_numpy()
                noise = (degrade(np.zeros(96), pool, rng) if method == 'block24'
                         else errors[rng.integers(len(errors))])
                simulated.append((ts, {'airtemp': o['airtemp'], 't_catchment': obs + noise}))
            sim_variants.append((method, ['airtemp', 't_catchment'], past, simulated))
        run(fc, df, subset, sim_variants, f'exp16_sim_{draw:02d}', provenance)
    LOG.info('DONE: %d main windows; %d simulation windows x %d draws x 2 methods',
             len(items), len(subset), DRAWS)


def archive_path():
    from eisbach import archive
    return archive.__file__


if __name__ == '__main__':
    main()
