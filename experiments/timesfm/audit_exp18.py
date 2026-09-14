"""Reproducible input/decoder audit and full-context sensitivity for exp18."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from importlib.metadata import version
from pathlib import Path

import bench
import numpy as np
import pandas as pd
from exp6_upstream import TARGET, pick_anchors
from exp8_catchment_weather import load_all
from exp18_discharge_alone import FLUX, WEATHER, Q, T, select_anchors, sweep

VARIANTS = [
    ('nur Eisbach', [], []), ('T', [], [T]), ('Q', [], [Q]),
    ('T+Q', [], [T, Q]), ('T×Q (Wärmestrom)', [], [FLUX]),
    ('T+Q+T×Q', [], [T, Q, FLUX]), ('Wetter', WEATHER, []),
    ('Wetter + T', WEATHER, [T]), ('Wetter + Q', WEATHER, [Q]),
    ('Wetter + T+Q', WEATHER, [T, Q]), ('Wetter + T×Q', WEATHER, [FLUX]),
]
PAIRS = [
    ('T+Q', 'T'), ('Q', 'nur Eisbach'), ('T×Q (Wärmestrom)', 'T'),
    ('T+Q+T×Q', 'T+Q'), ('Wetter + T+Q', 'Wetter + T'),
    ('Wetter + Q', 'Wetter'), ('Wetter + T×Q', 'Wetter + T'),
    ('T×Q (Wärmestrom)', 'T+Q'), ('Wetter + T×Q', 'Wetter + T+Q'),
]


def dataset():
    df = load_all()
    df[FLUX] = df[Q] * df[T]
    anchors = pick_anchors(df, [*WEATHER, T, Q, TARGET], n=250)
    return df, anchors


def data_audit(df, anchors):
    q = df[Q]
    diff = q.diff()
    gauges = df[[Q, 'q_lenggries', 'q_puppling']]
    spacing = pd.Series(anchors).diff().dt.total_seconds() / 3600
    out = {
        'range': [str(df.index.min()), str(df.index.max())],
        'q_count': int(q.count()), 'q_missing': int(q.isna().sum()),
        'longest_q_gap_hours': int(q.isna().groupby(q.notna().cumsum()).sum().max()),
        'q_quantiles': q.quantile([0, .01, .5, .95, .99, 1]).to_dict(),
        'abs_hourly_diff': diff.abs().quantile([.5, .9, .95, .99, 1]).to_dict(),
        'unchanged_pct': 100 * float(diff.eq(0).sum() / diff.count()),
        'jumps_gt10': int(diff.abs().gt(10).sum()),
        'jumps_gt20': int(diff.abs().gt(20).sum()),
        'level_corr': gauges.corr().to_dict(),
        'difference_corr': gauges.diff().corr().to_dict(),
        'anchors_year': pd.Series(anchors).dt.year.value_counts().sort_index().to_dict(),
        'anchor_spacing_hours': spacing.describe().to_dict(),
        'overlap_pairs_adjacent': int(spacing.lt(96).sum()),
        'largest_jumps': {str(k): float(v) for k, v in diff.abs().nlargest(12).items()},
    }
    coverage = []
    for context in (1024, 8760):
        for col in (TARGET, T, Q):
            windows = [df[col].iloc[df.index.get_loc(a) - context + 1:df.index.get_loc(a) + 1]
                       for a in anchors]
            coverage.append({'context': context, 'column': col,
                             'max_missing_fraction': max(float(w.isna().mean()) for w in windows),
                             'all_missing_windows': sum(w.isna().all() for w in windows),
                             'leading_missing_windows': sum(pd.isna(w.iloc[0]) for w in windows)})
    out['coverage'] = coverage
    (bench.CACHE / 'exp18_data_audit.json').write_text(json.dumps(out, indent=2, default=int))


def decoder_audit(fc, df, anchors):
    records = []
    # Include the largest target-gap case, plus times spread across the record.
    for context in (1024, 8760):
        gap_anchor = max(anchors, key=lambda a: df[TARGET].iloc[
            df.index.get_loc(a) - context + 1:df.index.get_loc(a) + 1].isna().sum())
        selected = list(dict.fromkeys([anchors[0], anchors[len(anchors)//2], anchors[-1], gap_anchor]))
        targets, past, future = [], [], []
        for a in selected:
            pos = df.index.get_loc(a)
            sl = slice(pos-context+1, pos+1)
            raw = df[TARGET].iloc[sl].to_numpy(np.float32)
            po = df[[T, Q]].iloc[sl].to_numpy(np.float32).T
            pf = df[WEATHER].iloc[pos-context+1:pos+97].to_numpy(np.float32).T
            bare, _, _ = bench.prepare_inputs(raw, None, None)
            with_q, po, pf = bench.prepare_inputs(raw, po, pf)
            np.testing.assert_array_equal(bare, with_q)
            assert po.shape[1] == len(bare) and pf.shape[1] == len(bare)+96
            assert all(np.isfinite(z).all() for z in (bare, po, pf))
            targets.append(bare)
            past.append(po)
            future.append(pf)
        fast = list(fc.predict_batch(targets, horizon=96, return_quantiles=True))
        general = list(fc.predict_batch([a[None, :] for a in targets], horizon=96,
                                        return_quantiles=True))
        for a, x, y in zip(selected, fast, general, strict=True):
            delta = float(np.max(np.abs(x.quantiles-y.quantiles[0])))
            assert delta < 1e-4, delta
            records.append({'context': context, 'anchor': str(a), 'test': 'fast_vs_general',
                            'max_quantile_difference': delta})
        for i, a in enumerate(selected):
            for pf in (None, future[i]):
                one = fc.predict(targets[i], horizon=96, past_only_covariates=past[i],
                                 past_future_covariates=pf, return_quantiles=True)
                stack = fc.predict(np.vstack([targets[i], past[i]]), horizon=96,
                                   past_future_covariates=pf, return_quantiles=True)
                delta = float(np.max(np.abs(one.quantiles-stack.quantiles[0])))
                assert delta < 1e-4, delta
                records.append({'context': context, 'anchor': str(a), 'test': 'past_only_vs_target',
                                'weather': pf is not None, 'max_quantile_difference': delta})
    import inspect

    import timesfm3.mlx.model
    import timesfm3.mlx.timesfm3_forecaster
    sources = [Path(inspect.getfile(module)) for module in
               (timesfm3.mlx.model, timesfm3.mlx.timesfm3_forecaster, bench)]
    revision = bench.CACHE / 'hf/hub/models--google--timesfm-3.0-pytorch/refs/main'
    out = {'checks': records, 'platform': platform.platform(),
           'packages': {p: version(p) for p in ('timesfm', 'mlx', 'numpy', 'pandas')},
           'cached_hf_revision': revision.read_text().strip(),
           'sources_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (bench.CACHE / 'exp18_decoder_audit.json').write_text(json.dumps(out, indent=2))
    logging.info('All %d decoder comparisons passed; maximum quantile delta %.9g',
                 len(records), max(r['max_quantile_difference'] for r in records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm', action='store_true',
                        help='Check screened-out variants at 8760 h on the same 100 stage-1 anchors')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    df, anchors = dataset()
    data_audit(df, anchors)
    fc = bench.load_forecaster()
    decoder_audit(fc, df, anchors)
    if args.confirm:
        stage3 = pd.read_csv(bench.CACHE / 'exp18_stage3.csv', parse_dates=['reference_time'])
        selected = select_anchors(anchors, 100)
        missing = [v for v in VARIANTS if v[0] not in set(stage3.label)]
        sweep(fc, df, df[TARGET], selected, missing, bench.CACHE / 'exp18_context_extra.csv',
              context=8760)


if __name__ == '__main__':
    main()
