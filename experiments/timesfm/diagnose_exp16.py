"""Empirical lead-error curves and temporal-dependence audit, no model fitting."""
import json

import bench
import numpy as np
import pandas as pd
from exp8_catchment_weather import load_all
from exp16_honesty import degrade, error_pool
from replay_audit import field, legacy_reader
from report_exp9_exp14 import table

from eisbach import archive


def correlation(x, lag):
    # Centre each lead across reference times before pooling pairs; otherwise the
    # systematic lead-bias curve itself can be mistaken for serial error dependence.
    z = x - x.mean(axis=0, keepdims=True)
    return float(np.corrcoef(z[:, :-lag].ravel(), z[:, lag:].ravel())[0, 1])


def main():
    errors = pd.read_csv(bench.CACHE / 'exp16_errors.csv', index_col=0, parse_dates=[0])
    errors.columns = errors.columns.astype(int)
    a = errors.to_numpy()
    long = errors.stack().rename('error').reset_index()
    long.columns = ['reference_time', 'lead', 'error']
    pool = error_pool(long)
    rng = np.random.default_rng(0)
    block = np.stack([degrade(np.zeros(96), pool, rng) for _ in range(10000)])
    trace = a[rng.integers(len(a), size=10000)]
    parts = ['## DWD error by lead hour\n',
             'Error = forecast minus cached observation, °C. Lead is relative to the '
             'hourly reference time, not an archived DWD model issue time. No such issue '
             f'timestamp is available. Each hour has the same {len(a)} replay windows.\n']
    rows = [[i + 1, len(a), f'{a[:, i].mean():.3f}', f'{np.abs(a[:, i]).mean():.3f}',
             f'{np.sqrt(np.mean(a[:, i]**2)):.3f}'] for i in range(96)]
    parts.append(table(['Lead h', 'N', 'Bias °C', 'MAE °C', 'RMSE °C'], rows))
    rows = [[lag, *[f'{correlation(x, lag):.4f}' for x in (a, block, trace)]]
            for lag in (1, 2, 3, 6, 12, 24, 48, 72)]
    parts += ['## Forecast-error autocorrelation\n',
              'Pooled correlation after centring each lead across forecasts. Simulations '
              'use 10,000 traces, seed 0; these are dependence diagnostics, not extra '
              'independent weather observations.\n',
              table(['Lag h', 'Real', 'Original 24h blocks', 'Whole 96h trace'], rows)]
    rows = []
    for left in (12, 24, 36, 48, 60, 72, 84):
        rows.append([f'{left} → {left+1}',
                     *[f'{np.corrcoef(x[:, left-1], x[:, left])[0, 1]:.4f}'
                       for x in (a, block, trace)]])
    parts += [table(['Adjacent leads', 'Real', '24h blocks', '96h trace'], rows)]
    audit = pd.read_csv(bench.CACHE / 'exp16_lookup_audit.csv', parse_dates=['reference_time'])
    good = audit[~audit.late & audit.air & audit.complete]
    rows = []
    for month, g in good.groupby(good.reference_time.dt.strftime('%Y-%m')):
        rows.append([month, len(g), g.fetched.nunique(), int(g.solar.sum())])
    parts += ['## Replay coverage\n', table(['Month', 'Windows', 'Distinct fetches', 'With solar'], rows)]
    times = errors.index
    diffs = times.to_series().diff().dropna().dt.total_seconds().to_numpy() / 3600
    unique_targets = set()
    selected = []
    for ts in times:
        unique_targets.update(pd.date_range(ts + pd.Timedelta(hours=1), periods=96, freq='h'))
        if not selected or ts - selected[-1] >= pd.Timedelta(hours=96):
            selected.append(ts)
    diagnostics = dict(windows=len(a), first=str(times.min()), last=str(times.max()),
                       unique_targets=len(unique_targets), total_targets=len(a)*96,
                       adjacent_overlap_fraction=float((diffs < 96).mean()),
                       median_spacing_hours=float(np.median(diffs)),
                       nonoverlapping_windows=len(selected), fetches=good.fetched.nunique())
    (bench.CACHE / 'exp16_dependence.json').write_text(json.dumps(diagnostics, indent=2))
    solar_rows = []
    df = load_all()
    with legacy_reader():
        for ts in good.loc[good.solar, 'reference_time']:
            targets = pd.date_range(ts + pd.Timedelta(hours=1), periods=96, freq='h')
            snap, _ = archive.load_weather_snapshot(ts)
            forecast = field(snap, targets, ('solar',))
            observed = df.solar_muenchen.reindex(targets).to_numpy()
            if not np.isfinite(observed).all():
                continue
            solar_rows.extend([dict(reference_time=ts, lead=i + 1, forecast=f, observed=o, error=f-o)
                               for i, (f, o) in enumerate(zip(forecast, observed, strict=True))])
    solar = pd.DataFrame(solar_rows)
    solar.to_csv(bench.CACHE / 'exp16_solar_errors.csv', index=False)
    rows = []
    for lo, hi in ((0, 24), (24, 48), (48, 72), (72, 96)):
        v = solar[solar.lead.between(lo + 1, hi)].error
        rows.append([f'{lo+1}–{hi}', len(v), f'{v.mean():+.5f}', f'{v.abs().mean():.5f}'])
    parts += ['## Archived solar-field error (kWh/m²)\n',
              f'{solar.reference_time.nunique()} windows with complete observed solar horizons; '
              f'zero observed radiation in {(solar.observed == 0).mean():.1%} of hours. '
              f'MAE on positive-observed-radiation hours: '
              f'{solar.loc[solar.observed > 0, "error"].abs().mean():.5f} kWh/m².\n',
              table(['Lead hours', 'Forecast-hour pairs', 'Bias', 'MAE'], rows)]
    parts.append('Overlap diagnostics: `' + json.dumps(diagnostics) + '`\n')
    (bench.CACHE / 'exp16_diagnostics.md').write_text('\n'.join(parts))
    print(json.dumps(diagnostics, indent=2))
    print(parts[5])


if __name__ == '__main__':
    main()
