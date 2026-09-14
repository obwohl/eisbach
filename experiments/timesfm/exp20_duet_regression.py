"""Paired diagnosis of the September DUET band explosion, without archive writes."""
import io
import json
import logging
import subprocess

import bench
import matplotlib
import numpy as np
import pandas as pd
import torch

from eisbach.data import assemble_long_frame
from eisbach.model import forecast, load_model, long_to_wide

matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = bench.CACHE / 'duet_regression'
REF = pd.Timestamp('2026-09-14 17:00', tz='UTC')
FAULT = pd.Timestamp('2026-09-09 08:00', tz='UTC')
PRODUCTION = '722f467f196c2162fb5403b9a9e157ceeb5bf8d4'


def archive(kind):
    parts = []
    for month in ['2026-08', '2026-09']:
        raw = subprocess.check_output(['git', 'show', f'{PRODUCTION}:data/archive/{kind}/{month}.csv'])
        parts.append(pd.read_csv(io.BytesIO(raw)))
    return pd.concat(parts, ignore_index=True)


def main():
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model, cfg = load_model(device='cpu')
    obs = archive('observations')
    obs.timestamp = pd.to_datetime(obs.timestamp, utc=True, format='mixed')
    water = obs.drop_duplicates('timestamp', keep='last').set_index('timestamp').wassertemp.sort_index()
    meteo = bench.load_dataset()[['airtemp', 'pressure']].rename(columns={'airtemp': 'lufttemperatur_c'})
    snap = archive('weather')
    snap.reference_time = pd.to_datetime(snap.reference_time, utc=True, format='mixed')
    snap.timestamp = pd.to_datetime(snap.timestamp, utc=True, format='mixed')
    snap = snap[snap.reference_time.eq(REF)].drop_duplicates('timestamp').set_index('timestamp')
    weather = snap[['lufttemperatur_c', 'pressure']].combine_first(meteo).sort_index()
    start = REF - pd.Timedelta(hours=383)
    raw = water.loc[start:REF].reindex(pd.date_range(start, REF, freq='h'))
    repaired = raw.copy()
    assert raw.loc[FAULT] == 154.4
    repaired.loc[FAULT] = np.nan
    repaired = repaired.interpolate(method='time')
    recorded = archive('forecasts')
    recorded.reference_time = pd.to_datetime(recorded.reference_time, utc=True, format='mixed')
    recorded.target_time = pd.to_datetime(recorded.target_time, utc=True, format='mixed')
    recorded = recorded[recorded.reference_time.eq(REF) & recorded.kind.eq('live')].sort_values('target_time')
    cols = [f'wassertemp_q{q}' for q in bench.DUET_QUANTILES]
    results = {'archived_live': recorded[cols].to_numpy()}
    stats = {}
    for name, series in [('with_spike', raw), ('one_hour_repaired', repaired)]:
        frame = assemble_long_frame(series.rename('wassertemp').rename_axis('timestamp').reset_index(),
                                    weather.loc[start:REF + pd.Timedelta(hours=96)])
        wide = long_to_wide(frame[frame.date <= REF])
        q = forecast(model, cfg, wide)[cols].to_numpy()
        results[name] = q
        values = wide.wassertemp.to_numpy()
        stats[name] = dict(median_last=float(q[-1, 3]),
                           width98=float(np.mean(q[:, -1] - q[:, 0])),
                           water_rms_about_median=float(np.sqrt(np.mean((values - np.median(values)) ** 2))),
                           max_abs_difference_from_archived=float(np.max(abs(q - results['archived_live']))))
    stats['repaired_value'] = float(repaired.loc[FAULT])
    (OUT / 'diagnosis.json').write_text(json.dumps(stats, indent=2))
    np.savez_compressed(OUT / 'quantiles.npz', **results)
    times = pd.date_range(REF + pd.Timedelta(hours=1), periods=96, freq='h').tz_convert('Europe/Berlin')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True, layout='constrained')
    for ax, name, title in zip(axes, ['with_spike', 'one_hour_repaired'],
                               ['DUET mit 154,4-°C-Ausreißer', 'Dasselbe DUET, nur diese Stunde repariert'],
                               strict=True):
        q = results[name]
        ax.fill_between(times, q[:, 0], q[:, -1], color='#1771F1', alpha=.13, label='98-%-Band')
        ax.fill_between(times, q[:, 1], q[:, -2], color='#1771F1', alpha=.16, label='90-%-Band')
        ax.fill_between(times, q[:, 2], q[:, -3], color='#1771F1', alpha=.2, label='50-%-Band')
        ax.plot(times, q[:, 3], color='#1771F1', label='Median')
        ax.set_title(title, loc='left')
        ax.set_ylabel('Wassertemperatur (°C)')
        ax.grid(alpha=.2)
        ax.legend(ncols=4, fontsize=9)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))
    fig.suptitle('Ursachentest · gleiche Gewichte, gleicher Wetterforecast, gleicher Prognosestart\n'
                 '14.09.2026, 19:00 Uhr · nur eine historische Wasserbeobachtung geändert')
    fig.savefig(OUT / 'diagnosis.png', dpi=150)
    logging.info('%s', stats)


if __name__ == '__main__':
    main()
