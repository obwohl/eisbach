"""Actually rerun the pre-change DUET adapter on the healthy comparison windows."""
import importlib.util
import json
import logging
import subprocess

import bench
import exp19_final_vs_duet as gallery
import numpy as np
import pandas as pd
import torch
from exp17_selection import load

OLD = '7024d7ff5ee2'
OUT = bench.CACHE / 'old_duet_vs_timesfm'


def old_module(name, relative, dest):
    content = subprocess.check_output(['git', 'show', f'{OLD}:{relative}'])
    dest.write_bytes(content)
    spec = importlib.util.spec_from_file_location(name, dest)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    logging.basicConfig(level=logging.INFO)
    OUT.mkdir(parents=True, exist_ok=True)
    api = old_module('eisbach.model.api_before_change', 'eisbach/model/api.py', OUT / 'old_api.py')
    data = old_module('data_before_change', 'eisbach/data.py', OUT / 'old_data.py')
    torch.set_num_threads(4)
    model, config = api.load_model(device='cpu')
    df = load()
    weather = bench.load_dataset()[['airtemp', 'pressure']].rename(columns={'airtemp': 'lufttemperatur_c'})
    source = bench.CACHE / 'final_oracle_vs_duet'
    manifest = json.loads((source / 'manifest.json').read_text())
    origins, diffs = [], []
    for j, ref in enumerate(manifest['origins']):
        ts = pd.Timestamp(ref)
        if ts >= pd.Timestamp('2026-09-09 08:00', tz='UTC'):
            continue
        w = dict(np.load(source / f'window_{j:02d}.npz'))
        lo = ts - pd.Timedelta(hours=config.seq_len - 1)
        water = df.eisbach.loc[lo:ts].rename('wassertemp').rename_axis('timestamp').reset_index()
        frame = data.assemble_long_frame(water, weather.loc[lo:ts + pd.Timedelta(hours=96)])
        result = api.forecast(model, config, api.long_to_wide(frame[frame.date <= ts]))
        cols = [f'wassertemp_q{q}' for q in bench.DUET_QUANTILES]
        q = bench.to_deciles(result[cols].to_numpy(), bench.DUET_QUANTILES)
        diffs.append(float(np.max(abs(q - w['q'][0]))))
        w['q'][0] = q
        np.savez_compressed(OUT / f'window_{len(origins):02d}.npz', **w)
        origins.append(ref)
    manifest.update(origins=origins, old_code=OLD, max_quantile_difference=max(diffs))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    gallery.OUT = OUT
    gallery.LABELS = ['DUET vor Umstellung · Orakel', 'TimesFM final · Orakel',
                      'DUET damals live', 'TimesFM Luft · Replay']
    gallery.COLORS = ['#1771F1', '#E35B38', '#777777', '#27806B']
    gallery.render(report_name='REPORT_exp20.md')
    logging.info('Old vs current oracle decoder: max absolute difference %s over %s windows',
                 max(diffs), len(origins))


if __name__ == '__main__':
    main()
