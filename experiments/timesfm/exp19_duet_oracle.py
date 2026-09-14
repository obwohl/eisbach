"""Add a true oracle-vs-oracle comparison using unchanged production DUET weights."""
import hashlib
import json
import logging
from pathlib import Path

import bench
import exp19_final_vs_duet as showcase
import numpy as np
import pandas as pd
import torch
from exp17_selection import load

from eisbach.data import assemble_long_frame
from eisbach.model import CHECKPOINT_SHA256, forecast, load_model, long_to_wide

OUT = bench.CACHE / 'final_oracle_vs_duet'


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    source = showcase.OUT
    manifest = json.loads((source / 'manifest.json').read_text())
    df = load()
    weather = bench.load_dataset()[['airtemp', 'pressure']].rename(columns={'airtemp': 'lufttemperatur_c'})
    model, config = load_model(device='cpu')
    if config.horizon != 96 or config.seq_len != 384:
        raise ValueError('Unexpected production model configuration')
    manifest['duet_oracle'] = dict(checkpoint=CHECKPOINT_SHA256, context=config.seq_len,
                                 channels=['wassertemp', 'airtemp_96', 'pressure_96'],
                                 future='observed Munich air and pressure, shifted by 96 hours',
                                 source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for i, ref in enumerate(manifest['origins']):
        ts = pd.Timestamp(ref)
        original = dict(np.load(source / f'window_{i:02d}.npz'))
        if str(original['model_id']) != CHECKPOINT_SHA256[:12]:
            raise ValueError('Archived and oracle DUET checkpoints differ')
        start = ts - pd.Timedelta(hours=config.seq_len - 1)
        water = df.eisbach.loc[start:ts].rename('wassertemp').rename_axis('timestamp').reset_index()
        meteo = weather.loc[start:ts + pd.Timedelta(hours=96)].copy()
        if meteo.iloc[-96:].isna().any().any():
            raise ValueError('Missing future oracle weather')
        # Same cleaned, measured history as TimesFM; no future water supplied.
        frame = assemble_long_frame(water, meteo)
        wide = long_to_wide(frame[frame.date <= ts])
        result = forecast(model, config, wide)
        expected = pd.date_range(ts + pd.Timedelta(hours=1), periods=96, freq='h')
        if not result.index.equals(expected):
            raise ValueError('Misaligned DUET forecast')
        cols = [f'wassertemp_q{q}' for q in bench.DUET_QUANTILES]
        q = bench.to_deciles(result[cols].to_numpy(), bench.DUET_QUANTILES)
        if not np.isfinite(q).all():
            raise ValueError('Nonfinite DUET oracle output')
        original['q'] = np.stack([q, original['q'][1], original['q'][0], original['q'][2]])
        np.savez_compressed(OUT / f'window_{i:02d}.npz', **original)
        logging.info('DUET oracle %d/%d %s', i + 1, len(manifest['origins']), ts)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    showcase.OUT = OUT
    showcase.LABELS = ['DUET · Orakel', 'TimesFM final · Orakel', 'DUET live', 'TimesFM Luft · Replay']
    showcase.COLORS = ['#1771F1', '#E35B38', '#777777', '#27806B']
    showcase.render()


if __name__ == '__main__':
    main()
