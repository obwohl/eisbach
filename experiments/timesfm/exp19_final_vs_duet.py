"""Small four-week showcase of the fixed TimesFM set against archived live DUET."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import subprocess
from pathlib import Path

import bench
import matplotlib
import numpy as np
import pandas as pd
from exp17_selection import MODEL_REVISION, WEATHER, load
from huggingface_hub import snapshot_download
from replay_audit import field

matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = bench.CACHE / 'final_vs_duet'
PRODUCTION = '722f467f196c2162fb5403b9a9e157ceeb5bf8d4'
START = pd.Timestamp('2026-08-17', tz='UTC')
END = pd.Timestamp('2026-09-14', tz='UTC')
PAST = ['isar_toelz', 'loisach_beuerberg']
LABELS = ['DUET live', 'TimesFM final · Orakel', 'TimesFM Luft · Replay']
COLORS = ['#1771F1', '#E35B38', '#27806B']
LOG = logging.getLogger(__name__)


def read_production(kind):
    frames = []
    for month in ['2026-08', '2026-09']:
        path = f'data/archive/{kind}/{month}.csv'
        raw = subprocess.check_output(['git', 'show', f'{PRODUCTION}:{path}'])
        dest = OUT / 'sources' / kind / f'{month}.csv'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        frames.append(pd.read_csv(io.BytesIO(raw)))
    return pd.concat(frames, ignore_index=True)


def select(df):
    forecasts = read_production('forecasts')
    for col in ['reference_time', 'target_time']:
        forecasts[col] = pd.to_datetime(forecasts[col], utc=True, format='mixed')
    forecasts = forecasts[forecasts.kind.eq('live') & forecasts.reference_time.between(START, END)]
    weather = read_production('weather')
    weather['archive_timestamp'] = pd.to_datetime(weather.archive_timestamp, utc=True, format='mixed')
    weather['reference_time'] = pd.to_datetime(weather.reference_time, utc=True, format='mixed')
    snapshots = [(ts, g.archive_timestamp.max(), g)
                 for ts, g in weather.groupby('reference_time')]
    valid, audit = [], []
    qcols = [f'wassertemp_q{q}' for q in bench.DUET_QUANTILES]
    for ts, g in forecasts.groupby('reference_time'):
        g = g.sort_values('target_time')
        expected = pd.date_range(ts + pd.Timedelta(hours=1), periods=96, freq='h')
        reason = ''
        if len(g) != 96 or not pd.DatetimeIndex(g.target_time).equals(expected):
            reason = 'incomplete/duplicate forecast grid'
        elif ts not in df.index or expected[-1] > df.index[-1]:
            reason = '96h not yet observed'
        elif not np.isfinite(g[qcols].to_numpy()).all():
            reason = 'nonfinite DUET quantiles'
        else:
            pos = df.index.get_loc(ts)
            hist = df[['eisbach', *WEATHER, *PAST]].iloc[pos - 8759:pos + 1]
            if len(hist) != 8760 or hist.notna().mean().min() < .95 or hist.iloc[0].isna().any():
                reason = 'history incomplete'
            elif df[['eisbach', *WEATHER]].reindex(expected).isna().any().any():
                reason = 'missing truth or oracle weather'
        replay = None
        # Use only snapshots archived before the nominal origin. The snapshot
        # written with this DUET run is later and is deliberately ineligible.
        if not reason:
            for anchor, fetched, snap in sorted(snapshots, key=lambda t: t[0], reverse=True):
                if anchor <= ts and fetched <= ts and ts - anchor <= pd.Timedelta(hours=24):
                    air = field(snap, expected, ('lufttemperatur_c', 'temperature'))
                    if air is not None:
                        replay = dict(air=air, anchor=str(anchor), fetched=str(fetched))
                        break
            if replay is None:
                reason = 'no timely complete Munich weather snapshot'
        audit.append(dict(reference_time=str(ts), reason=reason or 'eligible'))
        if not reason:
            valid.append((ts, g, replay))
    # One origin per day, nearest 09 UTC; selection never looks at forecast errors.
    chosen = {}
    for item in valid:
        ts = item[0]
        key = ts.date()
        if key not in chosen or abs(ts.hour - 9) < abs(chosen[key][0].hour - 9):
            chosen[key] = item
    selected = sorted(chosen.values(), key=lambda x: x[0])
    (OUT / 'eligibility.json').write_text(json.dumps(audit, indent=2))
    return selected


def run():
    if os.environ.get('TIMESFM_BACKEND') != 'mlx':
        raise ValueError('MLX required')
    df = load()
    selected = select(df)
    LOG.info('%d daily windows, %s to %s', len(selected), selected[0][0], selected[-1][0])
    spec = dict(production=PRODUCTION, model=MODEL_REVISION, future=WEATHER, past=PAST,
                context=8760, horizon=96, origins=[str(x[0]) for x in selected],
                frame_hash=hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest(),
                source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                replay=[{k: v for k, v in r.items() if k != 'air'} for _, _, r in selected])
    manifest = OUT / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError('Inputs changed: do not reuse these predictions')
    manifest.write_text(json.dumps(spec, indent=2))
    model = bench.load_forecaster(snapshot_download('google/timesfm-3.0-pytorch',
                                 revision=MODEL_REVISION, local_files_only=True))
    qcols = [f'wassertemp_q{q}' for q in bench.DUET_QUANTILES]
    for i, (ts, g, replay) in enumerate(selected):
        path = OUT / f'window_{i:02d}.npz'
        if path.exists():
            continue
        pos = df.index.get_loc(ts)
        target = df.eisbach.iloc[pos - 8759:pos + 1].to_numpy(dtype=np.float32)
        po = df[PAST].iloc[pos - 8759:pos + 1].to_numpy(dtype=np.float32).T
        pf = df[WEATHER].iloc[pos - 8759:pos + 97].to_numpy(dtype=np.float32).T
        q = [bench.to_deciles(g[qcols].to_numpy(dtype=float), bench.DUET_QUANTILES)]
        for past, future in [(po, pf), (None, np.concatenate([
                pf[0, :8760], replay['air']]).astype(np.float32)[None])]:
            t, p, f = bench.prepare_inputs(target, past, future)
            result = model.predict(t, horizon=96, past_only_covariates=p,
                                   past_future_covariates=f, return_quantiles=True)
            values = result.quantiles
            q.append(values if values.ndim == 2 else values[0])
        q = np.stack(q)
        if q.shape != (3, 96, 9) or not np.isfinite(q).all() or (np.diff(q, axis=2) < 0).any():
            raise ValueError('Invalid quantiles')
        np.savez_compressed(path, q=q, truth=df.eisbach.iloc[pos + 1:pos + 97].to_numpy(),
                            history=df.eisbach.iloc[pos - 47:pos + 1].to_numpy(),
                            reference=str(ts), model_id=str(g.model_id.iloc[0]))
        LOG.info('%d/%d %s', i + 1, len(selected), ts)


def local(times):
    return pd.DatetimeIndex(times).tz_convert('Europe/Berlin').tz_localize(None)


def render():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2, 'figure.facecolor': 'white'})
    manifest = json.loads((OUT / 'manifest.json').read_text())
    windows = [dict(np.load(OUT / f'window_{i:02d}.npz')) for i in range(len(manifest['origins']))]
    rows = []
    for w in windows:
        ts = pd.Timestamp(str(w['reference']))
        targets = pd.date_range(ts + pd.Timedelta(hours=1), periods=96, freq='h')
        for label, q in zip(LABELS, w['q'], strict=True):
            r = bench.Run(label, ts, targets, q, w['truth'])
            rows += bench.score(r, buckets=bench.FINE_BUCKETS)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / 'scores.csv', index=False)
    pooled = bench.pool(scores).set_index('label').loc[LABELS]
    # The recorded 9 September sensor fault reached the live DUET input, while
    # the historical TimesFM input is cleaned. Show a pre-fault sensitivity.
    prefault = scores[scores.reference_time < pd.Timestamp('2026-09-09 08:00', tz='UTC')]
    sensitivity = bench.pool(prefault).set_index('label').loc[LABELS]
    pooled.to_json(OUT / 'summary.json', orient='index', indent=2)
    sensitivity.to_json(OUT / 'prefault_summary.json', orient='index', indent=2)
    fig = plt.figure(figsize=(15, 10), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1])
    overview_axes = []
    for k in range(2):
        ax = fig.add_subplot(grid[k, :])
        for w in windows:
            ts = pd.Timestamp(str(w['reference']))
            times = local(pd.date_range(ts + pd.Timedelta(hours=1), periods=96, freq='h'))
            q = w['q'][k]
            ax.fill_between(times[:24], q[:24, 0], q[:24, -1], color=COLORS[k], alpha=.15)
            ax.fill_between(times[:24], (q[:24, 1] + q[:24, 2]) / 2,
                            (q[:24, 6] + q[:24, 7]) / 2, color=COLORS[k], alpha=.2)
            ax.plot(times[:24], q[:24, 4], color=COLORS[k], lw=1.5, ls='--' if k else '-')
            ax.plot(times[:24], w['truth'][:24], color='#242424', lw=1.4)
        overview_axes.append(ax)
        ax.set_title(f'{LABELS[k]} — täglich die ersten 24 Prognosestunden', loc='left')
        ax.set_ylabel('Wassertemperatur (°C)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))
    ylim = (min(a.get_ylim()[0] for a in overview_axes),
            max(a.get_ylim()[1] for a in overview_axes))
    for ax in overview_axes:
        ax.set_ylim(ylim)
    buckets = bench.pool(scores, by=['label', 'lead_lo', 'lead_hi'])
    for j, metric in enumerate(['mae', 'crps']):
        ax = fig.add_subplot(grid[2, j])
        for k, label in enumerate(LABELS):
            g = buckets[buckets.label.eq(label)].sort_values('lead_lo')
            ax.plot((g.lead_lo + g.lead_hi) / 2, g[metric], 'o-', color=COLORS[k], label=label)
        ax.set_xlabel('Prognosevorlauf (Stunden)')
        ax.set_ylabel(f'{metric.upper()} (°C)')
        ax.set_title(f'{metric.upper()} über alle 96 Stunden', loc='left')
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f'Finaler TimesFM-Satz gegen DUET · {len(windows)} tägliche 96-h-Fenster\n'
                 '17.08.–14.09.2026 · Schwarz: Messung · Bänder: 80 % und 50 % · '
                 'Orakel = gemessenes zukünftiges Wetter', fontsize=15)
    fig.savefig(OUT / 'overview.png', dpi=150)
    plt.close(fig)
    for i, w in enumerate(windows):
        ts = pd.Timestamp(str(w['reference']))
        targets = local(pd.date_range(ts + pd.Timedelta(hours=1), periods=96, freq='h'))
        history_times = local(pd.date_range(ts - pd.Timedelta(hours=47), periods=48, freq='h'))
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, sharey=True, layout='constrained')
        for k, ax in enumerate(axes):
            q = w['q'][k]
            ax.fill_between(targets, q[:, 0], q[:, -1], color=COLORS[k], alpha=.15, label='80-%-Band')
            ax.fill_between(targets, (q[:, 1] + q[:, 2]) / 2,
                            (q[:, 6] + q[:, 7]) / 2, color=COLORS[k], alpha=.23, label='50-%-Band')
            ax.plot(targets, q[:, 4], color=COLORS[k], ls='--' if k else '-', lw=2, label='Median')
            ax.plot(history_times, w['history'], color='#222222', lw=1.5)
            ax.plot(targets, w['truth'], color='#222222', lw=1.8, label='Messung')
            ax.axvline(local([ts])[0], color='#777777', lw=1, ls=':')
            mae = np.mean(abs(q[:, 4] - w['truth']))
            crps = np.mean(bench.crps(q, w['truth']))
            ax.set_title(f'{LABELS[k]}    |    MAE {mae:.3f} °C    |    CRPS {crps:.3f} °C', loc='left')
            ax.set_ylabel('Wassertemperatur (°C)')
            ax.legend(loc='upper right', ncols=4, fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.\n%H:%M'))
        axes[-1].set_xlabel('Zeit (Europe/Berlin)')
        fig.suptitle(f'96-Stunden-Prognose vom {local([ts])[0]:%d.%m.%Y %H:%M}\n'
                     f'Links: 48 h Verlauf · Rechts: {LABELS[0]} und {LABELS[1]}',
                     fontsize=13)
        fig.savefig(OUT / f'window_{i:02d}.png', dpi=140)
        plt.close(fig)
    options = ''.join(f'<option value="{i:02d}">'
                      f'{local([pd.Timestamp(str(w["reference"]))])[0]:%d.%m.%Y %H:%M}</option>'
                      for i, w in enumerate(windows))
    table = pooled[['mae', 'rmse', 'bias', 'crps', 'cov_80', 'width_80']].round(4).to_html()
    html = '''<!doctype html><html lang="de"><meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>TimesFM final gegen DUET</title><style>body{font:16px system-ui;margin:30px auto;padding:0 20px;
max-width:1400px;color:#222;background:#fff}img{width:100%;height:auto}table{border-collapse:collapse}
th,td{padding:10px;text-align:right;border-bottom:1px solid #ddd}select,button{font:inherit;padding:8px}
.note{max-width:950px;line-height:1.5}a{color:#1771f1}</style><h1>Finaler TimesFM-Satz gegen DUET</h1>
<p class="note">Ein täglicher Lauf, 96 Stunden Horizont. TimesFM final sieht tatsächlich gemessenes
zukünftiges Wetter (Orakel); DUET sind echte Produktionsprognosen. Der Luft-Replay nutzt nur
rechtzeitig archivierte
Münchner Wettervorhersagen und ist eine schlanke Referenz, nicht der finale Featuresatz.
Alle Scores auf denselben Fenstern und denselben Quantilen.</p>
<img src="overview.png" alt="Vier-Wochen-Vergleich">
<h2>Vier-Tage-Fenster</h2><label for="window">Prognosestart: </label>
<select id="window">''' + options + '''</select>
<img id="detail" src="window_00.png" alt="Quantilprognosen beider Modelle mit Messung">
<h2>Alle 96 Stunden</h2>''' + table + '''<p><a href="REPORT.md">Versuchsbericht und Einschränkungen</a></p>
<script>document.getElementById('window').addEventListener('change',e=>{
document.getElementById('detail').src='window_'+e.target.value+'.png';});</script></html>'''
    if 'Orakel' in LABELS[0]:
        html = html.replace('Wetter (Orakel); DUET sind echte Produktionsprognosen.',
                            'Wetter; auch DUET wird mit gemessenem Wetter neu gerechnet '
                            '(Orakel gegen Orakel).')
    html = html.replace('window_00.png', f'window_{len(windows) - 1:02d}.png')
    html = html.replace('</select>', '</select><script>document.getElementById("window").selectedIndex='
                        f'{len(windows) - 1};</script>')
    (OUT / 'index.html').write_text(html)
    LOG.info('Summary:\n%s', pooled.to_string())
    LOG.info('Before sensor fault:\n%s', sensitivity.to_string())


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if not args.render_only:
        run()
    render()


if __name__ == '__main__':
    main()
