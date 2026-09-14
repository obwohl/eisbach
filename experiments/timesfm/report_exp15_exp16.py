"""Reproduce paired tables, dependence sensitivities and simulation uncertainty."""
import json
from pathlib import Path

import bench
import numpy as np
import pandas as pd
from exp15_rain import RAW_24H, load
from report_exp9_exp14 import load_scores, monthly_table, paired_table, table

OUT = Path(__file__).with_name('REPORT_exp15_exp16.md')


def blocked(values, days=7, draws=8000):
    """Shared draws of fixed calendar blocks, retaining all windows in each block."""
    groups = (values.index - values.index.min()).days // days
    sums = values.groupby(groups).sum().to_numpy()
    counts = values.groupby(groups).size().to_numpy()
    rng = np.random.default_rng(0)
    ix = rng.integers(len(counts), size=(draws, len(counts)))
    return sums[ix].sum(axis=1) / counts[ix].sum(axis=1)[:, None]


def block_table(scores, reference, days=7):
    rows = []
    for label in scores.label.unique():
        if label == reference:
            continue
        row = [label]
        for metric in ('mae', 'crps'):
            v = bench.per_run(scores, metric)[[reference, label]].dropna()
            boot = blocked(v, days)
            ci = 100 * np.percentile(boot[:, 1] - boot[:, 0], [2.5, 97.5]) / v[reference].mean()
            row.append(f'[{ci[0]:+.3f}, {ci[1]:+.3f}]')
        rows.append(row)
    return table([f'Variant vs {reference}', f'MAE % {days}d block CI',
                  f'CRPS % {days}d block CI'], rows)


def advantage(scores, base, oracle, replay):
    rows = []
    for metric in ('mae', 'crps'):
        v = bench.per_run(scores, metric)[[base, oracle, replay]].dropna()
        b, o, r = v.mean()
        bs = blocked(v)
        cost = bs[:, 2] - bs[:, 1]
        denominator = bs[:, 0] - bs[:, 1]
        shrink = 100 * cost / denominator
        lo, hi = np.percentile(cost, [2.5, 97.5])
        slo, shi = np.percentile(shrink, [2.5, 97.5])
        rows.append([metric, f'{b:.6f}', f'{o:.6f}', f'{r:.6f}',
                     f'{r-o:+.6f} [{lo:+.6f}, {hi:+.6f}]',
                     f'{100*(r-o)/(b-o):.2f}% [{slo:.2f}, {shi:.2f}]',
                     f'{100*np.mean(denominator<=0):.2f}%'])
    return table(['Metric', base, oracle, replay, 'Replay cost °C [7d CI]',
                  'Lost oracle advantage % [7d CI]', 'Bootstrap nonpositive advantage'], rows)


def main():
    screen = load_scores('exp15_rain')
    full = pd.concat([load_scores('exp15_confirm'), load_scores('exp15_hourly_confirm')])
    scores = load_scores('exp16_audited')
    parts = ['## Paired scores and intervals\n',
             'Negative deltas favour the named variant. Ordinary paired bootstrap: '
             '8,000 shared draws, 95% percentile intervals; these assume independent '
             'windows and are secondary for the overlapping replay. Block intervals '
             'resample all windows of a calendar block together. No multiplicity '
             'adjustment; neither screen confirmation on the same windows nor '
             'retrospective yearly comparisons are held-out validation. Absolute '
             'scores pool observed hours; percent deltas average per-window scores. '
             'CRPS integrates only deciles 0.1–0.9.\n',
             '### exp15, screen 1024 h\n', paired_table(screen, 'ohne Regen'),
             '### exp15, confirmation 8760 h\n', paired_table(full, 'ohne Regen'),
             monthly_table(full, 'ohne Regen', [x for x in full.label.unique() if x != 'ohne Regen'])]
    for ref, labels in [
        ('Mittel, stündlich', ['Mittel, 24h-Summe (gesetzt)', 'vier Rohstationen, stündlich']),
        ('vier Rohstationen, stündlich', ['vier Rohstationen, je 24h-Summe']),
        ('Mittel, 24h-Summe (gesetzt)', ['vier Rohstationen, stündlich',
                                        'vier Rohstationen, je 24h-Summe']),
    ]:
        parts += ['### exp15 direct construction contrast, 8760 h\n', paired_table(full, ref, labels),
                  monthly_table(full, ref, labels)]
    data = load()
    incomplete = []
    for ts in sorted(full.reference_time.unique()):
        pos = data.index.get_loc(ts)
        if data.iloc[pos + 1:pos + 97][RAW_24H].isna().any().any():
            incomplete.append(ts)
    parts += ['### Accumulation sensitivity: complete future rolling sums only\n',
              f'Excluded from this diagnostic only: {incomplete}. Main tables retain every window.\n',
              paired_table(full[~full.reference_time.isin(incomplete)],
                           'vier Rohstationen, stündlich', ['vier Rohstationen, je 24h-Summe'])]
    for year, group in full.groupby(full.reference_time.dt.year):
        parts += [f'### exp15 year {year}\n']
        if group.reference_time.nunique() < 2:
            pooled = bench.pool(group)
            parts += ['One window: no inferential interval can be estimated.\n',
                      table(['Variant', 'MAE °C', 'CRPS °C'],
                            [[r.label, f'{r.mae:.6f}', f'{r.crps:.6f}']
                             for r in pooled.itertuples()])]
        else:
            parts += [paired_table(group, 'ohne Regen')]
    rows = []
    for year in sorted(full.reference_time.dt.year.unique()):
        g = full[full.reference_time.dt.year != year]
        for label in ['Mittel, 24h-Summe (gesetzt)', 'vier Rohstationen, stündlich']:
            changes = []
            for m in ('mae', 'crps'):
                v = bench.per_run(g, m)
                changes.append(f'{100*(v[label]-v["ohne Regen"]).mean()/v["ohne Regen"].mean():+.3f}')
            rows.append([year, label, *changes])
    parts += ['### exp15 leave-one-year-out\n',
              table(['Omitted year', 'Variant', 'MAE %', 'CRPS %'], rows),
              '### exp16 full replay sample, 8760 h\n', paired_table(scores, 'Ohne Wetter'),
              '### exp16 real Munich weather cost\n',
              paired_table(scores[scores.label.isin(['Muenchen Orakel', 'Muenchen Replay'])],
                           'Muenchen Orakel')]
    for days in (4, 7, 14):
        parts += [block_table(scores[scores.label.isin(['Muenchen Orakel', 'Muenchen Replay'])],
                              'Muenchen Orakel', days)]
    parts += [advantage(scores, 'Ohne Wetter', 'Muenchen Orakel', 'Muenchen Replay'),
              '### exp16 conditional cost with southern oracle retained\n',
              paired_table(scores[scores.label.isin(['Zwei Luft Orakel',
                                                     'Muenchen Replay + Sued Orakel'])],
                           'Zwei Luft Orakel'),
              block_table(scores[scores.label.isin(['Zwei Luft Orakel',
                                                    'Muenchen Replay + Sued Orakel'])],
                          'Zwei Luft Orakel'),
              advantage(scores, 'Ohne Wetter', 'Zwei Luft Orakel', 'Muenchen Replay + Sued Orakel')]
    for month, group in scores.groupby(scores.reference_time.dt.strftime('%Y-%m')):
        parts += [f'### exp16 {month}\n',
                  paired_table(group[group.label.isin(['Muenchen Orakel', 'Muenchen Replay'])],
                               'Muenchen Orakel')]
    sim = [load_scores(f'exp16_sim_{i:02d}') for i in range(20)]
    anchors = sorted(sim[0].reference_time.unique())
    subset = scores[scores.reference_time.isin(anchors)]
    parts += [f'### Non-overlapping 96h subset ({len(anchors)} windows), actual replay\n',
              paired_table(subset[subset.label.isin(['Muenchen Orakel', 'Muenchen Replay'])],
                           'Muenchen Orakel')]
    rows = []
    for method in ('block24', 'trace96'):
        for metric in ('mae', 'crps'):
            base = bench.per_run(subset, metric)['Zwei Luft Orakel']
            partial = bench.per_run(subset, metric)['Muenchen Replay + Sued Orakel']
            matrix = np.stack([bench.per_run(s, metric)[method].reindex(base.index).to_numpy()
                               for s in sim])
            delta = matrix - base.to_numpy()[None, :]
            means = delta.mean(axis=1)
            rng = np.random.default_rng(0)
            # Resample windows and simulation realizations separately. No 20x inflation of N.
            boots = []
            for _ in range(8000):
                wi = rng.integers(len(base), size=len(base))
                di = rng.integers(20, size=20)
                boots.append(delta[np.ix_(di, wi)].mean())
            lo, hi = 100 * np.percentile(boots, [2.5, 97.5]) / base.mean()
            rows.append([method, metric, len(base), f'{matrix.mean():.6f}',
                         f'{100*delta.mean()/base.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]',
                         f'{100*means.std(ddof=1)/base.mean():.3f}',
                         f'{matrix.mean()-partial.mean():+.6f}'])
    parts += ['### Simulation on that same subset: 20 draws per method\n',
              'Intervals resample windows and Monte Carlo realizations independently. They '
              'are conditional on the fixed empirical donor pool and the unverified '
              'Munich-to-south error transfer, not confidence in real southern forecast skill.\n',
              table(['Method', 'Metric', 'Windows', 'Score °C', 'Δ vs two-air oracle % [95% CI]',
                     'Across-draw Δ SD (percentage points)', 'Cost vs partial replay °C'], rows)]
    solar = load_scores('exp16_solar')
    solar_main = scores[scores.reference_time.isin(solar.reference_time.unique()) &
                        scores.label.isin(['Muenchen Orakel', 'Muenchen Replay'])]
    solar = pd.concat([solar, solar_main])
    parts += [f'### Munich solar, matched {solar.reference_time.nunique()} windows\n',
              paired_table(solar, 'Muenchen Replay'),
              block_table(solar, 'Muenchen Replay'),
              paired_table(solar[solar.label.isin(['Luft Solar Orakel', 'Luft Solar Replay'])],
                           'Luft Solar Orakel'),
              block_table(solar[solar.label.isin(['Luft Solar Orakel', 'Luft Solar Replay'])],
                          'Luft Solar Orakel')]
    parts += [(bench.CACHE / 'exp16_diagnostics.md').read_text()]
    marker = '\n<!-- generated tables -->\n'
    narrative = OUT.read_text().split(marker)[0] if OUT.exists() else ''
    OUT.write_text(narrative + marker + '\n'.join(parts))
    summary = {name: bench.pool(s)[['label', 'mae', 'crps']].to_dict('records')
               for name, s in [('exp15', full), ('exp16', scores), ('solar', solar)]}
    (bench.CACHE / 'exp15_exp16_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
