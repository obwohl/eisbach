"""Render exp17's measured comparisons without rerunning inference."""
import json
from pathlib import Path

import bench
import pandas as pd
from exp17_selection import comparisons

NAMES = {'0': 'Fixed baseline', '1': '+ Lenggries', '2': '+ Puppling',
         '3': '+ Lenggries + Puppling', '4': '+ Beuerberg',
         '5': '+ Lenggries + Beuerberg', '6': '+ Puppling + Beuerberg', '7': '+ all three'}


def interval(rec):
    return f"{rec['pct']:+.2f}% [{rec['ci'][0]:+.2f}, {rec['ci'][1]:+.2f}]"


def main():
    sections = []
    for ctx in [1024, 8760]:
        path = bench.CACHE / f'exp17_fixed_{ctx}.csv'
        scores = pd.read_csv(path, parse_dates=['reference_time'], dtype={'label': str})
        result = json.loads(path.with_name(path.stem + '_summary.json').read_text())
        sections += [f'## {ctx}-hour context', '',
                     '| Added inputs | MAE (°C) | MAE change [95% CI] | CRPS change [95% CI] |',
                     '|---|---:|---:|---:|']
        pooled = bench.pool(scores).set_index('label')
        sections.append(f"| Fixed baseline | {pooled.loc['0', 'mae']:.6f} | — | — |")
        for label in sorted(set(scores.label) - {'0'}):
            a, b = [next(r for r in result['block_3'] if r['label'] == label and r['metric'] == m)
                    for m in ['mae', 'crps']]
            sections.append(f"| {NAMES[label]} | {a['value']:.6f} | {interval(a)} | {interval(b)} |")
        if ctx == 1024:
            continue
        sections += ['', '## Conditional additions at 8760 hours', '',
                     '| Addition | MAE change [95% CI] | CRPS change [95% CI] |',
                     '|---|---:|---:|']
        for ref, label, name in [('4', '6', 'Puppling beside Beuerberg'),
                                 ('2', '6', 'Beuerberg beside Puppling'),
                                 ('6', '7', 'Lenggries beside Puppling + Beuerberg')]:
            recs = result['conditional'][ref]
            a, b = [next(r for r in recs if r['label'] == label and r['metric'] == m)
                    for m in ['mae', 'crps']]
            sections.append(f'| {name} | {interval(a)} | {interval(b)} |')
        sections += ['', '## Annual MAE checks at 8760 hours', '',
                     'Three-origin block intervals within each year; '
                     'fewer than eight origins: no interval.', '',
                     '| Year | Origins | Lenggries vs base | Puppling vs base | Beuerberg vs base | '
                     'Puppling + Beuerberg vs base | All three vs pair |',
                     '|---|---:|---:|---:|---:|---:|---:|']
        for year, group in scores.groupby(scores.reference_time.dt.year):
            n = group.reference_time.nunique()
            items = []
            for ref, label in [('0', '1'), ('0', '2'), ('0', '4'), ('0', '6'), ('6', '7')]:
                rec = next(r for r in comparisons(group, reference=ref)
                           if r['label'] == label and r['metric'] == 'mae')
                items.append(interval(rec) if n >= 8 else f"{rec['pct']:+.2f}%")
            sections.append(f"| {year} | {n} | " + ' | '.join(items) + ' |')
        sections += ['', '## Lead buckets at 8760 hours', '',
                     '| Hours | Lenggries MAE change | Puppling MAE change | Beuerberg MAE change | '
                     'Puppling + Beuerberg MAE change |', '|---|---:|---:|---:|---:|']
        for lo, hi in bench.FINE_BUCKETS:
            recs = result['buckets'][str(lo)]
            vals = [next(r for r in recs if r['label'] == lab and r['metric'] == 'mae')
                    for lab in ['1', '2', '4', '6']]
            sections.append(f'| {lo + 1}–{hi} | ' + ' | '.join(interval(r) for r in vals) + ' |')
        sections += ['', '## Block-length sensitivity at 8760 hours', '',
                     '| Block origins | Lenggries vs base | Puppling vs base | Beuerberg vs base | '
                     'Pair vs base | Lenggries vs pair |', '|---|---:|---:|---:|---:|---:|']
        for block in [1, 3, 6]:
            vals = []
            for ref, lab in [('0', '1'), ('0', '2'), ('0', '4'), ('0', '6'), ('6', '7')]:
                rec = next(r for r in comparisons(scores, reference=ref, block=block)
                           if r['label'] == lab and r['metric'] == 'mae')
                vals.append(interval(rec))
            sections.append(f'| {block} | ' + ' | '.join(vals) + ' |')
        sections += ['', '## Leave-one-year-out MAE at 8760 hours', '',
                     '| Omitted year | Lenggries vs base | Puppling vs base | '
                     'Beuerberg vs base | Pair vs base |',
                     '|---|---:|---:|---:|---:|']
        for year in sorted(scores.reference_time.dt.year.unique()):
            sub = scores[scores.reference_time.dt.year != year]
            recs = comparisons(sub)
            vals = [next(r['pct'] for r in recs if r['label'] == lab and r['metric'] == 'mae')
                    for lab in ['1', '2', '4', '6']]
            sections.append(f'| {year} | ' + ' | '.join(f'{v:+.2f}%' for v in vals) + ' |')
    Path(bench.CACHE / 'exp17_tables.md').write_text('\n'.join(sections) + '\n')


if __name__ == '__main__':
    main()
