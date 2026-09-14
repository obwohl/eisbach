"""Checks for exp17's paired inference and temporal eligibility."""
import numpy as np
import pandas as pd
from exp17_selection import BASE, CANDIDATES, WEATHER, anchors_for, comparisons, subset


def test_paired_ratio_preserves_constant_relative_effect_across_blocks():
    rows = []
    for i, ts in enumerate(pd.date_range('2020-01-01', periods=24, freq='MS', tz='UTC')):
        for label, factor in [('0', 1.), ('1', .99)]:
            rows.append(dict(label=label, reference_time=ts, n=96,
                             mae=(i + 1) * factor, crps=(i + 1) * factor / 2))
    for block in [1, 3, 6]:
        result = comparisons(pd.DataFrame(rows), block=block)
        for rec in result:
            assert np.isclose(rec['pct'], -1)
            assert np.allclose(rec['ci'], [-1, -1])


def test_eligibility_does_not_require_future_water_or_impute_truth():
    idx = pd.date_range('2020-01-01', periods=8856, freq='h', tz='UTC')
    df = pd.DataFrame(1., index=idx, columns=['eisbach', *WEATHER, *BASE, *CANDIDATES])
    df.loc[idx[8760:], CANDIDATES] = np.nan
    assert anchors_for(df) == [idx[8759]]
    df.loc[idx[-1], 'eisbach'] = np.nan
    assert anchors_for(df) == []


def test_fixed_bad_toelz_present_in_all_eight_subsets():
    arms = [subset(mask) for mask in range(8)]
    assert all(a[0] == 'isar_toelz' for a in arms)
    assert len({tuple(a) for a in arms}) == 8
    assert arms[7] == [*BASE, *CANDIDATES]
