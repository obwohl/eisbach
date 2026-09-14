"""Guard the no-weather path and input-time alignment used by exp18."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from exp12_pairs import evaluate


def test_screen_includes_both_ends_of_confirmation_sample():
    from exp18_discharge_alone import select_anchors

    anchors = list(range(250))
    selected = select_anchors(anchors, 100)
    assert len(selected) == len(set(selected)) == 100
    assert selected[0] == 0 and selected[-1] == 249
    assert set(selected).issubset(select_anchors(anchors, 250))
    assert select_anchors(anchors[:3], 100) == anchors[:3]


def test_bare_and_past_only_share_target_without_future_leak(monkeypatch):
    monkeypatch.setenv('TIMESFM_BACKEND', 'mlx')
    idx = pd.date_range('2020-01-01', periods=104, freq='h', tz='UTC')
    target = np.arange(104, dtype=float)
    target[:2] = np.nan
    df = pd.DataFrame({'eisbach': target, 'Q': np.arange(104, dtype=float)+100}, index=idx)
    # A leading Q gap must not change the target's trim point.
    df.loc[idx[:3], 'Q'] = np.nan
    seen = []

    class Forecaster:
        def predict_batch(self, contexts, *, horizon, past_only_covariates,
                          past_future_covariates, return_quantiles):
            seen.append((contexts, past_only_covariates, past_future_covariates))
            assert horizon == 96 and return_quantiles
            for _ in contexts:
                yield SimpleNamespace(quantiles=np.zeros((96, 9)))

    for columns in ([], ['Q']):
        rows = evaluate(Forecaster(), df, [idx[7]], df.eisbach, label=str(columns),
                        past_only=columns, context=8, future=[])
        assert len(rows) == 7 and sum(r['n'] for r in rows) == 96
    np.testing.assert_array_equal(seen[0][0][0], seen[1][0][0])
    assert seen[0][2] == [None] and seen[1][2] == [None]
    assert seen[0][1] == [None]
    np.testing.assert_array_equal(seen[1][1][0], [[103, 103, 104, 105, 106, 107]])


def test_block_interval_does_not_treat_duplicate_windows_as_independent():
    from report_exp18 import interval

    values = pd.Series([-.4, .2, -.1, .3],
                       index=pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01'],
                                            utc=True))
    # Duplicating each observation in its month adds no independent information.
    replicated = pd.concat([values] * 20).sort_index()
    np.testing.assert_allclose(interval(values), interval(replicated), atol=1e-12)
