"""Run explicitly; experiment checks are outside the production test suite."""
import numpy as np
import pandas as pd
import pytest
from exp11_resolution import FUTURE, PAST_ONLY, TARGET, evaluate, model_inputs


def test_missing_inputs_are_filled_without_changing_truth():
    ctx = pd.DataFrame({c: [np.nan, 1.0, np.nan, 5.0]
                        for c in [TARGET, *PAST_ONLY, *FUTURE]})
    hor = pd.DataFrame({c: [99.0, 100.0] for c in [TARGET, *PAST_ONLY, *FUTURE]})
    original_ctx, original_hor = ctx.copy(deep=True), hor.copy(deep=True)
    target, po, pf = model_inputs(ctx, hor)
    np.testing.assert_array_equal(target, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(po, np.tile([1.0, 3.0, 5.0], (4, 1)))
    np.testing.assert_array_equal(pf, np.tile([1.0, 3.0, 5.0, 99.0, 100.0], (2, 1)))
    pd.testing.assert_frame_equal(ctx, original_ctx)
    pd.testing.assert_frame_equal(hor, original_hor)


def test_all_missing_and_edge_inputs_match_torch_convention():
    ctx = pd.DataFrame({c: [np.nan, np.nan, np.nan]
                        for c in [TARGET, *PAST_ONLY, *FUTURE]})
    ctx[PAST_ONLY[0]] = [np.nan, 7.0, np.nan]
    hor = pd.DataFrame({c: [np.nan] for c in FUTURE})
    target, po, pf = model_inputs(ctx, hor)
    np.testing.assert_array_equal(target, np.zeros(3))
    np.testing.assert_array_equal(po[0], [7.0, 7.0, 7.0])
    np.testing.assert_array_equal(po[1:], np.zeros((3, 3)))
    np.testing.assert_array_equal(pf, np.zeros((2, 4)))


def test_nonfinite_forecast_fails_before_scoring():
    class BrokenForecaster:
        def predict_batch(self, contexts, **kwargs):
            class Output:
                quantiles = np.full((96, 9), np.nan)
            return [Output() for _ in contexts]

    df = pd.DataFrame(1.0, index=pd.date_range("2024-01-01", periods=400,
                                             freq="15min", tz="UTC"),
                      columns=[TARGET, *PAST_ONLY, *FUTURE])
    with pytest.raises(ValueError, match="non-finite forecast quantiles"):
        evaluate(BrokenForecaster(), df, [df.index[4]], label="broken",
                 steps_per_hour=1, context_steps=2)
