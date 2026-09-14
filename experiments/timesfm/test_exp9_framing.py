"""Role assignment must not change the model's input context."""
import numpy as np
from exp9_targets import prepare_framing


def test_roles_preserve_identical_values_with_leading_upstream_gap(monkeypatch):
    monkeypatch.setenv('TIMESFM_BACKEND', 'mlx')
    eisbach = np.array([np.nan, 10, 11, 12, 13], dtype=np.float32)
    upstream = np.array([[np.nan, np.nan, 8, 9, 10], [7, 8, 9, 10, 11]], dtype=np.float32)
    weather = np.array([[1, 2, 3, 4, 5, 6, 7]], dtype=np.float32)
    for count in (1, 2):
        cov_t, cov_po, cov_pf = prepare_framing(eisbach, upstream[:count], weather)
        tgt_t, tgt_po, tgt_pf = prepare_framing(np.vstack([eisbach, upstream[:count]]), None, weather)
        np.testing.assert_array_equal(np.vstack([cov_t, cov_po]), tgt_t)
        np.testing.assert_array_equal(cov_pf, tgt_pf)
        assert tgt_po is None
        assert tgt_t.shape == (count + 1, 4)
        assert np.isfinite(tgt_t).all()
