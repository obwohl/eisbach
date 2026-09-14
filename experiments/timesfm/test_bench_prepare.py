"""Tests for the MLX missing-input preparation.

Run them explicitly; they are not part of the production suite:

    python3 -m pytest experiments/timesfm/test_bench_prepare.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

NAN = np.nan


def prep(target, past_only=None, past_future=None):
    return bench.prepare_inputs(np.asarray(target, dtype=np.float32),
                                None if past_only is None else np.asarray(past_only, np.float32),
                                None if past_future is None else np.asarray(past_future, np.float32),
                                force=True)


def test_it_is_a_no_op_on_torch_unless_forced(monkeypatch):
    monkeypatch.delenv("TIMESFM_BACKEND", raising=False)
    target = np.array([NAN, 1.0, 2.0], dtype=np.float32)
    out, _, _ = bench.prepare_inputs(target, None, None)
    assert out is target


def test_leading_missing_values_are_trimmed():
    out, _, _ = prep([NAN, NAN, 3.0, 4.0])
    assert out.tolist() == [3.0, 4.0]


def test_interior_gaps_are_interpolated():
    out, _, _ = prep([1.0, NAN, 3.0])
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_covariates_are_trimmed_with_the_target():
    _, past_only, past_future = prep(
        [NAN, 2.0, 3.0], past_only=[[7.0, 8.0, 9.0]], past_future=[[1.0, 2.0, 3.0, 4.0]])
    assert past_only.tolist() == [[8.0, 9.0]]
    assert past_future.tolist() == [[2.0, 3.0, 4.0]]


def test_an_all_missing_target_becomes_zero_rather_than_nan():
    out, _, _ = prep([NAN, NAN])
    assert out.tolist() == [0.0, 0.0]
    assert not np.isnan(out).any()


def test_a_stack_of_targets_is_trimmed_over_time_not_over_variates():
    """The bug this pins: ``target[first:]`` on a stack cuts away variates.

    With two variates and one leading gap, slicing as if the array were one series
    would drop the Eisbach itself and forecast whatever happened to be variate 1.
    """
    out, _, _ = prep([[NAN, 1.0, 2.0], [NAN, 10.0, 20.0]])
    assert out.shape == (2, 2)
    assert out.tolist() == [[1.0, 2.0], [10.0, 20.0]]


def test_a_stack_trims_where_every_variate_is_present():
    out, _, _ = prep([[1.0, 2.0, 3.0], [NAN, 20.0, 30.0]])
    assert out.tolist() == [[2.0, 3.0], [20.0, 30.0]]


def test_a_stack_keeps_its_covariates_aligned():
    _, past_only, _ = prep([[NAN, 1.0, 2.0], [NAN, 10.0, 20.0]],
                           past_only=[[5.0, 6.0, 7.0]])
    assert past_only.tolist() == [[6.0, 7.0]]


def test_gaps_inside_a_stack_are_interpolated_per_variate():
    out, _, _ = prep([[1.0, NAN, 3.0], [10.0, NAN, 50.0]])
    assert out.tolist() == [[1.0, 2.0, 3.0], [10.0, 30.0, 50.0]]


def test_the_caller_s_arrays_are_left_alone():
    target = np.array([NAN, 1.0, 2.0], dtype=np.float32)
    past_only = np.array([[1.0, NAN, 3.0]], dtype=np.float32)
    before = past_only.copy()
    prep(target, past_only=past_only)
    assert np.isnan(target[0])
    assert np.array_equal(past_only, before, equal_nan=True)


@pytest.mark.parametrize("shape", [(1, 5), (3, 5)])
def test_a_clean_stack_is_returned_unchanged(shape):
    target = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    out, _, _ = prep(target)
    assert out.tolist() == target.tolist()
