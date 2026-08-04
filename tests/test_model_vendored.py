"""Equivalence tests for the vendored inference package (``eisbach.model``).

These run the ORIGINAL ``ts_proba_cuda/run_single_forecast.py`` as a subprocess
and the new in-process path on the same synthetic input, then compare the two
forecasts element by element. They are skipped once the submodule is gone.

The checkpoint is fetched once (~10 MB) and cached in data/model/; everything else is offline.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE_ROOT = REPO_ROOT / "ts_proba_cuda"
ORIGINAL_SCRIPT = SUBMODULE_ROOT / "run_single_forecast.py"
LEGACY_CHECKPOINT = SUBMODULE_ROOT / "checkpoints" / "best_model.pt"

SERIES_ORDER = ["wassertemp", "airtemp_96", "pressure_96"]
QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

requires_submodule = pytest.mark.skipif(
    not (ORIGINAL_SCRIPT.is_file() and LEGACY_CHECKPOINT.is_file()),
    reason="ts_proba_cuda submodule is not checked out; nothing to compare against",
)

torch = pytest.importorskip("torch")


def make_synthetic_df_long(n_hours: int = 480, seed: int = 20260804) -> pd.DataFrame:
    """Deterministic long-format frame with all three channels on an hourly grid."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-06-01 00:00:00", periods=n_hours, freq="h")
    t = np.arange(n_hours, dtype=np.float64)

    signals = {
        # Water temperature: daily cycle + slow warming trend + noise.
        "wassertemp": 14.0
        + 1.5 * np.sin(2 * np.pi * t / 24.0)
        + 0.004 * t
        + rng.normal(0.0, 0.08, n_hours),
        # Air temperature (already shifted by -96h upstream).
        "airtemp_96": 18.0
        + 6.0 * np.sin(2 * np.pi * (t - 6) / 24.0)
        + 2.0 * np.sin(2 * np.pi * t / (24 * 7))
        + rng.normal(0.0, 0.5, n_hours),
        # Air pressure (already shifted by -96h upstream).
        "pressure_96": 1013.0
        + 8.0 * np.sin(2 * np.pi * t / (24 * 3.5))
        + rng.normal(0.0, 0.8, n_hours),
    }

    frames = [
        pd.DataFrame({"date": index, "cols": name, "data": values})
        for name, values in signals.items()
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def df_long() -> pd.DataFrame:
    return make_synthetic_df_long()


@pytest.fixture(scope="module")
def in_process_forecast(df_long: pd.DataFrame) -> pd.DataFrame:
    from eisbach.model import forecast, load_model, long_to_wide, resolve_checkpoint

    checkpoint = resolve_checkpoint()
    model, config = load_model(checkpoint, device="cpu")
    return forecast(model, config, long_to_wide(df_long))


@pytest.fixture(scope="module")
def original_forecast(tmp_path_factory, df_long: pd.DataFrame) -> pd.DataFrame:
    if not (ORIGINAL_SCRIPT.is_file() and LEGACY_CHECKPOINT.is_file()):
        pytest.skip("ts_proba_cuda submodule is not checked out")

    tmp_path = tmp_path_factory.mktemp("original_forecast")
    data_file = tmp_path / "df_long.csv"
    output_csv = tmp_path / "inference.csv"
    df_long.to_csv(data_file, index=False)

    # Force CPU so both paths run on the same backend.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

    result = subprocess.run(
        [
            sys.executable,
            str(ORIGINAL_SCRIPT),
            "--checkpoint",
            str(LEGACY_CHECKPOINT),
            "--data-file",
            str(data_file),
            "--output-csv",
            str(output_csv),
        ],
        cwd=str(SUBMODULE_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"original script failed:\n{result.stdout}\n{result.stderr}"
    # The original swallows errors and still exits 0, so check the artefact too.
    assert output_csv.is_file(), f"original script produced no CSV:\n{result.stdout}"

    return pd.read_csv(output_csv, parse_dates=[0], index_col=0)


# --------------------------------------------------------------------------
# Contract tests (do not need the submodule)
# --------------------------------------------------------------------------


def test_output_contract(in_process_forecast: pd.DataFrame, df_long: pd.DataFrame):
    df = in_process_forecast
    expected_columns = [f"{var}_q{q}" for var in SERIES_ORDER for q in QUANTILES]
    assert list(df.columns) == expected_columns
    assert len(df) == 96

    last_input = pd.to_datetime(df_long["date"]).max()
    assert df.index[0] == last_input + pd.Timedelta(hours=1)
    assert pd.infer_freq(df.index) == "h"
    assert np.isfinite(df.to_numpy()).all()


def test_quantiles_are_monotonic(in_process_forecast: pd.DataFrame):
    for var in SERIES_ORDER:
        block = in_process_forecast[[f"{var}_q{q}" for q in QUANTILES]].to_numpy()
        assert (np.diff(block, axis=1) >= -1e-6).all(), f"{var} quantiles are not monotonic"


def test_too_short_input_raises(df_long: pd.DataFrame):
    from eisbach.model import forecast, load_model, long_to_wide, resolve_checkpoint

    checkpoint = resolve_checkpoint()
    model, config = load_model(checkpoint, device="cpu")
    short = long_to_wide(df_long).iloc[:100]
    with pytest.raises(ValueError, match="Not enough data"):
        forecast(model, config, short)


def test_checksum_mismatch_is_refused(tmp_path):
    from eisbach.model.checkpoint import ChecksumError, resolve_checkpoint

    bogus = tmp_path / "best_model.pt"
    bogus.write_bytes(b"not a checkpoint")
    with pytest.raises(ChecksumError):
        resolve_checkpoint(bogus)


# --------------------------------------------------------------------------
# Equivalence test (needs the submodule)
# --------------------------------------------------------------------------


@requires_submodule
def test_vendored_matches_original(
    original_forecast: pd.DataFrame, in_process_forecast: pd.DataFrame
):
    old, new = original_forecast, in_process_forecast

    assert list(old.columns) == list(new.columns)
    assert old.index.equals(new.index)

    a = old.to_numpy(dtype=np.float64)
    b = new.to_numpy(dtype=np.float64)

    abs_diff = np.abs(a - b)
    max_abs = float(abs_diff.max())
    max_rel = float((abs_diff / np.maximum(np.abs(a), 1e-12)).max())
    print(f"\nmax abs diff = {max_abs:.3e}   max rel diff = {max_rel:.3e}")

    # The residual is entirely the original's CSV serialisation: it dumps
    # float32 values, whose repr keeps ~7 significant digits (rel. ~6e-8).
    # See test_vendored_is_bit_identical_through_csv for the exact check.
    np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-6)


@requires_submodule
def test_vendored_is_bit_identical_through_csv(
    original_forecast: pd.DataFrame, in_process_forecast: pd.DataFrame, tmp_path: Path
):
    """The in-process result, serialised the same way, reproduces the original CSV.

    The original pipeline's observable output was a CSV of float32 values. Passing
    the new result through the identical ``to_csv`` -> ``read_csv`` round-trip and
    getting an exact match proves the underlying tensors are bit-identical.
    """
    assert in_process_forecast.to_numpy().dtype == np.float32

    round_tripped_path = tmp_path / "vendored.csv"
    in_process_forecast.to_csv(round_tripped_path)
    round_tripped = pd.read_csv(round_tripped_path, parse_dates=[0], index_col=0)

    pd.testing.assert_frame_equal(round_tripped, original_forecast, check_exact=True)
