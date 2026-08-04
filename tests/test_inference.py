"""Tests for the three-track backtest resolution.

The model itself is stubbed out — these tests are about *which source* a backtest comes
from and how it is labelled, not about the numbers. See tests/test_model_vendored.py for
the model, and tests/test_archive.py for the storage layer.
"""

import numpy as np
import pandas as pd
import pytest

from eisbach import archive, inference

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
CHANNELS = ["wassertemp", "airtemp_96", "pressure_96"]

LAST_OBSERVATION = pd.Timestamp("2026-05-20 09:00", tz="UTC")


def make_long_frame(last=LAST_OBSERVATION, history_hours=500):
    """A df_long shaped like prepare_data's output, with covariates running ahead."""
    index = pd.date_range(last - pd.Timedelta(hours=history_hours), last, freq="1h")
    rng = np.random.default_rng(20260804)
    rows = []
    for channel, base in zip(CHANNELS, [15.0, 18.0, 1013.0], strict=True):
        rows.append(pd.DataFrame({
            "date": index,
            "cols": channel,
            "data": base + rng.normal(0, 0.5, len(index)),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["cols"] = pd.Categorical(df["cols"], categories=CHANNELS, ordered=True)
    return df


def make_weather(last=LAST_OBSERVATION, history_hours=500, ahead_hours=200):
    index = pd.date_range(last - pd.Timedelta(hours=history_hours),
                          last + pd.Timedelta(hours=ahead_hours), freq="1h")
    return pd.DataFrame(
        {"lufttemperatur_c": np.linspace(10, 20, len(index)),
         "pressure": np.linspace(1000, 1020, len(index)),
         "niederschlag_mm": 0.0},
        index=index,
    )


def make_water(last=LAST_OBSERVATION, history_hours=500):
    index = pd.date_range(last - pd.Timedelta(hours=history_hours), last, freq="1h")
    return pd.DataFrame({"timestamp": index, "wassertemp": np.linspace(14, 16, len(index))})


def make_forecast_frame(reference_time, periods=96, base=15.0):
    index = pd.date_range(pd.Timestamp(reference_time) + pd.Timedelta(hours=1),
                          periods=periods, freq="1h")
    data = {}
    for channel, offset in zip(CHANNELS, [0.0, 3.0, 998.0], strict=True):
        for q in QUANTILES:
            data[f"{channel}_q{q}"] = base + offset + (q - 0.5) * 2
    return pd.DataFrame(data, index=index)


@pytest.fixture
def stub_model(monkeypatch):
    """Replace the model with something deterministic and instant.

    Records every cutoff it was asked to predict from, so the tests can assert how many
    model runs a given archive state actually costs.
    """
    calls = []

    def fake_load_model(*args, **kwargs):
        return object(), object()

    def fake_forecast(model, config, df_wide):
        cutoff = df_wide.index.max()
        calls.append(cutoff)
        return make_forecast_frame(cutoff)

    monkeypatch.setattr(inference, "load_model", fake_load_model)
    monkeypatch.setattr(inference, "forecast", fake_forecast)
    monkeypatch.setattr(inference, "long_to_wide", lambda df: df.pivot(
        index="date", columns="cols", values="data")[CHANNELS])
    return calls


@pytest.fixture
def run(tmp_path, stub_model, monkeypatch):
    """Run the pipeline against a throwaway archive, in a throwaway working directory."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "archive"

    def _run():
        return inference.run_inference(
            make_long_frame(), make_weather(), make_water(), archive_root=root,
        )

    _run.root = root
    _run.calls = stub_model
    return _run


def test_main_forecast_is_archived_as_live(run):
    df_inference, _ = run()

    stored = archive.read_forecasts(root=run.root, kinds=[archive.KIND_LIVE])
    assert not stored.empty
    assert (stored["covariate_source"] == archive.COVARIATE_DWD_FORECAST).all()
    assert len(df_inference) == 96


def test_backtests_fall_back_to_oracle_on_an_empty_archive(run):
    """Nothing stored yet, so every backtest has to use observed weather."""
    _, backtests = run()

    assert set(backtests) == set(inference.BACKTEST_OFFSETS_HOURS)
    for backtest in backtests.values():
        assert backtest.kind == archive.KIND_ORACLE
        assert backtest.covariate_source == archive.COVARIATE_DWD_OBSERVED
        assert not backtest.is_honest


def test_oracle_backtests_are_labelled_as_using_perfect_weather(run):
    _, backtests = run()
    label = backtests[96].label
    assert "oracle" in label
    assert "perfect weather" in label


def test_a_stored_live_forecast_is_reused_instead_of_recomputed(run):
    """The cheap and honest path: we already ran then, so just look it up."""
    reference = LAST_OBSERVATION - pd.Timedelta(hours=96)
    archive.write_forecast(
        make_forecast_frame(reference, base=42.0),
        reference_time=reference,
        kind=archive.KIND_LIVE,
        covariate_source=archive.COVARIATE_DWD_FORECAST,
        root=run.root,
    )

    _, backtests = run()

    assert backtests[96].kind == archive.KIND_LIVE
    assert backtests[96].is_honest
    # 42.0 is the marker: it came out of the archive, not the stub model.
    assert backtests[96].forecast["wassertemp_q0.5"].iloc[0] == pytest.approx(42.0)


def test_reusing_a_stored_forecast_saves_a_model_run(run):
    reference = LAST_OBSERVATION - pd.Timedelta(hours=96)
    archive.write_forecast(
        make_forecast_frame(reference),
        reference_time=reference,
        kind=archive.KIND_LIVE,
        covariate_source=archive.COVARIATE_DWD_FORECAST,
        root=run.root,
    )

    run()

    # 1 main forecast + 2 recomputed backtests; the third came from the archive.
    assert len(run.calls) == 3


def test_a_weather_snapshot_enables_an_honest_replay(run):
    """With the DWD forecast as it was issued, the backtest becomes honest again."""
    reference = LAST_OBSERVATION - pd.Timedelta(hours=192)
    snapshot = pd.DataFrame({
        "timestamp": pd.date_range(reference, periods=120, freq="1h"),
        "temperature": np.linspace(12, 22, 120),
        "pressure_msl": np.linspace(1005, 1015, 120),
        "precipitation": 0.0,
    })
    archive.write_weather_snapshot(snapshot, archived_at=reference, root=run.root)

    _, backtests = run()

    assert backtests[192].kind == archive.KIND_REPLAY
    assert backtests[192].covariate_source == archive.COVARIATE_DWD_ARCHIVED
    assert backtests[192].is_honest
    assert backtests[96].kind == archive.KIND_ORACLE, "other offsets are unaffected"


def test_a_later_weather_snapshot_is_not_used(run):
    """A snapshot issued after the reference time knows things the run could not have.

    Using it would leak the future — less than an oracle backtest does, but in the same
    direction — so the one-sided window must reject it.
    """
    reference = LAST_OBSERVATION - pd.Timedelta(hours=192)
    snapshot = pd.DataFrame({
        "timestamp": pd.date_range(reference, periods=120, freq="1h"),
        "temperature": np.linspace(12, 22, 120),
        "pressure_msl": np.linspace(1005, 1015, 120),
    })
    archive.write_weather_snapshot(
        snapshot, archived_at=reference + pd.Timedelta(hours=3), root=run.root,
    )

    _, backtests = run()

    assert backtests[192].kind == archive.KIND_ORACLE


def test_a_stale_weather_snapshot_is_not_used(run):
    """Older than the window is also rejected, just less dangerously."""
    reference = LAST_OBSERVATION - pd.Timedelta(hours=192)
    snapshot = pd.DataFrame({
        "timestamp": pd.date_range(reference, periods=120, freq="1h"),
        "temperature": np.linspace(12, 22, 120),
        "pressure_msl": np.linspace(1005, 1015, 120),
    })
    archive.write_weather_snapshot(
        snapshot,
        archived_at=reference - pd.Timedelta(hours=inference.SNAPSHOT_MAX_AGE_HOURS + 1),
        root=run.root,
    )

    _, backtests = run()

    assert backtests[192].kind == archive.KIND_ORACLE


def test_recomputed_backtests_are_archived_with_their_kind(run):
    run()

    stored = archive.read_forecasts(root=run.root)
    oracle = stored[stored["kind"] == archive.KIND_ORACLE]
    assert oracle["reference_time"].nunique() == len(inference.BACKTEST_OFFSETS_HOURS)
    assert (oracle["covariate_source"] == archive.COVARIATE_DWD_OBSERVED).all()


def test_the_run_snapshots_weather_and_observations(run):
    """Every run must leave enough behind to replay itself later."""
    run()

    assert list((run.root / "weather").glob("*.csv")), "no weather snapshot written"
    assert list((run.root / "observations").glob("*.csv")), "no observations written"


def test_readable_csv_is_written_in_local_time(run, tmp_path):
    run()

    written = pd.read_csv(tmp_path / inference.MAIN_CSV_NAME, index_col=0)
    assert len(written) == 96
    # Local wall-clock strings, no timezone offset suffix.
    assert not written.index[0].endswith("+00:00")


def test_no_observations_is_a_hard_error(run, tmp_path):
    empty = make_long_frame()
    empty.loc[empty["cols"] == "wassertemp", "data"] = np.nan

    with pytest.raises(ValueError, match="no water temperature"):
        inference.run_inference(empty, make_weather(), make_water(),
                                archive_root=tmp_path / "archive")
