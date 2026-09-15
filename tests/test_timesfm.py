"""The candidate's production path: what it refuses to forecast on, and what it archives.

Nothing here loads the checkpoint or touches the network. The model itself is the one
part that is not this repository's to test — it is a fixed set of weights, pinned by
revision — so `predict` is exercised against a stand-in that returns the shape TimesFM
returns, and everything around it is tested for real.
"""
import numpy as np
import pandas as pd
import pytest

from eisbach import timesfm
from eisbach.archive import read_forecasts
from eisbach.covariates import SPECS, future_weather, write_hourly


class Stub:
    """Stands in for the forecaster. Records what it was handed."""

    def __init__(self, quantiles=None):
        self.quantiles = quantiles
        self.seen = {}

    def predict(self, target, *, horizon, past_only_covariates, past_future_covariates,
                return_quantiles):
        self.seen = dict(target=target, past_only=past_only_covariates,
                         past_future=past_future_covariates, horizon=horizon)
        if self.quantiles is not None:
            values = self.quantiles
        else:
            rising = np.arange(len(timesfm.DECILES), dtype=float) * 0.1
            values = 15.0 + rising + np.zeros((horizon, 1))
        return type("Result", (), {"quantiles": values})()


def hourly(name, index, values):
    """Write a series into a covariate store the way an ingest would."""
    series = pd.Series(values, index=index, dtype=float)
    return pd.DataFrame({"raw_value": series, "raw_min": series, "raw_max": series,
                         "samples": series.notna().astype(int),
                         "duplicate_count": 0, "conflict": False})


ANCHOR = pd.Timestamp("2026-09-01 00:00", tz="UTC")


def build_store(tmp_path, *, holes=None, spikes=None):
    """A year of plausible history for every series the candidate reads.

    `holes` maps a series to a slice of hours that were never reported. They have to be
    absent from the start: `write_hourly` never blanks a value it already holds, which is
    the right rule for an archive and makes a gap impossible to add afterwards.
    """
    holes, spikes = holes or {}, spikes or {}
    index = pd.date_range(ANCHOR - pd.Timedelta(hours=timesfm.CONTEXT_HOURS - 1), ANCHOR, freq="h")
    hours = np.arange(len(index))
    root = tmp_path / "covariates"
    for name in [timesfm.TARGET, *timesfm.PAST_ONLY, *timesfm.KNOWN_FUTURE]:
        low, high = SPECS[name].bounds
        middle = min(max(10.0, low + 1), high - 1)
        values = middle + np.sin(hours / 24 * 2 * np.pi) * min(1.0, (high - low) / 20)
        if name in holes:
            values[holes[name]] = np.nan
        for at, value in spikes.get(name, {}).items():
            values[index.get_loc(at)] = value
        write_hourly(name, hourly(name, index, values), root=root)
    return root


@pytest.fixture
def store(tmp_path):
    return build_store(tmp_path), ANCHOR


def future(anchor, hours=None):
    index = pd.date_range(anchor + pd.Timedelta(hours=1),
                          periods=hours or timesfm.HORIZON_HOURS, freq="h",
                          name="target_time")
    return pd.DataFrame({name: 10.0 for name in timesfm.KNOWN_FUTURE}, index=index)


def test_the_anchor_is_the_last_hour_the_eisbach_was_measured(store):
    root, anchor = store
    index = pd.date_range(anchor + pd.Timedelta(hours=1), periods=5, freq="h")
    write_hourly(timesfm.TARGET, hourly(timesfm.TARGET, index, [np.nan] * 5), root=root)
    found = timesfm.anchor_time(root=root, now=anchor + pd.Timedelta(hours=5))
    assert found == anchor


def test_a_thin_context_is_refused_rather_than_interpolated(tmp_path):
    """The research selected its origins at 95 % coverage; production runs on the same."""
    root = build_store(tmp_path, holes={"rain_kochel": slice(0, 700)})
    with pytest.raises(RuntimeError, match="Context too thin.*rain_kochel"):
        timesfm.context_frame(ANCHOR, root=root)


def test_a_few_missing_hours_are_left_to_the_model(tmp_path):
    """Below the threshold the gaps stay as NaN; the backend trims and interpolates."""
    root = build_store(tmp_path, holes={"rain_kochel": slice(100, 200)})
    frame = timesfm.context_frame(ANCHOR, root=root)
    assert frame.rain_kochel.isna().sum() == 100
    assert len(frame) == timesfm.CONTEXT_HOURS


def test_the_gate_reaches_the_candidate_too(tmp_path):
    """A 154.4 °C reading is a hole here, not a number: `model_values` runs first."""
    spike = ANCHOR - pd.Timedelta(hours=10)
    root = build_store(tmp_path, spikes={timesfm.TARGET: {spike: 154.4}})
    frame = timesfm.context_frame(ANCHOR, root=root)
    assert pd.isna(frame.loc[spike, timesfm.TARGET])


def test_a_missing_measurement_at_the_anchor_stops_the_run(tmp_path):
    root = build_store(tmp_path, holes={timesfm.TARGET: slice(-1, None)})
    with pytest.raises(RuntimeError, match="No eisbach measurement at the anchor"):
        timesfm.context_frame(ANCHOR, root=root)


def test_the_model_is_handed_the_target_the_past_and_the_future(store):
    root, anchor = store
    context = timesfm.context_frame(anchor, root=root)
    stub = Stub()
    timesfm.predict(context, future(anchor), forecaster=stub)
    assert stub.seen["target"].shape == (timesfm.CONTEXT_HOURS,)
    assert stub.seen["past_only"].shape == (len(timesfm.PAST_ONLY), timesfm.CONTEXT_HOURS)
    # The known-future block is what makes it known-future: context *and* horizon.
    assert stub.seen["past_future"].shape == (len(timesfm.KNOWN_FUTURE),
                                              timesfm.CONTEXT_HOURS + timesfm.HORIZON_HOURS)
    assert stub.seen["horizon"] == timesfm.HORIZON_HOURS


def test_crossing_quantiles_are_refused(store):
    """Every band on the plot and every pinball term assumes the deciles are ordered."""
    root, anchor = store
    crossing = np.tile(np.arange(len(timesfm.DECILES), dtype=float)[::-1],
                       (timesfm.HORIZON_HOURS, 1))
    with pytest.raises(RuntimeError, match="crossing quantiles"):
        timesfm.predict(timesfm.context_frame(anchor, root=root), future(anchor),
                        forecaster=Stub(crossing))


def test_non_finite_quantiles_are_refused(store):
    root, anchor = store
    values = np.full((timesfm.HORIZON_HOURS, len(timesfm.DECILES)), np.nan)
    with pytest.raises(RuntimeError, match="non-finite"):
        timesfm.predict(timesfm.context_frame(anchor, root=root), future(anchor),
                        forecaster=Stub(values))


def test_an_incomplete_weather_forecast_is_refused(store, mocker):
    """A hole in MOSMIX is a station that is not being forecast, not a late report."""
    _root, anchor = store
    holed = future(anchor)
    holed.loc[holed.index[3], "rain_toelz"] = np.nan
    mocker.patch("eisbach.timesfm.fetch_future", return_value=holed)
    with pytest.raises(RuntimeError, match="Incomplete weather forecast.*rain_toelz"):
        timesfm.future_frame(anchor)


def test_the_candidate_writes_to_its_own_store(store, tmp_path, mocker):
    """A candidate row must never displace the forecast this project published."""
    root, anchor = store
    archive = tmp_path / "archive"
    mocker.patch("eisbach.timesfm.fetch_future", return_value=future(anchor))
    quantiles, context, predicted = timesfm.run(root=archive, covariates=root,
                                                now=anchor, forecaster=Stub())
    assert quantiles.shape == (timesfm.HORIZON_HOURS, len(timesfm.DECILES))
    assert len(context) == timesfm.CONTEXT_HOURS
    assert not predicted.empty
    assert (archive / "timesfm").is_dir()
    assert not (archive / "forecasts").exists()
    stored = read_forecasts(root=archive, store=timesfm.ARCHIVE_STORE)
    assert len(stored) == timesfm.HORIZON_HOURS
    assert stored.model_id.eq(f"{timesfm.CHECKPOINT}@{timesfm.MODEL_REVISION[:12]}").all()


def test_the_weather_it_was_told_is_archived_with_it(store, tmp_path, mocker):
    """Without this the candidate could only ever be re-scored against hindsight."""
    root, anchor = store
    archive = tmp_path / "archive"
    mocker.patch("eisbach.timesfm.fetch_future", return_value=future(anchor))
    timesfm.run(root=archive, covariates=root, now=anchor, forecaster=Stub())
    kept = pd.read_csv(archive / "covariate_forecasts" / "2026-09.csv")
    assert len(kept) == timesfm.HORIZON_HOURS
    assert set(timesfm.KNOWN_FUTURE) <= set(kept.columns)


def test_backtests_come_from_the_candidates_own_archive(store, tmp_path, mocker):
    root, anchor = store
    archive = tmp_path / "archive"
    mocker.patch("eisbach.timesfm.fetch_future",
                 side_effect=lambda names, start, end: future(pd.Timestamp(start)
                                                              - pd.Timedelta(hours=1)))
    earlier = anchor - pd.Timedelta(hours=96)
    timesfm.run(root=archive, covariates=root, now=earlier, forecaster=Stub())
    found = timesfm.backtests(anchor, root=archive)
    assert set(found) == {96}
    assert len(found[96]) == timesfm.HORIZON_HOURS
    assert "wassertemp_q0.5" in found[96].columns


def test_an_empty_archive_yields_no_backtests(tmp_path):
    assert timesfm.backtests(pd.Timestamp("2026-09-01", tz="UTC"), root=tmp_path) == {}


def test_a_forecast_hour_is_kept_and_labelled_as_one():
    """Bright Sky serves the measurement where it has one; the difference is recorded."""
    payload = {
        "sources": [{"id": 1, "dwd_station_id": "03379", "observation_type": "current"},
                    {"id": 2, "dwd_station_id": "03379", "observation_type": "forecast"}],
        "weather": [{"timestamp": "2026-09-01T00:00:00+00:00", "source_id": 1, "temperature": 12.0},
                    {"timestamp": "2026-09-01T01:00:00+00:00", "source_id": 2, "temperature": 13.0}],
    }
    answered = future_weather(payload, "03379", "temperature")
    assert answered.value.tolist() == [12.0, 13.0]
    assert answered.kind.tolist() == ["current", "forecast"]


def test_a_substituted_station_is_refused_in_the_forecast_too():
    """A coordinate lookup resolves to the nearest reporting station and switches silently."""
    payload = {"sources": [{"id": 1, "dwd_station_id": "10865", "observation_type": "forecast"}],
               "weather": [{"timestamp": "2026-09-01T00:00:00+00:00", "source_id": 1,
                            "temperature": 12.0}]}
    with pytest.raises(ValueError, match="substitution"):
        future_weather(payload, "03379", "temperature")


# --------------------------------------------------------------------------------------
# What reaches the page
# --------------------------------------------------------------------------------------

def quantile_frame(anchor, hours=None, base=15.0):
    index = pd.date_range(anchor + pd.Timedelta(hours=1),
                          periods=hours or timesfm.HORIZON_HOURS, freq="h",
                          name="target_time")
    return pd.DataFrame({f"wassertemp_q{q}": base + (q - 0.5) * 2 for q in timesfm.DECILES},
                        index=index)


def test_the_backtest_image_is_only_written_once_there_is_one(store, tmp_path, monkeypatch):
    """The candidate's archive starts empty; an empty picture is worse than no picture."""
    import matplotlib

    matplotlib.use("Agg")
    from eisbach.plotting import plot_timesfm

    root, anchor = store
    monkeypatch.chdir(tmp_path)
    context = timesfm.context_frame(anchor, root=root)
    predicted, quantiles = future(anchor), quantile_frame(anchor)

    written = plot_timesfm(context, predicted, quantiles, issued_at=anchor)
    assert written == ["Prediction_timesfm.png"]
    assert not (tmp_path / "Prediction_Backtest_timesfm.png").exists()

    written = plot_timesfm(context, predicted, quantiles, issued_at=anchor,
                           backtests={96: quantile_frame(anchor - pd.Timedelta(hours=96))})
    assert written == ["Prediction_timesfm.png", "Prediction_Backtest_timesfm.png"]
    assert (tmp_path / "Prediction_Backtest_timesfm.png").stat().st_size > 0


def test_the_csv_is_local_time_like_the_production_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    timesfm.write_csv(quantile_frame(pd.Timestamp("2026-09-01 00:00", tz="UTC"), hours=2))
    rows = (tmp_path / timesfm.CSV_NAME).read_text().splitlines()
    assert rows[0].startswith("target_time,wassertemp_q0.1")
    assert rows[1].startswith("2026-09-01 03:00,")  # 01:00 UTC is 03:00 in Munich in September


def test_a_failing_candidate_does_not_take_the_run_down(mocker, caplog):
    """The published DUET forecast is already on disk by the time this is called."""
    import logging

    import main

    mocker.patch("eisbach.timesfm.run", side_effect=RuntimeError("no checkpoint"))
    with caplog.at_level(logging.ERROR):
        main.run_candidate(pd.Timestamp("2026-09-01", tz="UTC"))
    # Swallowed, but never silently: a candidate that quietly stopped running would look
    # exactly like one that is doing fine.
    assert "TimesFM candidate failed" in caplog.text
    assert "no checkpoint" in caplog.text
