"""Archive invariants: provenance, missingness, source identity and bounded repair."""
import json

import numpy as np
import pandas as pd
import pytest

from eisbach.covariates import (
    SPECS,
    classify,
    fill_short_gaps,
    hourly_gkd,
    hourly_weather,
    model_values,
    quality,
    read_hourly,
    refresh,
    write_hourly,
)
from eisbach.data import ImplausibleGaugeData


def series(values):
    return pd.Series(values, index=pd.date_range("2026-09-01", periods=len(values), freq="h", tz="UTC"))


def frame(values):
    s = series(values)
    return pd.DataFrame({"raw_value": s, "raw_min": s, "raw_max": s, "samples": s.notna().astype(int),
                         "duplicate_count": 0, "conflict": False})


def payload(station="03379", kind="historical", values=(12, None)):
    return {"sources": [{"id": 1, "dwd_station_id": station, "observation_type": kind}],
            "weather": [{"timestamp": str(ts), "source_id": 1, "temperature": v}
                        for ts, v in series(values).items()]}


def test_pinned_sources_and_forecast_exclusion():
    with pytest.raises(ValueError, match="substitution"):
        hourly_weather(payload(station="02738"), "03379", "temperature")
    assert hourly_weather(payload(kind="forecast"), "03379", "temperature").empty
    out = hourly_weather(payload(), "03379", "temperature")
    assert out.raw_value.iloc[0] == 12
    assert pd.isna(out.raw_value.iloc[1])
    assert hourly_weather(payload(), "03379", "precipitation").raw_value.isna().all()


def test_duplicate_conflict_does_not_become_an_average():
    data = payload(values=(12, 13))
    data["weather"][1]["timestamp"] = data["weather"][0]["timestamp"]
    out = hourly_weather(data, "03379", "temperature")
    assert pd.isna(out.raw_value.iloc[0])
    assert out.raw_min.iloc[0] == 12
    assert out.raw_max.iloc[0] == 13
    assert out.duplicate_count.iloc[0] == 2


def gkd(rows):
    return ('<table class="tblsort"><thead><tr><th>Datum</th>'
            '<th>Wassertemperatur [°C]</th></tr></thead><tbody>'
            + ''.join(f'<tr><td>{ts} Uhr</td><td>{v}</td></tr>' for ts, v in rows)
            + '</tbody></table>')


def test_gkd_takes_the_first_sample_and_still_sees_the_extreme():
    """The production convention is first-of-hour, and the extreme must survive it.

    Averaging would have hidden 154.4 °C inside a mean of 86.75 and moved every other
    hour besides. Taking the first sample keeps the convention every archived
    observation was written under, and `raw_max` is what lets `model_values` notice a
    bad quarter-hour that the hourly value does not show.
    """
    html = gkd([("09.09.2026 12:00", "19,1"), ("09.09.2026 12:15", "154,4")])
    out, issues = hourly_gkd(html)
    assert not issues
    assert out.raw_value.iloc[0] == pytest.approx(19.1)
    assert out.raw_mean.iloc[0] == pytest.approx((19.1 + 154.4) / 2)
    assert out.raw_max.iloc[0] == 154.4
    assert str(out.index[0]) == "2026-09-09 10:00:00+00:00"


def test_dst_is_explicit_not_shifted_or_guessed():
    html = gkd([("25.10.2026 01:00", "12"), ("25.10.2026 02:00", "13"),
                ("25.10.2026 02:00", "14"), ("25.10.2026 03:00", "15"),
                ("29.03.2026 02:15", "16")])
    out, issues = hourly_gkd(html)
    assert len(issues) == 3
    assert out.loc["2026-10-25 00:00":"2026-10-25 01:00", "raw_value"].isna().all()
    assert 16 not in out.raw_value.values


def test_short_gap_policy_never_partially_fills_long_gap_or_edges():
    s = series([np.nan, 10, np.nan, 12, np.nan, np.nan, 15, np.nan])
    out = fill_short_gaps(s, "eisbach")
    assert out.iloc[2] == 11
    assert out.iloc[[0, 4, 5, 7]].isna().all()
    pd.testing.assert_series_equal(fill_short_gaps(s, "rain_lenggries"), s)
    pd.testing.assert_series_equal(fill_short_gaps(s, "solar_hohenpeissenberg"), s)


def test_dry_rain_and_flat_water_are_review_only():
    s = series([0.] * 200)
    flags = quality(s, "eisbach")
    assert flags.frozen.all()
    assert not flags["range"].any()
    assert "physical_zero" in classify("rain_toelz", "frozen", 0)
    assert "undecidable" in classify("eisbach", "frozen", 0)
    pd.testing.assert_series_equal(fill_short_gaps(s, "eisbach"), s)


def test_archive_rejection_retains_raw_and_tombstone(tmp_path):
    raw = frame([19.1, 154.4, 19.1])
    write_hourly("eisbach", raw, root=tmp_path)
    s = model_values("eisbach", root=tmp_path)
    assert pd.isna(s.iloc[1])
    assert read_hourly("eisbach", root=tmp_path).raw_value.iloc[1] == 154.4
    decisions = pd.read_csv(tmp_path / "decisions/eisbach/2026-09.csv")
    assert decisions.status.tolist() == ["never_measured"]
    assert decisions.raw_value.tolist() == [154.4]
    assert decisions.value.isna().all()
    model_values("eisbach", root=tmp_path)
    assert len(pd.read_csv(tmp_path / "decisions/eisbach/2026-09.csv")) == 1


def test_small_subhourly_fault_is_not_diluted(tmp_path):
    raw = frame([19.1])
    raw["raw_max"] = 41
    write_hourly("eisbach", raw, root=tmp_path)
    assert model_values("eisbach", root=tmp_path).isna().all()


def test_merge_missing_does_not_erase_real_value_and_partition_boundary(tmp_path):
    raw = frame([12, np.nan])
    raw.index = pd.date_range("2026-08-31 23:00", periods=2, freq="h", tz="UTC")
    write_hourly("eisbach", raw, root=tmp_path)
    update = raw.copy()
    update["raw_value"] = [np.nan, 13]
    write_hourly("eisbach", update, root=tmp_path)
    out = read_hourly("eisbach", root=tmp_path)
    assert out.raw_value.tolist() == [12, 13]
    assert len(list((tmp_path / "hourly/eisbach").glob("*.csv"))) == 2


def test_refresh_pins_station_and_uses_bounded_delta(mocker, tmp_path):
    # All network calls are mocked, including sources and GKD.
    for name in SPECS:
        write_hourly(name, frame([12.] * 24), root=tmp_path)
    response = mocker.Mock()
    response.content = b"retained original response"
    response.text = "parsed by mock"
    response.json.return_value = {}
    get = mocker.patch("eisbach.covariates.request", return_value=response)
    mocker.patch("eisbach.covariates.hourly_weather", return_value=frame([12.] * 24))
    mocker.patch("eisbach.covariates.hourly_gkd", return_value=(frame([12.] * 24), []))
    refresh(root=tmp_path, now=pd.Timestamp("2026-09-02", tz="UTC"))
    assert get.call_count == len({s.station for s in SPECS.values()})
    for call in get.call_args_list:
        params = call.args[1]
        if "date" in params:
            assert "dwd_station_id" in params
            assert not {"lat", "lon"} & params.keys()
            assert pd.Timestamp(params["last_date"]) - pd.Timestamp(params["date"]) < pd.Timedelta(days=4)
    assert len(json.loads((tmp_path / "cursors.json").read_text())) == get.call_count
    assert list((tmp_path / "raw").glob("*/live/*.gz"))


def test_failed_refresh_does_not_advance_cursor(mocker, tmp_path):
    for name in SPECS:
        write_hourly(name, frame([12.] * 24), root=tmp_path)
    mocker.patch("eisbach.covariates.request", side_effect=RuntimeError("network down"))
    with pytest.raises(RuntimeError, match="network down"):
        refresh(root=tmp_path, now=pd.Timestamp("2026-09-02", tz="UTC"))
    assert not (tmp_path / "cursors.json").exists()


def test_station_id_survives_csv_and_null_row_retains_metadata(tmp_path):
    raw = frame([12.])
    write_hourly("airtemp", raw, root=tmp_path)
    update = frame([np.nan])
    write_hourly("airtemp", update, root=tmp_path)
    stored = read_hourly("airtemp", root=tmp_path)
    assert stored.station.iloc[0] == "03379"
    assert stored.samples.iloc[0] == 1
    assert stored.raw_min.iloc[0] == 12


def test_window_policy_rejects_fifty_five_days_and_rain_hole(tmp_path):
    from eisbach.covariates import load_window

    write_hourly("eisbach", frame([12, *([np.nan] * (55 * 24)), 13]), root=tmp_path)
    with pytest.raises(ValueError, match="Unusable window"):
        load_window(["eisbach"], "2026-09-01", "2026-10-27", root=tmp_path)
    write_hourly("rain_toelz", frame([0, np.nan, 0]), root=tmp_path)
    with pytest.raises(ValueError, match="Unusable window"):
        load_window(["rain_toelz"], "2026-09-01", "2026-09-01 02:00", root=tmp_path)


def test_model_wrapper_does_not_hide_unfillable_gaps():
    from types import SimpleNamespace

    from eisbach.model.api import SERIES_ORDER, forecast, long_to_wide

    wide = pd.DataFrame({n: [12, np.nan, 13] for n in SERIES_ORDER}, index=series([1, 2, 3]).index)
    long = wide.rename_axis("date").reset_index().melt(id_vars="date", var_name="cols", value_name="data")
    restored = long_to_wide(long)
    assert restored.iloc[1].isna().all()
    with pytest.raises(ValueError, match="unfillable gaps"):
        forecast(None, SimpleNamespace(seq_len=3), restored)


def test_leap_year_refresh_never_requests_next_local_year(mocker, tmp_path):
    for name in SPECS:
        raw = frame([12.] * 24)
        raw.index = pd.date_range("2024-12-31", periods=24, freq="h", tz="UTC")
        write_hourly(name, raw, root=tmp_path)
    response = mocker.Mock(content=b"source", text="source")
    response.json.return_value = {}
    get = mocker.patch("eisbach.covariates.request", return_value=response)
    mocker.patch("eisbach.covariates.hourly_weather", return_value=pd.DataFrame())
    mocker.patch("eisbach.covariates.hourly_gkd", return_value=(pd.DataFrame(), []))
    refresh(root=tmp_path, now=pd.Timestamp("2025-01-02", tz="UTC"))
    for call in get.call_args_list:
        params = call.args[1]
        if "beginn" in params and params["beginn"].endswith("2024"):
            assert params["ende"] == "31.12.2024"


def test_an_in_range_spike_is_rejected_for_the_model(tmp_path):
    """The whole point of the gate, and what bounds alone cannot do.

    30 °C is inside the plausible range for a river and no Eisbach hour ever did it
    between neighbours of 19 °C. Rejecting only out-of-range values catches 154.4 and
    lets this through — which is what production looked like when `main.py` moved off
    `reject_implausible_readings` and the neighbourhood verdict was computed but unused.
    """
    raw = frame([19.1, 19.0, 19.1, 19.0, 30.0, 19.1, 19.0, 19.1, 19.0])
    write_hourly("eisbach", raw, root=tmp_path)

    s = model_values("eisbach", root=tmp_path)

    assert pd.isna(s.iloc[4])
    assert s.drop(s.index[4]).notna().all()
    # The raw archive is untouched; only the model's view of it changes.
    assert read_hourly("eisbach", root=tmp_path).raw_value.iloc[4] == 30.0
    decisions = pd.read_csv(tmp_path / "decisions/eisbach/2026-09.csv")
    assert decisions.reason.tolist() == ["neighbourhood_spike"]


def test_a_calm_series_is_left_alone(tmp_path):
    """MAD is exactly zero over a still winter window; without the floor the threshold
    collapses to zero and the gate rejects most of the window."""
    raw = frame([4.1] * 12)
    write_hourly("eisbach", raw, root=tmp_path)
    assert model_values("eisbach", root=tmp_path).notna().all()


def test_a_broken_instrument_stops_the_run(tmp_path):
    """Past the budget the series is not a good signal with a spike in it, and a run
    that interpolated over it would forecast from mostly guesses."""
    values = [19.0] * 200
    for i in range(10, 200, 20):
        values[i] = 30.0
    write_hourly("eisbach", frame(values), root=tmp_path)
    with pytest.raises(ImplausibleGaugeData, match="broken, not spiky"):
        model_values("eisbach", root=tmp_path)


def test_one_spike_in_a_long_window_is_within_budget(tmp_path):
    """A single bad hour in a healthy window is repaired, not grounds for refusing."""
    values = [19.0] * 200
    values[100] = 30.0
    write_hourly("eisbach", frame(values), root=tmp_path)
    s = model_values("eisbach", root=tmp_path)
    assert pd.isna(s.iloc[100])
    assert s.notna().sum() == 199


def test_the_budget_cannot_condemn_a_window_too_short_to_judge(tmp_path):
    """One rejection in nine readings is 11 %, which says nothing about the instrument."""
    raw = frame([19.1, 19.0, 19.1, 19.0, 30.0, 19.1, 19.0, 19.1, 19.0])
    write_hourly("eisbach", raw, root=tmp_path)
    assert pd.isna(model_values("eisbach", root=tmp_path).iloc[4])


def test_rain_keeps_its_downpours(tmp_path):
    """A front, a downpour and sunrise are genuinely abrupt: the neighbourhood test is
    a review flag for weather, not grounds for rejection."""
    raw = frame([0.0, 0.0, 0.0, 0.0, 30.6, 0.0, 0.0, 0.0])
    write_hourly("rain_kochel", raw, root=tmp_path)
    assert model_values("rain_kochel", root=tmp_path).notna().all()


def test_ingestion_records_a_broken_gauge_instead_of_refusing(tmp_path, mocker):
    """`model_values` raises so no forecast is built on a broken instrument. Ingestion
    must still write: losing the evidence that the gauge broke defeats the archive.

    Today the decision pass uses a four-day window, below the budget's minimum sample
    size, so this cannot fire. The guard exists so widening that window later does not
    quietly turn a bad gauge into a failed ingestion.
    """
    for name in SPECS:
        raw = frame([12.0] * 24)
        raw.index = pd.date_range("2026-09-13", periods=24, freq="h", tz="UTC")
        write_hourly(name, raw, root=tmp_path)
    response = mocker.Mock(content=b"source", text="source")
    response.json.return_value = {}
    mocker.patch("eisbach.covariates.request", return_value=response)
    mocker.patch("eisbach.covariates.hourly_weather", return_value=pd.DataFrame())
    mocker.patch("eisbach.covariates.hourly_gkd", return_value=(pd.DataFrame(), []))
    mocker.patch("eisbach.covariates.model_values",
                 side_effect=ImplausibleGaugeData("gauge is broken, not spiky"))

    refresh(root=tmp_path, now=pd.Timestamp("2026-09-14 12:00", tz="UTC"))

    # The cursors were still written: the pass completed rather than aborting.
    assert (tmp_path / "cursors.json").exists()
