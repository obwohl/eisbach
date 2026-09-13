"""The input-side plausibility gate.

The forecast gate in ``eisbach.validate`` judges what the model produced. This one
judges what it was fed, which is the failure that actually happened: one 154.4 °C gauge
reading reached the model's input, the observation archive and 52 rows of the
append-only verification store.
"""

import numpy as np
import pandas as pd
import pytest

from eisbach.data import (
    MAX_REJECTED_FRACTION,
    SPIKE_FLOOR_C,
    ImplausibleGaugeData,
    assemble_long_frame,
    reject_implausible_readings,
)


def gauge(values, start="2026-09-08 00:00"):
    """An hourly gauge frame in the shape ``prepare_data`` hands on."""
    index = pd.date_range(start, periods=len(values), freq="1h", tz="Europe/Berlin")
    return pd.DataFrame({"timestamp": index, "wassertemp": list(values)})


def diurnal(hours=240, mean=18.0, amplitude=0.8):
    """A plausible river: a daily cycle with a slow drift, as the Eisbach really moves."""
    t = np.arange(hours)
    return np.round(mean + amplitude * np.sin(2 * np.pi * t / 24) + 0.004 * t, 1)


def test_the_reading_that_got_through_is_rejected():
    values = diurnal()
    values[100] = 154.4
    out = reject_implausible_readings(gauge(values))

    assert pd.isna(out.loc[100, "wassertemp"])
    assert out["wassertemp"].drop(index=100).notna().all()


def test_a_spike_inside_the_plausible_range_is_still_rejected():
    # 30 °C is a fault for this river but passes any absolute bound wide enough to be
    # safe, so only the neighbourhood test can catch it.
    values = diurnal(mean=8.0)
    values[57] = 30.0
    out = reject_implausible_readings(gauge(values))

    assert pd.isna(out.loc[57, "wassertemp"])


def test_a_three_hour_spike_is_rejected_whole():
    # A real run carries 40 days, so three rejected hours are well inside the budget;
    # the window here is long enough to say the same.
    values = diurnal(hours=480)
    values[80:83] = [61.0, 63.0, 60.0]
    out = reject_implausible_readings(gauge(values))

    assert out.loc[80:82, "wassertemp"].isna().all()


def test_a_calm_window_is_left_alone():
    """The regression that matters most.

    The gauge reports in steps of 0.1 °C and the river can sit still for days, so the
    residual MAD of a calm window is exactly zero — true of 11 % of 40-day windows in the
    sixteen-year record. Without the floor under the threshold this rejects everything.
    """
    values = np.full(240, 4.2)
    values[120] = 4.3
    out = reject_implausible_readings(gauge(values))

    assert out["wassertemp"].notna().all()


def test_a_steep_but_real_change_survives():
    """The largest neighbourhood residual that was ever real is 1.33 °C."""
    values = diurnal()
    values[130] += 1.3
    out = reject_implausible_readings(gauge(values))

    assert out["wassertemp"].notna().all()
    assert out.loc[130, "wassertemp"] == pytest.approx(values[130])


def test_a_thunderstorm_sized_swing_survives():
    """A genuine multi-hour plunge is a change of regime, not a spike."""
    values = diurnal()
    values[120:] -= 3.0
    out = reject_implausible_readings(gauge(values))

    assert out["wassertemp"].notna().all()


def test_a_broken_gauge_raises_rather_than_being_interpolated_over():
    values = diurnal(hours=200)
    values[50] = 154.4
    values[80] = 160.0
    values[120] = -99.0
    with pytest.raises(ImplausibleGaugeData, match="broken, not spiky"):
        reject_implausible_readings(gauge(values))


def test_the_rejection_budget_is_the_one_the_constant_names():
    values = diurnal(hours=400)
    values[200] = 154.4
    # One rejection in 400 readings is inside the budget; the check must pass.
    assert 1 / 400 <= MAX_REJECTED_FRACTION
    out = reject_implausible_readings(gauge(values))
    assert out["wassertemp"].isna().sum() == 1


def test_an_hour_the_gauge_never_sampled_is_not_a_rejection():
    values = list(diurnal())
    values[60] = np.nan
    out = reject_implausible_readings(gauge(values))

    assert pd.isna(out.loc[60, "wassertemp"])
    assert out["wassertemp"].isna().sum() == 1


def test_a_frame_with_nothing_measured_survives():
    out = reject_implausible_readings(gauge([np.nan] * 48))
    assert out["wassertemp"].isna().all()


def test_the_model_input_is_repaired_but_the_archive_is_not_invented():
    """One NaN, two behaviours — which is the whole point of rejecting this way.

    ``assemble_long_frame`` interpolates the model's input so it never sees a hole, while
    ``inference._observed_frame`` drops NaN before archiving, so the hour is recorded as
    never measured rather than as a value nobody read off an instrument.
    """
    values = diurnal(hours=200)
    values[100] = 154.4
    df_wt = reject_implausible_readings(gauge(values))

    weather = pd.DataFrame(
        {"lufttemperatur_c": np.linspace(10, 20, 200), "pressure": np.full(200, 1013.0)},
        index=df_wt["timestamp"],
    )
    long = assemble_long_frame(df_wt, weather)
    water = long[long["cols"] == "wassertemp"]
    assert water["data"].notna().all()

    repaired = water.iloc[100]["data"]
    assert 16.0 < repaired < 20.0

    from eisbach.inference import _observed_frame

    df_wt_utc = df_wt.copy()
    df_wt_utc["timestamp"] = df_wt_utc["timestamp"].dt.tz_convert("UTC")
    observed = _observed_frame(
        df_wt_utc, weather.tz_convert("UTC"), df_wt_utc["timestamp"].max(),
    )
    assert df_wt.loc[100, "timestamp"].tz_convert("UTC") not in set(observed.index)
    assert len(observed) == len(df_wt) - 1


def test_the_floor_is_what_keeps_a_quiet_window_safe():
    """A spike just under the floor is kept; one just over it is not."""
    values = diurnal()
    values[70] += SPIKE_FLOOR_C * 0.8
    assert reject_implausible_readings(gauge(values))["wassertemp"].notna().all()

    values = diurnal()
    values[70] += SPIKE_FLOOR_C * 1.5
    assert pd.isna(reject_implausible_readings(gauge(values)).loc[70, "wassertemp"])


def test_a_wide_spike_does_not_take_its_healthy_neighbours_with_it():
    """Why the check runs twice.

    A three-hour fault fills half of each adjacent hour's neighbourhood, so a single pass
    condemns the two good readings on either side of it as well — five rejections for a
    three-hour fault, and two real measurements thrown away. The second pass rebuilds
    each neighbourhood without the readings the first one rejected.
    """
    values = diurnal(hours=480)
    values[80:83] = [61.0, 63.0, 60.0]
    out = reject_implausible_readings(gauge(values))

    assert out["wassertemp"].isna().sum() == 3
    assert out.loc[79, "wassertemp"] == pytest.approx(values[79])
    assert out.loc[83, "wassertemp"] == pytest.approx(values[83])
