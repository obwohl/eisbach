"""Tests for the plausibility gate that decides whether a run may publish.

Nothing here runs the model: ``validate.py`` only ever looks at frames, so the frames
are built by hand and the numbers are chosen so the expected answer is exact. See
tests/test_inference.py for where those frames really come from.
"""

import logging

import pandas as pd
import pytest

from eisbach import archive, inference, validate

QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

#: Centre and half-width of each channel's band in a healthy forecast.
CENTRES = {"wassertemp": 15.0, "airtemp_96": 18.0, "pressure_96": 1013.0}
BANDS = {"wassertemp": 1.0, "airtemp_96": 5.0, "pressure_96": 10.0}

LAST_OBSERVATION = pd.Timestamp("2026-05-20 09:00", tz="UTC")
FORECAST_START = LAST_OBSERVATION + pd.Timedelta(hours=1)


def make_forecast(start=FORECAST_START, periods=4, water=15.0, band=1.0):
    """A frame shaped like the model's output: three channels, seven quantiles each.

    Constant along the index, so ``min``/``max`` of a column are the value itself and any
    excursion a test injects is the only one.
    """
    index = pd.date_range(pd.Timestamp(start), periods=periods, freq="1h")
    centres = dict(CENTRES, wassertemp=water)
    bands = dict(BANDS, wassertemp=band)
    data = {
        f"{channel}_q{q}": centre + (q - 0.5) * 2 * bands[channel]
        for channel, centre in centres.items()
        for q in QUANTILES
    }
    return pd.DataFrame(data, index=index)


def make_long(index, water=15.0):
    """Observations shaped like prepare_data's output: one (date, cols, data) row each."""
    frames = [pd.DataFrame({"date": index, "cols": "wassertemp", "data": water})]
    for channel in ("airtemp_96", "pressure_96"):
        frames.append(pd.DataFrame({"date": index, "cols": channel, "data": CENTRES[channel]}))
    return pd.concat(frames, ignore_index=True)


def make_backtest(start, periods=4, offset_hours=96, kind=archive.KIND_LIVE, **kwargs):
    reference_time = pd.Timestamp(start) - pd.Timedelta(hours=1)
    return inference.Backtest(
        offset_hours=offset_hours,
        reference_time=reference_time,
        forecast=make_forecast(start, periods=periods, **kwargs),
        kind=kind,
        covariate_source=archive.COVARIATE_DWD_FORECAST,
    )


def observed_index(periods=24, end=LAST_OBSERVATION):
    return pd.date_range(end=end, periods=periods, freq="1h")


# --------------------------------------------------------------------------- _check_ranges


def test_a_well_formed_forecast_passes_the_range_check():
    assert validate._check_ranges(make_forecast(), "forecast") is None


@pytest.mark.parametrize("channel", sorted(validate.PLAUSIBLE_RANGES))
def test_a_nan_in_a_quantile_column_raises(channel):
    df = make_forecast()
    df.loc[df.index[2], f"{channel}_q0.5"] = float("nan")

    with pytest.raises(validate.ImplausibleForecast, match=rf"forecast: {channel}_q0\.5 contains NaN"):
        validate._check_ranges(df, "forecast")


@pytest.mark.parametrize("channel", sorted(validate.PLAUSIBLE_RANGES))
def test_a_value_below_the_low_bound_raises(channel):
    low, _ = validate.PLAUSIBLE_RANGES[channel]
    df = make_forecast()
    df.loc[df.index[0], f"{channel}_q0.01"] = low - 1.0

    with pytest.raises(validate.ImplausibleForecast) as excinfo:
        validate._check_ranges(df, "forecast")

    assert f"{channel}_q0.01" in str(excinfo.value)
    assert "outside the plausible" in str(excinfo.value)


@pytest.mark.parametrize("channel", sorted(validate.PLAUSIBLE_RANGES))
def test_a_value_above_the_high_bound_raises(channel):
    _, high = validate.PLAUSIBLE_RANGES[channel]
    df = make_forecast()
    df.loc[df.index[-1], f"{channel}_q0.99"] = high + 1.0

    with pytest.raises(validate.ImplausibleForecast) as excinfo:
        validate._check_ranges(df, "forecast")

    assert f"{channel}_q0.99" in str(excinfo.value)
    assert "outside the plausible" in str(excinfo.value)


def test_a_value_exactly_on_a_bound_passes():
    """The bounds are generous already, so they are inclusive rather than strict."""
    low, high = validate.PLAUSIBLE_RANGES["wassertemp"]
    df = make_forecast()
    df.loc[df.index[0], "wassertemp_q0.01"] = low
    df.loc[df.index[0], "wassertemp_q0.99"] = high

    assert validate._check_ranges(df, "forecast") is None


def test_columns_without_a_quantile_suffix_are_not_range_checked():
    """Only ``<channel>_q…`` columns are model output; anything else is passed through."""
    df = make_forecast()
    df["wassertemp_mean"] = 9999.0

    assert validate._check_ranges(df, "forecast") is None


def test_the_label_argument_names_what_failed():
    df = make_forecast()
    df.loc[df.index[0], "wassertemp_q0.5"] = 300.0

    with pytest.raises(validate.ImplausibleForecast, match=r"^backtest -192h: "):
        validate._check_ranges(df, "backtest -192h")


# ------------------------------------------------------------------------------ _coverage


def test_coverage_is_none_without_overlap():
    index = pd.date_range(FORECAST_START, periods=4, freq="1h")
    forecast = pd.DataFrame(
        {"wassertemp_q0.01": 14.0, "wassertemp_q0.99": 16.0}, index=index,
    )
    actuals = pd.Series(15.0, index=observed_index(periods=4))

    assert validate._coverage(forecast, actuals) is None


def test_coverage_is_the_fraction_of_observations_inside_the_band():
    index = observed_index(periods=4)
    forecast = pd.DataFrame(
        {"wassertemp_q0.01": 14.0, "wassertemp_q0.99": 16.0}, index=index,
    )
    actuals = pd.Series([15.0, 15.5, 99.0, 14.5], index=index)

    assert validate._coverage(forecast, actuals) == 0.75


def test_coverage_counts_observations_on_the_band_edges_as_covered():
    index = observed_index(periods=2)
    forecast = pd.DataFrame(
        {"wassertemp_q0.01": 14.0, "wassertemp_q0.99": 16.0}, index=index,
    )
    actuals = pd.Series([14.0, 16.0], index=index)

    assert validate._coverage(forecast, actuals) == 1.0


def test_coverage_only_looks_at_the_overlap():
    """Observations outside the backtest's own horizon must not dilute its score."""
    index = observed_index(periods=6)
    forecast = pd.DataFrame(
        {"wassertemp_q0.01": 14.0, "wassertemp_q0.99": 16.0}, index=index[:2],
    )
    actuals = pd.Series([15.0, 15.0, 99.0, 99.0, 99.0, 99.0], index=index)

    assert validate._coverage(forecast, actuals) == 1.0


# --------------------------------------------------------------------------- validate_run


def test_an_empty_forecast_raises():
    df_long = make_long(observed_index())

    with pytest.raises(validate.ImplausibleForecast, match="the forecast is empty"):
        validate.validate_run(pd.DataFrame(), {}, df_long)


def test_an_implausible_forecast_raises_before_anything_else():
    df_long = make_long(observed_index())
    df_inference = make_forecast()
    df_inference.loc[df_inference.index[0], "wassertemp_q0.5"] = 300.0

    with pytest.raises(validate.ImplausibleForecast, match=r"forecast: wassertemp_q0\.5 ranges"):
        validate.validate_run(df_inference, {}, df_long)


def test_a_forecast_overlapping_the_observations_raises():
    """If the truncation leaked, the model was shown the answer it was asked for."""
    index = observed_index(periods=24)
    df_long = make_long(index)
    # Starts three hours before the last observation, so three timestamps are measured.
    df_inference = make_forecast(start=LAST_OBSERVATION - pd.Timedelta(hours=2), periods=10)

    with pytest.raises(validate.ImplausibleForecast) as excinfo:
        validate.validate_run(df_inference, {}, df_long)

    assert "3 forecast timestamps already have measurements" in str(excinfo.value)


def test_missing_measurements_inside_the_horizon_are_not_a_leak():
    """A gap in the sensor data is not the model seeing its own answer."""
    index = pd.date_range(end=LAST_OBSERVATION + pd.Timedelta(hours=3), periods=27, freq="1h")
    # The three timestamps the forecast covers have a row each, but no reading.
    df_long = make_long(index, water=[15.0] * 24 + [float("nan")] * 3)

    assert validate.validate_run(make_forecast(), {}, df_long) is None


def test_a_badly_covered_backtest_raises_naming_kind_and_percentage():
    index = observed_index(periods=4)
    # Two of four observations sit far outside the band: 50 %, well under the threshold.
    df_long = make_long(index, water=[15.0, 15.0, 30.0, 30.0])
    backtests = {96: make_backtest(index[0], periods=4, kind=archive.KIND_ORACLE)}

    with pytest.raises(validate.ImplausibleForecast) as excinfo:
        validate.validate_run(make_forecast(), backtests, df_long)

    message = str(excinfo.value)
    assert "backtest -96h" in message
    assert f"({archive.KIND_ORACLE})" in message
    assert "covered only 50%" in message
    assert "at least 80%" in message


def test_a_backtest_exactly_at_the_threshold_passes():
    """``MIN_COVERAGE`` is a floor, not a strict minimum: 80 % is still acceptable."""
    index = observed_index(periods=5)
    df_long = make_long(index, water=[15.0, 15.0, 15.0, 15.0, 30.0])
    backtests = {96: make_backtest(index[0], periods=5)}

    assert validate._coverage(backtests[96].forecast, pd.Series(
        [15.0, 15.0, 15.0, 15.0, 30.0], index=index)) == validate.MIN_COVERAGE
    assert validate.validate_run(make_forecast(), backtests, df_long) is None


def test_a_backtest_just_below_the_threshold_raises():
    index = observed_index(periods=5)
    df_long = make_long(index, water=[15.0, 15.0, 15.0, 30.0, 30.0])
    backtests = {96: make_backtest(index[0], periods=5)}

    with pytest.raises(validate.ImplausibleForecast, match="covered only 60%"):
        validate.validate_run(make_forecast(), backtests, df_long)


def test_an_implausible_backtest_raises_naming_its_offset():
    index = observed_index(periods=4)
    df_long = make_long(index)
    backtest = make_backtest(index[0], periods=4, offset_hours=288)
    backtest.forecast.loc[backtest.forecast.index[0], "airtemp_96_q0.99"] = 500.0

    with pytest.raises(validate.ImplausibleForecast, match=r"backtest -288h: airtemp_96_q0\.99"):
        validate.validate_run(make_forecast(), {288: backtest}, df_long)


def test_a_backtest_without_overlap_warns_instead_of_failing(caplog):
    """Nothing to compare against is not evidence of a broken model."""
    index = observed_index(periods=4)
    df_long = make_long(index)
    # Anchored a week after the last observation, so it overlaps nothing measured.
    backtests = {96: make_backtest(LAST_OBSERVATION + pd.Timedelta(days=7), periods=4)}

    with caplog.at_level(logging.WARNING, logger="eisbach.validate"):
        assert validate.validate_run(make_forecast(), backtests, df_long) is None

    assert "no overlap with observations" in caplog.text


def test_coverage_is_logged_with_the_backtest_kind(caplog):
    index = observed_index(periods=4)
    df_long = make_long(index)
    backtests = {192: make_backtest(index[0], periods=4, offset_hours=192,
                                    kind=archive.KIND_REPLAY)}

    with caplog.at_level(logging.INFO, logger="eisbach.validate"):
        validate.validate_run(make_forecast(), backtests, df_long)

    assert "Backtest -192h (replay): 100% of observations" in caplog.text


def test_a_plausible_run_with_covered_backtests_passes():
    """The happy path: this is what a publishable run looks like."""
    index = observed_index(periods=48)
    df_long = make_long(index)
    df_inference = make_forecast(periods=96)
    backtests = {
        96: make_backtest(index[-24], periods=24, offset_hours=96, kind=archive.KIND_LIVE),
        192: make_backtest(index[-36], periods=36, offset_hours=192, kind=archive.KIND_REPLAY),
        288: make_backtest(index[-48], periods=48, offset_hours=288, kind=archive.KIND_ORACLE),
    }

    assert validate.validate_run(df_inference, backtests, df_long) is None
