"""Sanity checks on a completed run.

These are not unit tests — they need a real forecast — but they are the difference
between publishing a plausible plot and publishing a broken one. The pipeline runs
unattended three times a day, so a forecast that comes out at 300 °C, or one that quietly
contains no water temperature at all, needs to fail the run rather than get committed.

The second half of that sentence is the harder half. A check written as "every value I
can see is plausible" says nothing at all about a frame with no values in it, so the gate
first establishes that the data it is about to judge is *there*: the full channel ×
quantile grid, a full horizon of rows, observations to compare against, and at least one
backtest that actually overlaps them. Only then does it ask whether the numbers are
sensible. Absence is the failure mode that looks most like success.

Called from ``main.py`` after inference and before plotting.
"""

from __future__ import annotations

import logging

import pandas as pd

from eisbach.data import CHANNELS, COVARIATE_SHIFT_HOURS
from eisbach.model import QUANTILES

logger = logging.getLogger(__name__)


class ImplausibleForecast(RuntimeError):
    """Raised when a forecast is outside the range physics or the sensor allow."""


#: Generous bounds. These are not accuracy checks — they catch a broken model, a
#: mis-scaled channel or a garbage input, not a mediocre forecast.
PLAUSIBLE_RANGES = {
    "wassertemp": (-10.0, 35.0),
    "airtemp_96": (-30.0, 50.0),
    "pressure_96": (900.0, 1100.0),
}

#: Every channel at every quantile. This is what the model emits, so anything less means
#: the forecast was mangled between inference and here — and a missing column is exactly
#: what a range check cannot notice, because it has nothing to range-check.
REQUIRED_FORECAST_COLUMNS = tuple(
    f"{channel}_q{q}" for channel in CHANNELS for q in QUANTILES
)

#: What a *backtest* must carry. Only the water channel, deliberately: a backtest may
#: have been read back out of the archive rather than recomputed, and the covariate
#: quantiles in that store are on their way out (PRD R4). Water is what the plot draws
#: and what the coverage check scores, so water is what is required; any covariate column
#: that is present is still range-checked.
REQUIRED_BACKTEST_COLUMNS = tuple(f"wassertemp_q{q}" for q in QUANTILES)

#: The two columns :func:`_coverage` scores against.
BAND_COLUMNS = ("wassertemp_q0.01", "wassertemp_q0.99")

#: How many rows a complete forecast has. The model's horizon and the covariate shift
#: must be equal — a forecast hour beyond the shift has no weather to look at — so the
#: shift doubles as the expected row count, and checking it here turns that constraint
#: from a comment into something a run can fail on.
REQUIRED_FORECAST_ROWS = COVARIATE_SHIFT_HOURS

#: An honest backtest should contain the truth inside its 1–99 % band most of the time.
#: Well below the nominal 98 % because the band is only sampled at the extremes and the
#: sample is short; this catches a badly miscalibrated model, not a slightly wide one.
MIN_COVERAGE = 0.80


def _check_columns(df: pd.DataFrame, required, what: str) -> None:
    """Raise unless every column in ``required`` is present."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ImplausibleForecast(
            f"{what}: missing {len(missing)} of {len(required)} required columns: "
            f"{', '.join(missing)}"
        )


def _check_ranges(df: pd.DataFrame, what: str, required=REQUIRED_FORECAST_COLUMNS) -> None:
    """Range-check every quantile column present, and refuse any NaN in a required one.

    A column *outside* ``required`` that is empty from end to end counts as absent rather
    than as a column full of NaN. That is the shape a mixed-schema archive partition
    hands back: `write_forecast` concatenates onto what the month already holds, so once
    R4 writes water-only rows into a month that also holds full ones, the covariate
    columns exist for every row and are blank for the new ones — and `load_forecast`
    returns them that way. Refusing that would block production on exactly the backtests
    this gate says it accepts.

    A *partly* empty optional column still raises. That is a real gap in data that was
    written, not a column that was never written at all.
    """
    for channel, (low, high) in PLAUSIBLE_RANGES.items():
        for column in [c for c in df.columns if c.startswith(f"{channel}_q")]:
            series = df[column]
            if column not in required and series.isna().all():
                continue
            if series.isna().any():
                raise ImplausibleForecast(f"{what}: {column} contains NaN")
            if series.min() < low or series.max() > high:
                raise ImplausibleForecast(
                    f"{what}: {column} ranges {series.min():.1f}..{series.max():.1f}, "
                    f"outside the plausible {low}..{high}"
                )


def _coverage(backtest_forecast: pd.DataFrame, actuals: pd.Series,
              what: str = "backtest") -> float | None:
    """Fraction of observations that fell inside the 1–99 % band.

    ``None`` when the backtest and the observations do not overlap at all. A backtest
    without the band columns is a broken backtest, not a missing measurement, so it
    raises rather than returning ``None`` — and raises :class:`ImplausibleForecast`,
    because a ``KeyError`` escaping the gate is a crash rather than a refusal.
    """
    _check_columns(backtest_forecast, BAND_COLUMNS, what)

    overlap = backtest_forecast.index.intersection(actuals.index)
    if len(overlap) == 0:
        return None
    truth = actuals.loc[overlap]
    low = backtest_forecast.loc[overlap, BAND_COLUMNS[0]]
    high = backtest_forecast.loc[overlap, BAND_COLUMNS[1]]
    return float(((truth >= low) & (truth <= high)).mean())


def validate_run(df_inference: pd.DataFrame, backtests: dict, df_long: pd.DataFrame) -> None:
    """Raise :class:`ImplausibleForecast` if the run produced something unusable."""
    if df_inference.empty:
        raise ImplausibleForecast("the forecast is empty")
    _check_columns(df_inference, REQUIRED_FORECAST_COLUMNS, "forecast")
    if len(df_inference) != REQUIRED_FORECAST_ROWS:
        raise ImplausibleForecast(
            f"the forecast has {len(df_inference)} rows, expected exactly "
            f"{REQUIRED_FORECAST_ROWS} — either it was truncated after inference, or the "
            "model horizon and COVARIATE_SHIFT_HOURS have drifted apart"
        )
    _check_ranges(df_inference, "forecast")

    actuals = (
        df_long[df_long["cols"] == "wassertemp"]
        .dropna(subset=["data"])
        .set_index("date")["data"]
    )
    if actuals.empty:
        # Without observations the leak check and every coverage check below degrade to
        # no-ops, so the run would report success having verified nothing at all.
        raise ImplausibleForecast(
            "no water temperature observations to check the run against — "
            "the leak check and every backtest would pass vacuously"
        )

    # The forecast horizon must lie beyond the last observation; if measured values show
    # up inside it, the truncation leaked and the model saw its own answer.
    leaked = df_inference.index.intersection(actuals.index)
    if len(leaked) > 0:
        raise ImplausibleForecast(
            f"{len(leaked)} forecast timestamps already have measurements — "
            "the input was not truncated correctly"
        )

    if not backtests:
        raise ImplausibleForecast(
            "the run produced no backtests, so nothing compares the model against "
            "what actually happened"
        )

    overlapping = 0
    for offset, backtest in sorted(backtests.items()):
        what = f"backtest -{offset}h"
        if backtest.forecast.empty:
            raise ImplausibleForecast(f"{what} is empty")
        _check_columns(backtest.forecast, REQUIRED_BACKTEST_COLUMNS, what)
        _check_ranges(backtest.forecast, what, required=REQUIRED_BACKTEST_COLUMNS)

        coverage = _coverage(backtest.forecast, actuals, what)
        if coverage is None:
            logger.warning("Backtest -%dh has no overlap with observations, cannot check it", offset)
            continue
        overlapping += 1

        logger.info(
            "Backtest -%dh (%s): %.0f%% of observations inside the 1-99%% band",
            offset, backtest.kind, coverage * 100,
        )
        if coverage < MIN_COVERAGE:
            raise ImplausibleForecast(
                f"backtest -{offset}h ({backtest.kind}) covered only {coverage:.0%} of "
                f"observations in its 1-99% band, expected at least {MIN_COVERAGE:.0%}"
            )

    if overlapping == 0:
        # Every backtest sat outside the measured window. Individually that is only a
        # warning — there is nothing to compare against — but if it holds for all of
        # them, the run has checked its own arithmetic and nothing else.
        raise ImplausibleForecast(
            f"none of the {len(backtests)} backtests overlapped the observations, so the "
            "run was never compared against what actually happened"
        )
