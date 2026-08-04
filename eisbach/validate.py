"""Sanity checks on a completed run.

These are not unit tests — they need a real forecast — but they are the difference
between publishing a plausible plot and publishing a broken one. The pipeline runs
unattended twice a day, so a forecast that comes out at 300 °C, or one that quietly
contains no water temperature at all, needs to fail the run rather than get committed.

Called from ``main.py`` after inference and before plotting.
"""

from __future__ import annotations

import logging

import pandas as pd

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

#: An honest backtest should contain the truth inside its 1–99 % band most of the time.
#: Well below the nominal 98 % because the band is only sampled at the extremes and the
#: sample is short; this catches a badly miscalibrated model, not a slightly wide one.
MIN_COVERAGE = 0.80


def _check_ranges(df: pd.DataFrame, what: str) -> None:
    for channel, (low, high) in PLAUSIBLE_RANGES.items():
        for column in [c for c in df.columns if c.startswith(f"{channel}_q")]:
            series = df[column]
            if series.isna().any():
                raise ImplausibleForecast(f"{what}: {column} contains NaN")
            if series.min() < low or series.max() > high:
                raise ImplausibleForecast(
                    f"{what}: {column} ranges {series.min():.1f}..{series.max():.1f}, "
                    f"outside the plausible {low}..{high}"
                )


def _coverage(backtest_forecast: pd.DataFrame, actuals: pd.Series) -> float | None:
    """Fraction of observations that fell inside the 1–99 % band."""
    overlap = backtest_forecast.index.intersection(actuals.index)
    if len(overlap) == 0:
        return None
    truth = actuals.loc[overlap]
    low = backtest_forecast.loc[overlap, "wassertemp_q0.01"]
    high = backtest_forecast.loc[overlap, "wassertemp_q0.99"]
    return float(((truth >= low) & (truth <= high)).mean())


def validate_run(df_inference: pd.DataFrame, backtests: dict, df_long: pd.DataFrame) -> None:
    """Raise :class:`ImplausibleForecast` if the run produced something unusable."""
    if df_inference.empty:
        raise ImplausibleForecast("the forecast is empty")
    _check_ranges(df_inference, "forecast")

    actuals = (
        df_long[df_long["cols"] == "wassertemp"]
        .dropna(subset=["data"])
        .set_index("date")["data"]
    )

    # The forecast horizon must lie beyond the last observation; if measured values show
    # up inside it, the truncation leaked and the model saw its own answer.
    leaked = df_inference.index.intersection(actuals.index)
    if len(leaked) > 0:
        raise ImplausibleForecast(
            f"{len(leaked)} forecast timestamps already have measurements — "
            "the input was not truncated correctly"
        )

    for offset, backtest in backtests.items():
        _check_ranges(backtest.forecast, f"backtest -{offset}h")

        coverage = _coverage(backtest.forecast, actuals)
        if coverage is None:
            logger.warning("Backtest -%dh has no overlap with observations, cannot check it", offset)
            continue

        logger.info(
            "Backtest -%dh (%s): %.0f%% of observations inside the 1-99%% band",
            offset, backtest.kind, coverage * 100,
        )
        if coverage < MIN_COVERAGE:
            raise ImplausibleForecast(
                f"backtest -{offset}h ({backtest.kind}) covered only {coverage:.0%} of "
                f"observations in its 1-99% band, expected at least {MIN_COVERAGE:.0%}"
            )
