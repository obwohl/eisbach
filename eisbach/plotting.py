"""Render the two forecast images.

Both PNGs are drawn on a single figure, in order:

``Prediction.png``
    The main forecast alone, with its quantile fan and annotated daily maxima.

``Prediction_Backtest.png``
    The same figure with the backtests drawn over it, so the forecast can be judged
    against what the model would have said days ago.

Backtests are not all equally trustworthy. An ``oracle`` backtest was computed from the
weather that *actually occurred*, which hands the model a perfect forecast and flatters
it; ``live`` and ``replay`` backtests saw only what was knowable at the time. That
distinction is carried by :class:`eisbach.inference.Backtest` and must survive into the
picture, so oracle backtests are drawn dashed and the plot carries a note whenever one
is present.

``Backtest`` is duck-typed here rather than imported: ``inference`` imports ``data`` and
would, on a module-scope import, close a cycle back through this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from cycler import cycler
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from eisbach.inference import Backtest

logger = logging.getLogger(__name__)

PLOT_COLORS = ['#1771F1', '#F85C50', '#35D073', '#FFC11E', '#8E44AD']

#: The channel this module plots. The model also emits ``airtemp_96`` and
#: ``pressure_96`` quantiles, but those are inputs, not the thing anyone came to see.
CHANNEL = 'wassertemp'
MEDIAN_COL = f'{CHANNEL}_q0.5'

QUANTILE_PAIRS = [(0.01, 0.99), (0.05, 0.95), (0.25, 0.75)]
BAND_ALPHAS = [0.1, 0.15, 0.2]
QUANTILE_LABELS = ['q0.01-q0.99', 'q0.05-q0.95', 'q0.25-q0.75']

PREDICTION_PNG = 'Prediction.png'
BACKTEST_PNG = 'Prediction_Backtest.png'

#: Dishonest backtests are dashed. Deliberately not a colour difference: the colour
#: cycle is already carrying the offset, and colour alone is the one cue a reader can
#: fail to perceive.
HONEST_LINESTYLE = '-'
ORACLE_LINESTYLE = '--'

ORACLE_NOTE = (
    'Dashed = oracle backtest: computed with the weather that actually occurred, '
    'so it flatters the model.'
)

#: Matplotlib rcParams for the house style.
PRIMER_STYLE = {
    "lines.linewidth": 1.0, "lines.linestyle": "-", "font.family": "sans-serif",
    "font.size": 10, "text.color": "#231F20", "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#231F20", "axes.linewidth": 0.8, "axes.grid": True,
    "axes.labelsize": 10, "axes.labelweight": "normal", "axes.labelcolor": "#231F20",
    "axes.prop_cycle": cycler(color=PLOT_COLORS),
    "xtick.major.size": 2, "xtick.minor.size": 1, "xtick.major.width": 0.8,
    "xtick.minor.width": 0.6, "xtick.major.top": True, "xtick.major.bottom": True,
    "xtick.minor.top": True, "xtick.minor.bottom": True, "xtick.color": "#231F20",
    "xtick.labelsize": 8,
    "ytick.major.size": 2, "ytick.minor.size": 1, "ytick.major.width": 0.8,
    "ytick.minor.width": 0.6, "ytick.color": "#231F20", "ytick.major.left": True,
    "ytick.major.right": True, "ytick.minor.left": True, "ytick.minor.right": True,
    "grid.color": "#231F20", "grid.linestyle": ":", "grid.linewidth": 0.4,
    "grid.alpha": 1.0, "legend.frameon": False, "legend.edgecolor": "#231F20",
    "figure.figsize": [12, 10], "figure.dpi": 96, "figure.facecolor": "#FFFFFF",
    "figure.edgecolor": "#FFFFFF",
}


# --------------------------------------------------------------------------------------
# Time handling
# --------------------------------------------------------------------------------------

def _to_local_naive(series_or_index):
    """Return the same timestamps as naive local (Europe/Berlin) wall-clock time.

    Matplotlib is unreliable with tz-aware datetimes, so the conversion is done here and
    the offset dropped afterwards. Naive input is assumed to be UTC, which is what
    everything upstream of this module produces.
    """
    is_series = hasattr(series_or_index, 'dt')
    tzinfo = series_or_index.dt.tz if is_series else series_or_index.tz

    if tzinfo is None:
        aware = (series_or_index.dt if is_series else series_or_index).tz_localize('UTC')
    else:
        aware = series_or_index

    if is_series:
        return aware.dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
    return aware.tz_convert('Europe/Berlin').tz_localize(None)


def _localized_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Copy a frame with its index converted to naive local time."""
    out = df.copy()
    out.index = _to_local_naive(out.index)
    return out


def _prepare_backtests(backtests: dict[int, Backtest]) -> list[tuple[Backtest, pd.DataFrame]]:
    """Pair each non-empty backtest with its forecast in local naive time.

    ``Backtest`` is only duck-typed here (``.forecast``, ``.label``, ``.is_honest``);
    see the module docstring for why it is not imported.
    """
    prepared = []
    for _offset, backtest in backtests.items():
        if backtest.forecast.empty:
            continue
        prepared.append((backtest, _localized_copy(backtest.forecast)))
    return prepared


# --------------------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------------------

def _plot_fan(ax, df_forecast: pd.DataFrame, label: str, color: str,
              linestyle: str = HONEST_LINESTYLE) -> None:
    """Draw one forecast: its median line plus the nested quantile bands."""
    ax.plot(df_forecast.index, df_forecast[MEDIAN_COL], label=label, color=color,
            linestyle=linestyle)
    for alpha, (q_low, q_high) in zip(BAND_ALPHAS, QUANTILE_PAIRS, strict=True):
        col_low = f'{CHANNEL}_q{q_low}'
        col_high = f'{CHANNEL}_q{q_high}'
        if col_low in df_forecast.columns and col_high in df_forecast.columns:
            ax.fill_between(df_forecast.index, df_forecast[col_low], df_forecast[col_high],
                            alpha=alpha, color=color, lw=0)


def _add_band_legend_entries(ax) -> None:
    """Add one invisible patch per quantile band, so the legend explains the shading."""
    for alpha, label in zip(BAND_ALPHAS, QUANTILE_LABELS, strict=True):
        ax.fill_between([], [], [], color='gray', alpha=alpha, label=label)


def _annotate_peaks(ax, df_forecast: pd.DataFrame) -> list:
    """Label the significant maxima of the median forecast; return the artists.

    The artists are returned so the caller can remove them again: the annotations only
    belong on the forecast-only image, where there is room for them.
    """
    if MEDIAN_COL not in df_forecast.columns:
        return []

    # At least 18 h between peaks, so a single day cannot be counted twice; prominence
    # keeps tiny ripples out.
    peaks, _ = find_peaks(df_forecast[MEDIAN_COL], distance=18, prominence=0.2)

    artists = []
    for peak_idx in peaks:
        max_row = df_forecast.iloc[peak_idx]
        max_val = max_row[MEDIAN_COL]
        max_time_local = max_row.name  # already naive local time
        artists.append(ax.annotate(
            f"Max: {max_val:.1f}°C\n{max_time_local.strftime('%H:%M')}",
            xy=(max_time_local, max_val),
            xytext=(0, 20), textcoords="offset points",
            ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0, alpha=0.7),
        ))
    return artists


def _refresh_legend(ax) -> None:
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left')


def _save(fig, path: str) -> None:
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(),
                edgecolor='none')
    logger.info("Plot saved to: %s", path)


# --------------------------------------------------------------------------------------
# Y-limits
# --------------------------------------------------------------------------------------

def _span(series) -> tuple[float, float] | None:
    """``(min, max)`` of a series, or ``None`` when there is nothing in it."""
    return None if series.empty else (series.min(), series.max())


def _widen(span: tuple[float, float], other: tuple[float, float] | None) -> tuple[float, float]:
    """Grow ``span`` to also contain ``other``; a missing ``other`` changes nothing."""
    if other is None:
        return span
    return min(span[0], other[0]), max(span[1], other[1])


def _clip(series, start, end):
    """The part of a series that falls inside the visible x-range."""
    return series.loc[(series.index >= start) & (series.index <= end)]


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def _issued_label(issued_at) -> str:
    """Human-readable issue time for the plot titles, in local time."""
    stamp = pd.Timestamp.now(tz="UTC") if issued_at is None else pd.Timestamp(issued_at)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M")


def plot_forecasts(df_long, df_weather, df_inference, backtests=None, issued_at=None) -> None:
    """Write ``Prediction.png`` and ``Prediction_Backtest.png``.

    ``backtests`` maps an offset in hours to a :class:`eisbach.inference.Backtest`.
    """
    if backtests is None:
        backtests = {}

    plt.rcParams.update(PRIMER_STYLE)
    colors = PRIMER_STYLE['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

    # Everything is plotted in naive local time; see _to_local_naive.
    df_long_plot = df_long.copy()
    df_long_plot['date'] = _to_local_naive(df_long_plot['date'])
    df_weather_plot = _localized_copy(df_weather)
    df_inference_plot = _localized_copy(df_inference)
    prepared_backtests = _prepare_backtests(backtests)

    historical = df_long_plot[df_long_plot['cols'] == CHANNEL]
    ax.plot(historical['date'], historical['data'], label='Measured water temperature',
            color='black', linestyle='--')
    _add_band_legend_entries(ax)

    # ------------------------------------------------------------------
    # Image 1: the forecast on its own.
    # ------------------------------------------------------------------
    _plot_fan(ax, df_inference_plot, 'Forecast', colors[0])
    ax.plot(df_weather_plot.index, df_weather_plot['lufttemperatur_c'], label='Air Temp (DWD)',
            color='purple', linestyle=':', linewidth=1.5, alpha=0.6)

    forecast_start = df_inference_plot.index.min() - pd.Timedelta(days=1)  # 1 day of history
    forecast_end = df_inference_plot.index.max()
    ax.set_xlim(left=forecast_start, right=forecast_end)

    inference_span = (df_inference[f'{CHANNEL}_q0.01'].min(), df_inference[f'{CHANNEL}_q0.99'].max())

    visible_weather = _clip(df_weather_plot['lufttemperatur_c'], forecast_start, forecast_end)
    visible_history = historical.loc[
        (historical['date'] >= forecast_start) & (historical['date'] <= forecast_end), 'data'
    ]
    y_span = _widen(inference_span, _span(visible_weather))
    y_span = _widen(y_span, _span(visible_history))
    ax.set_ylim(y_span[0] - 0.5, y_span[1] + 0.5)

    ax.set_title(f'Eisbach water temperature forecast\n'
                 f'Issued {_issued_label(issued_at)} · all times Europe/Berlin')
    ax.set_xlabel('')
    ax.set_ylabel('Temperature (°C)')
    _refresh_legend(ax)

    annotations = _annotate_peaks(ax, df_inference_plot)
    _save(fig, PREDICTION_PNG)

    # ------------------------------------------------------------------
    # Image 2: the same figure with the backtests laid over it.
    # ------------------------------------------------------------------
    for annotation in annotations:
        annotation.remove()

    for i, (backtest, df_bt) in enumerate(prepared_backtests):
        color = colors[i + 1] if i + 1 < len(colors) else colors[-1]
        linestyle = HONEST_LINESTYLE if backtest.is_honest else ORACLE_LINESTYLE
        _plot_fan(ax, df_bt, backtest.label, color, linestyle)

    backtest_start = min(
        (df_bt.index.min() for _bt, df_bt in prepared_backtests),
        default=df_inference_plot.index.min(),
    )
    backtest_end = df_inference_plot.index.max()  # the last point of the main forecast
    ax.set_xlim(left=backtest_start, right=backtest_end)

    y_span_bt = inference_span
    for _bt, df_bt in prepared_backtests:
        if f'{CHANNEL}_q0.01' in df_bt.columns and f'{CHANNEL}_q0.99' in df_bt.columns:
            y_span_bt = _widen(
                y_span_bt, (df_bt[f'{CHANNEL}_q0.01'].min(), df_bt[f'{CHANNEL}_q0.99'].max()),
            )
    y_span_bt = _widen(
        y_span_bt, _span(_clip(df_weather_plot['lufttemperatur_c'], backtest_start, backtest_end)),
    )
    ax.set_ylim(y_span_bt[0] - 0.5, y_span_bt[1] + 0.5)

    title = (f'Eisbach water temperature: forecast and backtests\n'
             f'Issued {_issued_label(issued_at)} · all times Europe/Berlin')
    # Only warn about the oracle when there is actually an oracle on the picture.
    if any(not backtest.is_honest for backtest, _df in prepared_backtests):
        title += f'\n{ORACLE_NOTE}'
    ax.set_title(title)
    _refresh_legend(ax)

    _save(fig, BACKTEST_PNG)
    plt.close(fig)
