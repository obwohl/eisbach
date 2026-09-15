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

``Backtest`` is duck-typed here rather than imported. There is no import cycle to avoid —
``inference`` reaches ``archive``, ``data`` and ``model``, and none of them come back
here. The cost is ``torch``: importing ``inference`` at module scope would pull it in
via ``eisbach.model``, for a type annotation, in a module that only draws pictures.
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

#: Series colours in fixed order — forecast first, then backtests oldest-last. Not a
#: cycle: a fourth series takes the fourth slot, never a generated hue. Validated rather
#: than chosen by eye (`scripts/validate_palette.js`, light surface, adjacent pairs):
#: worst colour-vision-deficient separation ΔE 9.1, worst normal-vision ΔE 22.9, both
#: clear of the floors. The palette this replaced failed: its yellow sat outside the
#: lightness band and was ΔE 6.3 from its green under protanopia — two lines a red-blind
#: reader could not tell apart. Aqua and yellow fall below 3:1 contrast on white, which
#: obliges visible labels; every series here is named in the legend.
PLOT_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']

#: Ink, for everything that is not a series. Text and the measured line wear these, never
#: a series colour.
INK = '#1a1a19'
INK_MUTED = '#6f6e66'

#: How much history both models draw before the forecast starts. One constant, because
#: the two pictures sit under one switch and a reader flipping between them is comparing
#: forecasts, not axes — they used to show 24 h and 96 h. Two days is the shortest window
#: that still shows two whole day/night cycles, which is the rhythm the forecast continues.
HISTORY_HOURS = 48

#: The air temperature gets its own panel rather than sharing the water axis. Same unit,
#: but three to four times the range: on one axis it set the y-limits, squashed the water
#: forecast into the middle third, and crossed every band on the way. This is the
#: "two measures of different scale" case, and the answer is two panels, not two y-axes.
AIR_PANEL_RATIO = 0.28

#: The channel this module plots. The model also emits ``airtemp_96`` and
#: ``pressure_96`` quantiles, but those are inputs, not the thing anyone came to see.
CHANNEL = 'wassertemp'
MEDIAN_COL = f'{CHANNEL}_q0.5'

QUANTILE_PAIRS = [(0.01, 0.99), (0.05, 0.95), (0.25, 0.75)]
BAND_ALPHAS = [0.1, 0.15, 0.2]
QUANTILE_LABELS = ['q0.01-q0.99', 'q0.05-q0.95', 'q0.25-q0.75']

PREDICTION_PNG = 'Prediction.png'
BACKTEST_PNG = 'Prediction_Backtest.png'

#: TimesFM 3.0 emits deciles and nothing else — there is no q0.25 or q0.05 to be had —
#: so one band, the widest the deciles can make. It holds 80 % of the distribution where
#: DUET's outer band holds 98 %, which is why a narrower ribbon here does not mean a more
#: confident model. The same band on both of the candidate's images, deliberately: two
#: pictures of one forecast that disagreed about what the shading meant would be worse
#: than either alone.
TIMESFM_QUANTILE_PAIRS = [(0.1, 0.9)]
TIMESFM_BAND_ALPHAS = [0.2]
TIMESFM_QUANTILE_LABELS = ['q0.1-q0.9 (80 %)']

TIMESFM_PNG = 'Prediction_timesfm.png'
TIMESFM_BACKTEST_PNG = 'Prediction_Backtest_timesfm.png'

#: Dishonest backtests are dashed. Deliberately not a colour difference: the colour
#: cycle is already carrying the offset, and colour alone is the one cue a reader can
#: fail to perceive.
HONEST_LINESTYLE = '-'
ORACLE_LINESTYLE = '--'

#: Said on the picture rather than left as a curve that silently is not there. A gauge
#: that has not reported the tail of a window cannot be interpolated across — there is
#: nothing on the right to interpolate towards — so the backtest waits for the data.
MISSING_NOTE = 'No backtest at {offsets}: the weather for it is not fully measured yet.'

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
    # A solid hairline, one shade off the surface. Dashing a grid adds noise and reads
    # as "threshold" or "projection" — and in these pictures dashed already means
    # something: an oracle backtest. One meaning per dash.
    "grid.color": "#D8D7D2", "grid.linestyle": "-", "grid.linewidth": 0.6,
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

def _split_figure():
    """One figure, water above and air below, sharing the time axis.

    The air temperature is drawn here because it explains the shape of the water
    forecast, not because anyone came to read it, so it gets a quarter of the height, a
    muted line and no legend of its own — a single series in its own panel is named by
    its axis label. Everything the reader is actually looking at keeps the big panel and
    a y-range fitted to water alone.
    """
    fig, (water, air) = plt.subplots(
        2, 1, sharex=True, height_ratios=[1, AIR_PANEL_RATIO],
        gridspec_kw={'hspace': 0.08})
    # Adaptive, not a fixed 2.5 °C step: with the air temperature out of the way the
    # water range is a few degrees, and a fixed step that used to give five ticks now
    # gives two. Restricted to steps a reader adds up in their head.
    water.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
    air.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    water.set_ylabel('Water temperature (°C)')
    air.set_ylabel('Air (°C)', color=INK_MUTED, fontsize=9)
    air.tick_params(labelcolor=INK_MUTED, labelsize=8)
    for spine in ('top', 'right'):
        water.spines[spine].set_visible(False)
        air.spines[spine].set_visible(False)
    return fig, water, air


def _draw_air(ax, series, start, end) -> None:
    """The DWD air temperature, quietly, in its own panel."""
    ax.plot(series.index, series, color=INK_MUTED, linewidth=1.0)
    span = _span(_clip(series, start, end))
    if span is not None:
        ax.set_ylim(span[0] - 1, span[1] + 1)
    ax.set_xlabel('')


def _plot_fan(ax, df_forecast: pd.DataFrame, label: str, color: str,
              linestyle: str = HONEST_LINESTYLE, *, pairs=None, alphas=None) -> None:
    """Draw one forecast: its median line plus the nested quantile bands."""
    pairs = QUANTILE_PAIRS if pairs is None else pairs
    alphas = BAND_ALPHAS if alphas is None else alphas
    ax.plot(df_forecast.index, df_forecast[MEDIAN_COL], label=label, color=color,
            linestyle=linestyle)
    for alpha, (q_low, q_high) in zip(alphas, pairs, strict=True):
        col_low = f'{CHANNEL}_q{q_low}'
        col_high = f'{CHANNEL}_q{q_high}'
        if col_low in df_forecast.columns and col_high in df_forecast.columns:
            ax.fill_between(df_forecast.index, df_forecast[col_low], df_forecast[col_high],
                            alpha=alpha, color=color, lw=0)


def _add_band_legend_entries(ax, *, alphas=None, labels=None) -> None:
    """Add one invisible patch per quantile band, so the legend explains the shading."""
    alphas = BAND_ALPHAS if alphas is None else alphas
    labels = QUANTILE_LABELS if labels is None else labels
    for alpha, label in zip(alphas, labels, strict=True):
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
    """Put the legend below the figure, never on top of the data.

    Inside the axes it had nowhere to go: a backtest starts at the left edge and the
    forecast ends at the right, so every corner is occupied at some point in the day and
    `upper left` was sitting on a curve. Outside, it cannot collide with anything, and
    `bbox_inches='tight'` at save time keeps it in frame.
    """
    lines, labels = ax.get_legend_handles_labels()
    if getattr(ax, '_legend_artist', None) is not None:
        ax._legend_artist.remove()
    columns = 3 if len(labels) > 4 else max(1, len(labels))
    ax._legend_artist = ax.figure.legend(
        lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.06), ncol=columns,
        frameon=False, fontsize=9)


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

    fig, ax, ax_air = _split_figure()

    # Everything is plotted in naive local time; see _to_local_naive.
    df_long_plot = df_long.copy()
    df_long_plot['date'] = _to_local_naive(df_long_plot['date'])
    df_weather_plot = _localized_copy(df_weather)
    df_inference_plot = _localized_copy(df_inference)
    prepared_backtests = _prepare_backtests(backtests)

    historical = df_long_plot[df_long_plot['cols'] == CHANNEL]
    air = df_weather_plot['lufttemperatur_c']
    forecast_end = df_inference_plot.index.max()

    def draw_history():
        ax.plot(historical['date'], historical['data'], label='Measured water temperature',
                color=INK, linewidth=1.2)
        _add_band_legend_entries(ax)

    def fit(forecast_frame, backtest_frames, start):
        """The forecast sets the range with its full band; backtests only with their medians.

        A backtest is context — you read its median against the measured line — and one
        damaged run must not decide the axis for the other three. The -96h live backtest
        on this picture is a forecast really published on 11 September, while the
        154.4 °C reading was still poisoning production, and its q0.01-q0.99 runs from
        0 to 31 °C. Fitted to that, every other curve collapses into a stripe. Its band
        now runs off the frame instead, which is the honest signal that it is that wide.
        """
        span = (forecast_frame[f'{CHANNEL}_q0.01'].min(),
                forecast_frame[f'{CHANNEL}_q0.99'].max())
        for frame in backtest_frames:
            span = _widen(span, (frame[MEDIAN_COL].min(), frame[MEDIAN_COL].max()))
        visible = historical.loc[(historical['date'] >= start)
                                 & (historical['date'] <= forecast_end), 'data']
        span = _widen(span, _span(visible))
        ax.set_xlim(left=start, right=forecast_end)
        ax.set_ylim(span[0] - 0.5, span[1] + 0.5)
        _draw_air(ax_air, air, start, forecast_end)

    # ------------------------------------------------------------------
    # Image 1: the forecast on its own.
    # ------------------------------------------------------------------
    draw_history()
    _plot_fan(ax, df_inference_plot, 'Forecast', PLOT_COLORS[0])
    fit(df_inference, [], df_inference_plot.index.min() - pd.Timedelta(hours=HISTORY_HOURS))

    ax.set_title(f'Eisbach water temperature forecast\n'
                 f'Issued {_issued_label(issued_at)} · all times Europe/Berlin')
    _refresh_legend(ax)

    annotations = _annotate_peaks(ax, df_inference_plot)
    _save(fig, PREDICTION_PNG)

    # ------------------------------------------------------------------
    # Image 2: the same figure with the backtests laid over it.
    # ------------------------------------------------------------------
    for annotation in annotations:
        annotation.remove()

    drawn = []
    start = df_inference_plot.index.min() - pd.Timedelta(hours=HISTORY_HOURS)
    for i, (backtest, df_bt) in enumerate(prepared_backtests):
        color = PLOT_COLORS[i + 1] if i + 1 < len(PLOT_COLORS) else PLOT_COLORS[-1]
        linestyle = HONEST_LINESTYLE if backtest.is_honest else ORACLE_LINESTYLE
        _plot_fan(ax, df_bt, backtest.label, color, linestyle)
        if MEDIAN_COL in df_bt.columns:
            drawn.append(df_bt)
        start = min(start, df_bt.index.min())
    fit(df_inference, drawn, start)

    title = (f'Eisbach water temperature: forecast and backtests\n'
             f'Issued {_issued_label(issued_at)} · all times Europe/Berlin')
    # Only warn about the oracle when there is actually an oracle on the picture.
    if any(not backtest.is_honest for backtest, _df in prepared_backtests):
        title += f'\n{ORACLE_NOTE}'
    ax.set_title(title)
    _refresh_legend(ax)

    _save(fig, BACKTEST_PNG)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# The candidate
# --------------------------------------------------------------------------------------

def plot_timesfm(context: pd.DataFrame, future: pd.DataFrame, df_forecast: pd.DataFrame,
                 *, issued_at=None, backtests=None, missing=()) -> list[str]:
    """Write the candidate's two images and return the paths written.

    The same pair, the same layout and the same history window as the production model:
    the two pictures sit under one switch, and a reader flipping between them is
    comparing forecasts, not axes.

    ``context`` is the frame `eisbach.timesfm.context_frame` built the forecast from, of
    which only the last days are drawn, and ``future`` the weather forecast it was handed.
    ``backtests`` maps an offset in hours to an `eisbach.timesfm.Backtest`; ``missing``
    names the offsets that could not be built, which the second image says out loud
    rather than quietly showing one curve fewer.

    Only the target, the air temperature and the forecast are drawn. The past-only
    covariates are what the model reads, not what anyone came to see.
    """
    from eisbach.timesfm import KNOWN_FUTURE, TARGET

    backtests = backtests or {}
    plt.rcParams.update(PRIMER_STYLE)

    fig, ax, ax_air = _split_figure()

    forecast = _localized_copy(df_forecast)
    # Reach back far enough to cover the earliest backtest as well. A backtest drawn over
    # a stretch with no measured line beside it cannot be judged against anything, which
    # is the one thing the second image exists for. Image 1 crops back to HISTORY_HOURS
    # through its x-limits, so this costs it nothing.
    needed = max([HISTORY_HOURS,
                  *(b.offset_hours + HISTORY_HOURS for b in backtests.values())])
    recent = _localized_copy(context.tail(needed))
    # The air temperature the model was told: measured before the anchor, forecast after
    # it — the same line the production image draws, from the same source.
    air = pd.concat([recent[KNOWN_FUTURE[0]],
                     _localized_copy(future)[KNOWN_FUTURE[0]]]).sort_index()
    forecast_end = forecast.index.max()

    def draw_history():
        ax.plot(recent.index, recent[TARGET], label='Measured water temperature',
                color=INK, linewidth=1.2)
        _add_band_legend_entries(ax, alphas=TIMESFM_BAND_ALPHAS,
                                 labels=TIMESFM_QUANTILE_LABELS)

    def fan(frame, label, color, linestyle=HONEST_LINESTYLE):
        _plot_fan(ax, frame, label, color, linestyle,
                  pairs=TIMESFM_QUANTILE_PAIRS, alphas=TIMESFM_BAND_ALPHAS)

    def fit(backtest_frames, start):
        """As on the production image: the forecast's band, the backtests' medians."""
        span = (float(forecast[f'{CHANNEL}_q0.1'].min()),
                float(forecast[f'{CHANNEL}_q0.9'].max()))
        for frame in backtest_frames:
            span = _widen(span, (float(frame[MEDIAN_COL].min()), float(frame[MEDIAN_COL].max())))
        span = _widen(span, _span(_clip(recent[TARGET], start, forecast_end)))
        ax.set_xlim(left=start, right=forecast_end)
        ax.set_ylim(span[0] - 0.5, span[1] + 0.5)
        _draw_air(ax_air, air, start, forecast_end)

    # ------------------------------------------------------------------
    # Image 1: the forecast on its own.
    # ------------------------------------------------------------------
    draw_history()
    fan(forecast, 'TimesFM forecast', PLOT_COLORS[0])
    fit([], forecast.index.min() - pd.Timedelta(hours=HISTORY_HOURS))
    ax.set_title('Eisbach water temperature — TimesFM\n'
                 f'Issued {_issued_label(issued_at)} · all times Europe/Berlin')
    _refresh_legend(ax)
    annotations = _annotate_peaks(ax, forecast)
    _save(fig, TIMESFM_PNG)

    # ------------------------------------------------------------------
    # Image 2: the same figure with the backtests laid over it.
    # ------------------------------------------------------------------
    for annotation in annotations:
        annotation.remove()

    drawn = []
    start = forecast.index.min() - pd.Timedelta(hours=HISTORY_HOURS)
    for i, offset in enumerate(sorted(backtests)):
        backtest = backtests[offset]
        frame = _localized_copy(backtest.forecast)
        color = PLOT_COLORS[i + 1] if i + 1 < len(PLOT_COLORS) else PLOT_COLORS[-1]
        fan(frame, backtest.label, color,
            HONEST_LINESTYLE if backtest.is_honest else ORACLE_LINESTYLE)
        drawn.append(frame)
        start = min(start, frame.index.min())
    fit(drawn, start)

    title = ('Eisbach water temperature — TimesFM with backtests\n'
             f'Issued {_issued_label(issued_at)} · all times Europe/Berlin')
    if any(not b.is_honest for b in backtests.values()):
        title += f'\n{ORACLE_NOTE}'
    if missing:
        title += ('\n' + MISSING_NOTE.format(
            offsets=', '.join(f'-{h}h' for h in sorted(missing))))
    ax.set_title(title)
    _refresh_legend(ax)
    _save(fig, TIMESFM_BACKTEST_PNG)

    plt.close(fig)
    return [TIMESFM_PNG, TIMESFM_BACKTEST_PNG]
