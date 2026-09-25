"""Render the forecast images of both models, on shared axes.

Each model gets the same pair:

``Prediction.png`` / ``Prediction_timesfm.png``
    The forecast alone, with its q0.1-q0.9 band and annotated daily maxima.

``Prediction_Backtest.png`` / ``Prediction_Backtest_timesfm.png``
    The same figure with the backtests drawn over it, so the forecast can be judged
    against what the model would have said days ago.

The page shows one model at a time under a switch, so the two images of a pair must be
interchangeable: the same band, the same x- and y-limits, and the axes at the same pixel
position. The limits are the union over both models — whichever is wider decides, so
neither runs out of frame — and the layout is fixed rather than fitted to the text, which
is what `bbox_inches='tight'` used to do, moving the axes whenever one title had a line
more than the other. A narrower ribbon then reads as a sharper model, not as a zoom.

Backtests are not all equally trustworthy. An ``oracle`` backtest was computed from the
weather that *actually occurred*, which hands the model a perfect forecast and flatters
it; ``live`` and ``replay`` backtests saw only what was knowable at the time. That
distinction must survive into the picture, so oracle backtests are drawn dashed and the
plot carries a note whenever one is present.

``Backtest`` is duck-typed here rather than imported (``.forecast``, ``.label``,
``.is_honest``, ``.offset_hours``). Importing ``inference`` at module scope would pull in
``torch`` via ``eisbach.model``, for a type annotation, in a module that only draws.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
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

#: One band for both models, and the widest one both can draw: TimesFM emits deciles and
#: nothing beyond them. DUET used to show three nested bands up to q0.01-q0.99, which
#: made the two pictures disagree about what the shading meant. Its q0.1 and q0.9 are
#: computed exactly (`eisbach.model.BAND_QUANTILES`); forecasts archived before that
#: carry neither and have them interpolated, see `_with_band`.
BAND = (0.1, 0.9)
BAND_ALPHA = 0.2
BAND_LABEL = 'q0.1-q0.9 (80 %)'
BAND_LOW, BAND_HIGH = (f'{CHANNEL}_q{q}' for q in BAND)

PREDICTION_PNG = 'Prediction.png'
BACKTEST_PNG = 'Prediction_Backtest.png'
TIMESFM_PNG = 'Prediction_timesfm.png'
TIMESFM_BACKTEST_PNG = 'Prediction_Backtest_timesfm.png'

#: Where the axes sit in the figure, fixed, so both models' images put them on the same
#: pixels. The top leaves room for a four-line title (heading, issue time, oracle note,
#: missing note); the bottom for the dates and a two-row legend.
LAYOUT = dict(left=0.07, right=0.98, top=0.87, bottom=0.15)

#: Dishonest backtests are dashed. Deliberately not a colour difference: the colour
#: cycle is already carrying the offset, and colour alone is the one cue a reader can
#: fail to perceive.
HONEST_LINESTYLE = '-'
ORACLE_LINESTYLE = '--'

#: Said on the picture rather than left as a curve that silently is not there. Either a
#: gauge has not reported the tail of the window yet, or it never reported part of it —
#: rain is never interpolated, so a single outage anywhere in the four days is enough.
#: "Not measured yet" was the old wording and was wrong for the second case.
MISSING_NOTE = 'No backtest at {offsets}: the measured data for that window is incomplete.'

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



# --------------------------------------------------------------------------------------
# Band
# --------------------------------------------------------------------------------------

def _with_band(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame with q0.1 and q0.9, interpolating them where they were never stored.

    DUET forecasts archived before it emitted the band levels carry only its seven scored
    quantiles, and the backtests read them back for twelve days. Linear in the quantile
    level, the rule `eisbach.verification.to_grid` puts DUET on the decile grid with, so
    the picture and the comparison agree. It is slightly wider than the exact band, and
    only ever for runs made before the exact one existed.
    """
    if frame.empty or all(c in frame.columns and frame[c].notna().all()
                          for c in (BAND_LOW, BAND_HIGH)):
        return frame
    levels = sorted(float(c.split('_q')[-1]) for c in frame.columns
                    if c.startswith(f'{CHANNEL}_q') and frame[c].notna().all())
    if not levels or min(levels) > BAND[0] or max(levels) < BAND[1]:
        return frame
    values = np.sort(frame[[f'{CHANNEL}_q{q}' for q in levels]].to_numpy(float), axis=1)
    out = frame.copy()
    for q, column in zip(BAND, (BAND_LOW, BAND_HIGH), strict=True):
        out[column] = [np.interp(q, levels, row) for row in values]
    return out


# --------------------------------------------------------------------------------------
# One model's picture, before it is drawn
# --------------------------------------------------------------------------------------

@dataclass
class ModelView:
    """Everything one model's pair of images shows, already in naive local time."""

    name: str
    forecast_png: str
    backtest_png: str
    measured: pd.Series
    air: pd.Series
    forecast: pd.DataFrame
    issued_at: object = None
    backtests: list = field(default_factory=list)
    missing: tuple = ()

    @property
    def paths(self) -> list[str]:
        return [self.forecast_png, self.backtest_png]


def _prepare_backtests(backtests) -> list[tuple[Backtest, pd.DataFrame]]:
    """Pair each non-empty backtest with its banded forecast in local time, oldest last."""
    prepared = []
    for _offset, backtest in sorted((backtests or {}).items()):
        if backtest.forecast.empty:
            continue
        prepared.append((backtest, _with_band(_localized_copy(backtest.forecast))))
    return prepared


def duet_view(df_long, df_weather, df_inference, backtests=None, issued_at=None) -> ModelView:
    """The production model, from what `run_inference` returns."""
    history = df_long[df_long['cols'] == CHANNEL]
    measured = pd.Series(history['data'].to_numpy(),
                         index=pd.DatetimeIndex(_to_local_naive(history['date'])))
    return ModelView(
        name='Proprietary model', forecast_png=PREDICTION_PNG, backtest_png=BACKTEST_PNG,
        measured=measured.sort_index(),
        air=_localized_copy(df_weather)['lufttemperatur_c'],
        forecast=_with_band(_localized_copy(df_inference)),
        issued_at=issued_at, backtests=_prepare_backtests(backtests))


def timesfm_view(context, future, df_forecast, *, issued_at=None, backtests=None,
                 missing=()) -> ModelView:
    """The candidate, from the frame it forecast on and the weather forecast it was handed.

    The air line is what the model was told: measured before the anchor, forecast after
    it. Only the target and the air temperature are drawn — the past-only covariates are
    what the model reads, not what anyone came to see.
    """
    from eisbach.timesfm import KNOWN_FUTURE, TARGET

    recent = _localized_copy(context)
    air = pd.concat([recent[KNOWN_FUTURE[0]],
                     _localized_copy(future)[KNOWN_FUTURE[0]]]).sort_index()
    return ModelView(
        name='TimesFM', forecast_png=TIMESFM_PNG, backtest_png=TIMESFM_BACKTEST_PNG,
        measured=recent[TARGET], air=air, forecast=_localized_copy(df_forecast),
        issued_at=issued_at, backtests=_prepare_backtests(backtests),
        missing=tuple(sorted(missing)))


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
    fig.subplots_adjust(**LAYOUT)
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


def _draw_air(ax, series) -> None:
    """The air temperature, quietly, in its own panel. Limits are set by the caller."""
    ax.plot(series.index, series, color=INK_MUTED, linewidth=1.0)
    ax.set_xlabel('')


def _plot_fan(ax, frame: pd.DataFrame, label: str, color: str,
              linestyle: str = HONEST_LINESTYLE) -> None:
    """Draw one forecast: its median line plus the q0.1-q0.9 band."""
    ax.plot(frame.index, frame[MEDIAN_COL], label=label, color=color, linestyle=linestyle)
    if BAND_LOW in frame.columns and BAND_HIGH in frame.columns:
        ax.fill_between(frame.index, frame[BAND_LOW], frame[BAND_HIGH],
                        alpha=BAND_ALPHA, color=color, lw=0)


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
    `LAYOUT` leaves room for two rows of it.
    """
    lines, labels = ax.get_legend_handles_labels()
    if getattr(ax, '_legend_artist', None) is not None:
        ax._legend_artist.remove()
    columns = 3 if len(labels) > 4 else max(1, len(labels))
    ax._legend_artist = ax.figure.legend(
        lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.06), ncol=columns,
        frameon=False, fontsize=9)


def _save(fig, path: str) -> None:
    # No `bbox_inches='tight'`: it crops to the content, so a longer title would move
    # the axes and the two models' images would no longer line up. See `LAYOUT`.
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    logger.info("Plot saved to: %s", path)


# --------------------------------------------------------------------------------------
# Limits, shared by every model on the page
# --------------------------------------------------------------------------------------

def _span(series) -> tuple[float, float] | None:
    """``(min, max)`` of a series, or ``None`` when there is nothing in it."""
    series = series.dropna()
    return None if series.empty else (float(series.min()), float(series.max()))


def _widen(span, other):
    """Grow ``span`` to also contain ``other``; a missing side changes nothing."""
    if other is None:
        return span
    if span is None:
        return other
    return min(span[0], other[0]), max(span[1], other[1])


def _clip(series, start, end):
    """The part of a series that falls inside the visible x-range."""
    return series.loc[(series.index >= start) & (series.index <= end)]


@dataclass(frozen=True)
class Limits:
    start: pd.Timestamp
    end: pd.Timestamp
    water: tuple[float, float]
    air: tuple[float, float] | None

    def apply(self, ax, ax_air) -> None:
        ax.set_xlim(left=self.start, right=self.end)
        ax.set_ylim(self.water[0] - 0.5, self.water[1] + 0.5)
        if self.air is not None:
            ax_air.set_ylim(self.air[0] - 1, self.air[1] + 1)


def _limits(views: list[ModelView], *, with_backtests: bool) -> Limits:
    """One set of limits every model's image of this kind is drawn with.

    The union: a model whose band runs wider must not run out of frame, and the other is
    then drawn with the same room around it, so switching changes the curves and nothing
    else. Each forecast counts with its full band; a backtest only with its median. A
    backtest is context — you read its median against the measured line — and one
    damaged run must not decide the axis for the rest: the -96h live backtest of
    11 September, made while a 154.4 °C reading was poisoning production, has a band from
    0 to 31 °C. Its band runs off the frame instead, which is the honest signal that it
    is that wide.
    """
    start = min(v.forecast.index.min() for v in views) - pd.Timedelta(hours=HISTORY_HOURS)
    if with_backtests:
        for view in views:
            for _backtest, frame in view.backtests:
                start = min(start, frame.index.min())
    end = max(v.forecast.index.max() for v in views)
    water = air = None
    for view in views:
        columns = [c for c in (BAND_LOW, BAND_HIGH, MEDIAN_COL) if c in view.forecast.columns]
        water = _widen(water, _span(view.forecast[columns].stack()))
        water = _widen(water, _span(_clip(view.measured, start, end)))
        if with_backtests:
            for _backtest, frame in view.backtests:
                water = _widen(water, _span(frame[MEDIAN_COL]))
        air = _widen(air, _span(_clip(view.air, start, end)))
    return Limits(start, end, water, air)


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------

def _issued_label(issued_at) -> str:
    """Human-readable issue time for the plot titles, in local time."""
    stamp = pd.Timestamp.now(tz="UTC") if issued_at is None else pd.Timestamp(issued_at)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M")


def _draw(view: ModelView, limits: Limits, *, with_backtests: bool, path: str) -> None:
    fig, ax, ax_air = _split_figure()
    ax.plot(view.measured.index, view.measured, label='Measured water temperature',
            color=INK, linewidth=1.2)
    ax.fill_between([], [], [], color='gray', alpha=BAND_ALPHA, label=BAND_LABEL)
    _plot_fan(ax, view.forecast, f'{view.name} forecast', PLOT_COLORS[0])
    _draw_air(ax_air, view.air)

    title = f'Eisbach water temperature — {view.name}'
    if with_backtests:
        title += ' with backtests'
        for i, (backtest, frame) in enumerate(view.backtests):
            color = PLOT_COLORS[min(i + 1, len(PLOT_COLORS) - 1)]
            _plot_fan(ax, frame, backtest.label, color,
                      HONEST_LINESTYLE if backtest.is_honest else ORACLE_LINESTYLE)
    title += f'\nIssued {_issued_label(view.issued_at)} · all times Europe/Berlin'
    if with_backtests:
        # Only warn about the oracle when there is actually an oracle on the picture.
        if any(not backtest.is_honest for backtest, _frame in view.backtests):
            title += f'\n{ORACLE_NOTE}'
        if view.missing:
            title += '\n' + MISSING_NOTE.format(
                offsets=', '.join(f'-{h}h' for h in view.missing))
    else:
        # Only on the forecast-only image, where there is room for them.
        _annotate_peaks(ax, view.forecast)

    limits.apply(ax, ax_air)
    ax.set_title(title)
    _refresh_legend(ax)
    _save(fig, path)
    plt.close(fig)


def plot_models(views: list[ModelView]) -> list[str]:
    """Write every model's pair of images on shared limits; return the paths written."""
    plt.rcParams.update(PRIMER_STYLE)
    written = []
    for with_backtests in (False, True):
        limits = _limits(views, with_backtests=with_backtests)
        for view in views:
            path = view.backtest_png if with_backtests else view.forecast_png
            _draw(view, limits, with_backtests=with_backtests, path=path)
            written.append(path)
    return written


def plot_forecasts(df_long, df_weather, df_inference, backtests=None, issued_at=None) -> None:
    """The production model's pair alone, on limits fitted to it."""
    plot_models([duet_view(df_long, df_weather, df_inference, backtests, issued_at)])


def plot_timesfm(context, future, df_forecast, *, issued_at=None, backtests=None,
                 missing=()) -> list[str]:
    """The candidate's pair alone, on limits fitted to it."""
    view = timesfm_view(context, future, df_forecast, issued_at=issued_at,
                        backtests=backtests, missing=missing)
    return plot_models([view])


def remove(paths) -> None:
    """Delete images a failed render may have left half-written."""
    for path in paths:
        Path(path).unlink(missing_ok=True)
