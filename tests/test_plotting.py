"""Tests for the plotting layer.

No network, no model: the frames are synthetic but match the shapes the real pipeline
produces. ``Backtest`` is stood in for locally so this test never imports
``eisbach.inference`` (and therefore never imports torch).
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from eisbach.plotting import (  # noqa: E402
    ModelView,
    _with_band,
    duet_view,
    plot_forecasts,
    plot_models,
)

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
CHANNELS = ["wassertemp", "airtemp_96", "pressure_96"]

REFERENCE = pd.Timestamp("2026-04-01 00:00", tz="UTC")


@dataclass(frozen=True)
class FakeBacktest:
    """Stand-in for :class:`eisbach.inference.Backtest` with the same surface."""

    offset_hours: int
    reference_time: pd.Timestamp
    forecast: pd.DataFrame
    kind: str
    covariate_source: str = "dwd_observed"

    @property
    def is_honest(self) -> bool:
        return self.kind != "oracle"

    @property
    def label(self) -> str:
        suffix = "" if self.is_honest else ", perfect weather"
        return f"Backtest -{self.offset_hours}h ({self.kind}{suffix})"


def _quantile_frame(index: pd.DatetimeIndex, base: float = 12.0) -> pd.DataFrame:
    """A forecast frame: one column per channel per quantile, indexed by target time."""
    hours = pd.Series(range(len(index)), index=index, dtype=float)
    # A daily triangle wave, so find_peaks has real maxima to latch onto.
    shape = base + (12.0 - (hours % 24 - 12).abs()) / 6.0
    data = {}
    for channel in CHANNELS:
        for q in QUANTILES:
            data[f"{channel}_q{q}"] = shape + (q - 0.5) * 4.0
    return pd.DataFrame(data, index=index)


@pytest.fixture
def frames():
    history_index = pd.date_range(REFERENCE - pd.Timedelta(days=14), REFERENCE, freq="h", tz="UTC")
    future_index = pd.date_range(
        REFERENCE + pd.Timedelta(hours=1), periods=96, freq="h", tz="UTC",
    )

    rows = []
    for channel in CHANNELS:
        offset = {"wassertemp": 0.0, "airtemp_96": 3.0, "pressure_96": 1000.0}[channel]
        for i, ts in enumerate(history_index):
            rows.append({"date": ts, "cols": channel, "data": offset + 10.0 + (i % 24) / 12})
    df_long = pd.DataFrame(rows)
    df_long["cols"] = pd.Categorical(df_long["cols"], categories=CHANNELS, ordered=True)
    df_long = df_long.sort_values(by=["cols", "date"])

    weather_index = pd.date_range(
        REFERENCE - pd.Timedelta(days=14), REFERENCE + pd.Timedelta(days=5), freq="h", tz="UTC",
    )
    df_weather = pd.DataFrame(
        {"lufttemperatur_c": [8.0 + (i % 24) / 6 for i in range(len(weather_index))]},
        index=weather_index,
    )

    df_inference = _quantile_frame(future_index)
    return df_long, df_weather, df_inference


def _backtest(kind: str, offset_hours: int) -> FakeBacktest:
    reference = REFERENCE - pd.Timedelta(hours=offset_hours)
    index = pd.date_range(reference + pd.Timedelta(hours=1), periods=96, freq="h", tz="UTC")
    return FakeBacktest(
        offset_hours=offset_hours,
        reference_time=reference,
        forecast=_quantile_frame(index, base=11.0),
        kind=kind,
    )


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run with ``tmp_path`` as the working directory, so PNGs never land in the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def keep_figure(monkeypatch):
    """Stop ``plot_forecasts`` closing its figure, so the axes stay inspectable."""
    captured = {}

    real_close = plt.close

    def _capture(fig=None):
        captured["fig"] = fig
        # Deliberately do not close: the test reads the artists afterwards.

    monkeypatch.setattr(plt, "close", _capture)
    yield captured
    real_close("all")


def test_writes_both_pngs_and_no_html(frames, in_tmp_cwd):
    df_long, df_weather, df_inference = frames
    backtests = {96: _backtest("live", 96), 192: _backtest("oracle", 192)}

    plot_forecasts(df_long, df_weather, df_inference, backtests, pd.Timestamp("2026-04-01 00:00", tz="UTC"))

    for name in ("Prediction.png", "Prediction_Backtest.png"):
        png = in_tmp_cwd / name
        assert png.exists(), f"{name} was not written"
        assert png.stat().st_size > 0, f"{name} is empty"

    assert list(in_tmp_cwd.glob("*.html")) == [], "plotly output should be gone"


def test_empty_backtests_does_not_raise(frames, in_tmp_cwd):
    df_long, df_weather, df_inference = frames

    plot_forecasts(df_long, df_weather, df_inference, {}, pd.Timestamp("2026-04-01 00:00", tz="UTC"))

    assert (in_tmp_cwd / "Prediction.png").exists()
    assert (in_tmp_cwd / "Prediction_Backtest.png").exists()
    assert list(in_tmp_cwd.glob("*.html")) == []


def test_backtests_default_to_none(frames, in_tmp_cwd):
    df_long, df_weather, df_inference = frames

    plot_forecasts(df_long, df_weather, df_inference)

    assert (in_tmp_cwd / "Prediction_Backtest.png").exists()


def test_oracle_is_labelled_and_dashed(frames, in_tmp_cwd, keep_figure):
    df_long, df_weather, df_inference = frames
    live = _backtest("live", 96)
    oracle = _backtest("oracle", 192)

    plot_forecasts(df_long, df_weather, df_inference, {96: live, 192: oracle},
                   pd.Timestamp("2026-04-01 00:00", tz="UTC"))

    ax = keep_figure["fig"].axes[0]
    handles, labels = ax.get_legend_handles_labels()
    styles = {label: handle for label, handle in zip(labels, handles, strict=True)}

    assert live.label in styles
    assert oracle.label in styles
    assert live.label != oracle.label
    assert "oracle" in oracle.label and "oracle" not in live.label

    # Not colour alone: the oracle line is dashed, the honest one solid.
    assert styles[oracle.label].get_linestyle() == "--"
    assert styles[live.label].get_linestyle() == "-"

    # The caption is present because an oracle backtest is on the picture.
    assert "oracle" in ax.get_title().lower()


def test_no_oracle_note_when_all_backtests_are_honest(frames, in_tmp_cwd, keep_figure):
    df_long, df_weather, df_inference = frames
    backtests = {96: _backtest("live", 96), 192: _backtest("replay", 192)}

    plot_forecasts(df_long, df_weather, df_inference, backtests, pd.Timestamp("2026-04-01 00:00", tz="UTC"))

    title = keep_figure["fig"].axes[0].get_title()
    assert "actually occurred" not in title
    assert "with backtests" in title


def test_empty_backtest_frames_are_skipped(frames, in_tmp_cwd, keep_figure):
    df_long, df_weather, df_inference = frames
    empty = FakeBacktest(
        offset_hours=288,
        reference_time=REFERENCE - pd.Timedelta(hours=288),
        forecast=pd.DataFrame(),
        kind="oracle",
    )

    plot_forecasts(df_long, df_weather, df_inference, {288: empty},
                   pd.Timestamp("2026-04-01 00:00", tz="UTC"))

    _handles, labels = keep_figure["fig"].axes[0].get_legend_handles_labels()
    assert empty.label not in labels
    # An empty oracle is not on the picture, so it must not trigger the note either.
    assert "actually occurred" not in keep_figure["fig"].axes[0].get_title()


def test_plots_in_local_berlin_time(frames, in_tmp_cwd, keep_figure):
    """The x-axis is naive Europe/Berlin, i.e. shifted from the UTC input."""
    df_long, df_weather, df_inference = frames

    plot_forecasts(df_long, df_weather, df_inference, {96: _backtest("live", 96)},
                   pd.Timestamp("2026-04-01 00:00", tz="UTC"))

    ax = keep_figure["fig"].axes[0]
    right = pd.Timestamp(mdates.num2date(ax.get_xlim()[1])).tz_convert(None)
    expected = df_inference.index.max().tz_convert("Europe/Berlin").tz_localize(None)
    assert abs(right - expected) < pd.Timedelta(minutes=1)


@pytest.fixture
def every_figure(monkeypatch):
    """Keep every figure the renderer makes, in order, instead of closing it."""
    figures = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: figures.append(fig))
    yield figures
    real_close("all")


def test_archived_forecasts_without_the_band_get_it_interpolated():
    """Forecasts archived before q0.1/q0.9 existed: linear in the level, like `to_grid`."""
    index = pd.date_range(REFERENCE, periods=3, freq="h", tz="UTC")
    legacy = _quantile_frame(index)
    assert "wassertemp_q0.1" not in legacy.columns
    banded = _with_band(legacy)
    # The fixture is linear in q, so interpolation is exact.
    median = legacy["wassertemp_q0.5"]
    assert ((banded["wassertemp_q0.1"] - (median - 0.4 * 4.0)).abs() < 1e-9).all()
    assert ((banded["wassertemp_q0.9"] - (median + 0.4 * 4.0)).abs() < 1e-9).all()


def test_exact_band_columns_are_left_alone():
    index = pd.date_range(REFERENCE, periods=3, freq="h", tz="UTC")
    frame = _quantile_frame(index).assign(**{"wassertemp_q0.1": 1.0, "wassertemp_q0.9": 2.0})
    assert _with_band(frame)["wassertemp_q0.1"].eq(1.0).all()


def test_both_models_share_limits_and_axes_position(frames, in_tmp_cwd, every_figure):
    """Switching models on the page must change the curves and nothing else."""
    df_long, df_weather, df_inference = frames
    duet = duet_view(df_long, df_weather, df_inference, {96: _backtest("live", 96)},
                     issued_at=REFERENCE)
    # A candidate with a much wider band, one fewer backtest and a missing-window note,
    # so its titles have a different number of lines.
    candidate = ModelView(
        name="TimesFM", forecast_png="a.png", backtest_png="b.png",
        measured=duet.measured, air=duet.air + 5.0,
        forecast=_with_band(duet.forecast.assign(**{
            "wassertemp_q0.1": duet.forecast["wassertemp_q0.5"] - 6.0,
            "wassertemp_q0.9": duet.forecast["wassertemp_q0.5"] + 6.0})),
        issued_at=REFERENCE, missing=(288,))

    written = plot_models([duet, candidate])
    assert written == ["Prediction.png", "a.png", "Prediction_Backtest.png", "b.png"]
    for first, second in (every_figure[0:2], every_figure[2:4]):
        for ax_a, ax_b in zip(first.axes, second.axes, strict=True):
            assert ax_a.get_xlim() == ax_b.get_xlim()
            assert ax_a.get_ylim() == ax_b.get_ylim()
            assert ax_a.get_position().bounds == ax_b.get_position().bounds
    # The wider band decides, so it is not cut off on either image.
    water_low, water_high = every_figure[0].axes[0].get_ylim()
    assert water_low <= candidate.forecast["wassertemp_q0.1"].min()
    assert water_high >= candidate.forecast["wassertemp_q0.9"].max()
    assert "incomplete" in every_figure[3].axes[0].get_title()


def test_duet_shows_the_same_single_band_as_timesfm(frames, in_tmp_cwd, keep_figure):
    df_long, df_weather, df_inference = frames
    plot_forecasts(df_long, df_weather, df_inference)
    _handles, labels = keep_figure["fig"].axes[0].get_legend_handles_labels()
    assert [label for label in labels if label.startswith("q")] == ["q0.1-q0.9 (80 %)"]
