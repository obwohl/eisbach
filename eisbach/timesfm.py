"""TimesFM 3.0 beside DUET: the candidate forecast, run for real and archived.

This is the second forecast on the published page, not a replacement for the first.
`main.py` runs it after DUET and swallows its failures, so a candidate that breaks
costs the page one of two graphs and never the thrice-daily run itself.

What it forecasts and from what is settled in `experiments/timesfm/STATUS.md`:
the Eisbach from a year of its own history, two upstream water temperatures the model
may look at only in the past, and seven weather series it is told in advance. Nothing
here is constructed — no discharge-weighted mixtures, no 24-hour sums. The scope was
always *which raw series to hand over, and as what*, and in both places a construction
was tested against its own ingredients the raw series won.

Two things it does that the research could not:

*The weather is a real forecast.* Every covariate number in the research sweeps is the
weather that **actually occurred**, because archived DWD forecasts begin in May 2026 and
only for Munich. exp16 measured what that flatters — about 4.8 % of the oracle advantage
in MAE — but only for Munich air, with the other six simulated from it. Here the seven
known-future series come from MOSMIX, as a forecast, the way they will always come.

*Its own forecasts are archived*, under `data/archive/timesfm/`, separately from the
production store so a candidate can never outrank or displace a DUET row. That archive is
the honest comparison the oracle sweeps cannot give: in a few months there will be live
TimesFM forecasts and live DUET forecasts over the same windows, scored against what
happened, with neither having seen the future.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from eisbach.archive import DEFAULT_ROOT, KIND_LIVE, read_forecasts, write_forecast
from eisbach.covariates import STORE, fetch_future, model_values

logger = logging.getLogger(__name__)

#: The target, and the two upstream Wassertemperaturen the model sees only in the past.
TARGET = "eisbach"
PAST_ONLY = ("isar_toelz", "loisach_beuerberg")

#: Weather the model is told in advance. Order is fixed so a stored forecast can be
#: read back knowing which column was which; unlike DUET's `CHANNELS` it is not a
#: trained order, since TimesFM is zero-shot and takes covariates by position only
#: within a single call.
KNOWN_FUTURE = ("airtemp", "t_catchment", "rain_toelz", "rain_lenggries",
                "rain_kochel", "rain_garmisch", "solar_hohenpeissenberg")

#: A year of context and a four-day horizon. Context is the dominant lever — measured
#: three times, on two machines and two backends, at −12 % to −16 % MAE against shorter
#: windows, which is larger than every covariate put together.
CONTEXT_HOURS = 8760
HORIZON_HOURS = 96

#: Pinned by digest, like the DUET checkpoint: a silently updated checkpoint would move
#: the published forecast with nothing in the archive to explain it.
CHECKPOINT = "google/timesfm-3.0-pytorch"
MODEL_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"

#: TimesFM 3.0 emits exactly these and nothing else. There is no q0.25 or q0.05 to be
#: had, which is why the published bands are 20–80 and 10–90 rather than DUET's 25–75
#: and 1–99, and why the two cannot be compared by eye.
DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: How much of the context may be missing before this refuses to forecast. The research
#: selected its origins with exactly this rule, so production runs on windows of the
#: quality the numbers were measured on. Gaps below it are left to the model: the PyTorch
#: backend trims leading NaN and interpolates the rest before the network sees anything.
MIN_CONTEXT_COVERAGE = 0.95

ARCHIVE_STORE = "timesfm"
CSV_NAME = "Prediction_timesfm.csv"

#: The same three the DUET image uses, so the two backtest pictures ask the same question.
BACKTEST_OFFSETS_HOURS = (96, 192, 288)


def load_forecaster(checkpoint: str = CHECKPOINT, revision: str = MODEL_REVISION):
    """The forecaster, on MLX where it was asked for and PyTorch everywhere else.

    ``TIMESFM_BACKEND=mlx`` selects the Apple-silicon backend, which every research
    number on this model was produced on. It mirrors the PyTorch interface and matches it
    to about 2e-6 on the quantiles. CI has no Apple silicon, so production is PyTorch.
    """
    import os

    from huggingface_hub import snapshot_download

    path = snapshot_download(checkpoint, revision=revision)
    if os.environ.get("TIMESFM_BACKEND", "torch").lower() == "mlx":
        from timesfm3.mlx import TimesFM3Forecaster as MlxForecaster

        logger.info("TimesFM on the MLX backend")
        return MlxForecaster.from_pretrained(path)

    from timesfm3 import TimesFM3Forecaster

    logger.info("TimesFM on the PyTorch backend")
    return TimesFM3Forecaster.from_pretrained(path)


def anchor_time(*, root: Path = STORE, now=None) -> pd.Timestamp:
    """The most recent hour the Eisbach was actually measured at.

    Forecasting from a later hour would hand the model a trailing gap in the one series
    it is being asked about, which is the one place interpolation cannot help.
    """
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC").floor("h")
    # A lookup, not a decision pass: `refresh` has already recorded this window.
    recent = model_values(TARGET, root=root, start=now - pd.Timedelta(days=7), end=now,
                          record=False)
    last = recent.last_valid_index()
    if last is None:
        raise RuntimeError("No usable Eisbach measurement in the last seven days")
    return pd.Timestamp(last)


def context_frame(anchor, *, root: Path = STORE) -> pd.DataFrame:
    """A year of gated history for every series, ending at `anchor`.

    `model_values` applies the quality gate, so a 154.4 °C reading is already a hole here
    rather than a number. The holes are then counted, not filled: below
    `MIN_CONTEXT_COVERAGE` this refuses, and above it the backend interpolates.
    """
    anchor = pd.Timestamp(anchor).tz_convert("UTC")
    start = anchor - pd.Timedelta(hours=CONTEXT_HOURS - 1)
    names = [TARGET, *PAST_ONLY, *KNOWN_FUTURE]
    # `record=False`: the same verdicts, none of them written down. The quality log is
    # for the live window `refresh` reads, and a year of flags re-derived three times a
    # day would bury the findings anyone actually wants to look at.
    frame = pd.concat([model_values(n, root=root, start=start, end=anchor, record=False)
                       for n in names], axis=1).reindex(pd.date_range(start, anchor, freq="h"))
    coverage = frame.notna().mean()
    thin = coverage[coverage < MIN_CONTEXT_COVERAGE]
    if not thin.empty:
        raise RuntimeError(
            "Context too thin to forecast on: "
            + ", ".join(f"{name} {share:.1%} present" for name, share in thin.items())
            + f" (need {MIN_CONTEXT_COVERAGE:.0%})")
    if pd.isna(frame[TARGET].iloc[-1]):
        raise RuntimeError(f"No {TARGET} measurement at the anchor {anchor}")
    return frame[names]


def future_frame(anchor, *, now=None) -> pd.DataFrame:
    """The DWD forecast for the known-future covariates over the horizon.

    The horizon starts at the anchor, which is the last hour the Eisbach was measured, so
    its first hours have usually already happened by the time this runs. Bright Sky
    serves those as measurements and the rest as MOSMIX, and the two kinds of hole mean
    opposite things:

    *After now* the answer is a forecast, and MOSMIX does not have gaps. A hole there is
    a station that is not being forecast, and interpolating across it would put an
    invented number where the model expects to be told the truth. Refuse.

    *At or before now* a hole is a gauge that has not reported yet — the same thing that
    happens ~34 times per series per 180 days inside the context, where it is tolerated
    and left to the backend to interpolate. Refusing here as well would be inconsistent
    (it is one array: `past_future_covariates` spans context and horizon together) and
    expensive: measured over 180 days of the archive, a late rain gauge in the last few
    hours would have blocked **2 % of runs**, about one every three weeks. The complete
    forecast to its right guarantees the interpolation always has something to reach.
    """
    anchor = pd.Timestamp(anchor).tz_convert("UTC")
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC").floor("h")
    frame = fetch_future(list(KNOWN_FUTURE),
                         anchor + pd.Timedelta(hours=1),
                         anchor + pd.Timedelta(hours=HORIZON_HOURS))
    predicted = frame.loc[frame.index > now, list(KNOWN_FUTURE)]
    missing = predicted.isna().sum()
    if missing.any():
        raise RuntimeError("Incomplete weather forecast: "
                           + ", ".join(f"{name} missing {int(n)} h"
                                       for name, n in missing[missing > 0].items()))
    late = frame.loc[frame.index <= now, list(KNOWN_FUTURE)].isna().sum()
    if late.any():
        logger.warning("Not yet reported at the start of the horizon, left to the model: %s",
                       ", ".join(f"{name} {int(n)} h" for name, n in late[late > 0].items()))
    return frame


def _fill_for_mlx(target: np.ndarray, past_only: np.ndarray, past_future: np.ndarray):
    """Do what the PyTorch backend does internally, for the backend that does not.

    PyTorch trims leading missing target values and linearly interpolates the rest before
    the network sees anything. MLX 3.0.2 passes NaN straight through and the forecast
    comes back all NaN, which `predict` would then refuse — so on MLX this is not an edge
    case but every run: `context_frame` deliberately tolerates up to 5 % missing, and
    measured on a real year of the archive **all ten series carry gaps**, from one hour to
    fifty-three. Without this the advertised MLX path cannot produce a forecast at all.

    Only on MLX. Doing it on PyTorch as well would move the numbers — measured at
    1.8e-5 °C hourly in `experiments/timesfm/bench.py`, whose `prepare_inputs` this
    mirrors — for no gain, since the backend already does it.

    Never touches the arrays it is given.
    """
    target = np.array(target, dtype=np.float32, copy=True)
    past_only = np.array(past_only, dtype=np.float32, copy=True)
    past_future = np.array(past_future, dtype=np.float32, copy=True)

    valid = ~np.isnan(target)
    if valid.any():
        first = int(np.argmax(valid))
        # All three start at the same hour, so the same leading columns come off each.
        target, past_only, past_future = target[first:], past_only[:, first:], past_future[:, first:]
    else:
        target[...] = 0.0

    for array in (target, past_only, past_future):
        for row in np.atleast_2d(array):
            missing = np.isnan(row)
            if not missing.any():
                continue
            present = ~missing
            row[missing] = (np.interp(np.flatnonzero(missing), np.flatnonzero(present),
                                      row[present]) if present.any() else 0.0)
    return target, past_only, past_future


def predict(context: pd.DataFrame, future: pd.DataFrame, forecaster=None) -> pd.DataFrame:
    """Run the model and return the deciles, indexed by target time.

    The three arrays are exactly what exp19 handed it: the target as one series, the
    past-only covariates as `(variates, context)`, and the known-future ones as
    `(variates, context + horizon)` — the future block is what makes them known-future.

    Gaps reach the model as NaN and the backend fills them, except on MLX, where
    `_fill_for_mlx` does it first.
    """
    import os

    forecaster = forecaster if forecaster is not None else load_forecaster()
    target = context[TARGET].to_numpy(dtype=np.float32)
    past_only = context[list(PAST_ONLY)].to_numpy(dtype=np.float32).T
    past_future = np.concatenate(
        [context[list(KNOWN_FUTURE)].to_numpy(dtype=np.float32),
         future[list(KNOWN_FUTURE)].to_numpy(dtype=np.float32)]).T
    if os.environ.get("TIMESFM_BACKEND", "torch").lower() == "mlx":
        target, past_only, past_future = _fill_for_mlx(target, past_only, past_future)
    result = forecaster.predict(target, horizon=HORIZON_HOURS,
                                past_only_covariates=past_only,
                                past_future_covariates=past_future,
                                return_quantiles=True)
    quantiles = np.asarray(result.quantiles, dtype=float)
    if quantiles.ndim == 3:
        quantiles = quantiles[0]
    if quantiles.shape != (HORIZON_HOURS, len(DECILES)):
        raise RuntimeError(f"Expected {(HORIZON_HOURS, len(DECILES))} quantiles, "
                           f"got {quantiles.shape}")
    if not np.isfinite(quantiles).all():
        raise RuntimeError("TimesFM returned non-finite quantiles")
    if (np.diff(quantiles, axis=1) < 0).any():
        # Crossing quantiles are not a cosmetic defect: every band on the plot and every
        # pinball term in the CRPS assumes the deciles are ordered.
        raise RuntimeError("TimesFM returned crossing quantiles")
    return pd.DataFrame(quantiles, index=future.index,
                        columns=[f"wassertemp_q{q}" for q in DECILES]).rename_axis("target_time")


def backtests(anchor, *, root: Path = DEFAULT_ROOT,
              offsets=BACKTEST_OFFSETS_HOURS, tolerance_hours: float = 4) -> dict:
    """Earlier TimesFM forecasts from this archive, by offset from `anchor`.

    Not a re-run: these are forecasts the candidate really published, read back. A
    candidate has no oracle track and never will — the whole point of running it live is
    that its weather was a forecast at the time — so an offset with nothing archived is
    simply absent, and on the first runs that is all of them.
    """
    stored = read_forecasts(root=root, store=ARCHIVE_STORE)
    if stored.empty:
        return {}
    anchor = pd.Timestamp(anchor).tz_convert("UTC")
    found = {}
    for offset in offsets:
        wanted = anchor - pd.Timedelta(hours=offset)
        near = stored[(stored.reference_time - wanted).abs()
                      <= pd.Timedelta(hours=tolerance_hours)]
        if near.empty:
            continue
        chosen = near[near.reference_time.eq(
            near.reference_time.iloc[(near.reference_time - wanted).abs().argmin()])]
        quantile_columns = [c for c in chosen.columns if c.startswith("wassertemp_q")]
        found[offset] = chosen.set_index("target_time")[quantile_columns].sort_index()
    return found


def write_csv(quantiles: pd.DataFrame, path: str = CSV_NAME) -> None:
    """The human-facing CSV, in local time, alongside DUET's."""
    local = quantiles.copy()
    local.index = local.index.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M")
    local.to_csv(path)
    logger.info("Wrote %s", path)


def run(*, root: Path = DEFAULT_ROOT, covariates: Path = STORE, now=None,
        forecaster=None, issued_at=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Forecast, archive the forecast and the weather it was built on, and return all three.

    `root` is the archive this writes to; `covariates` the station-pinned store it reads
    the history from. Returns `(quantiles, context, future)` — the last two so the caller
    can plot the history the forecast continues from, and the weather it was told, without
    reading the archive twice.
    """
    anchor = anchor_time(root=covariates, now=now)
    logger.info("TimesFM anchor %s, %d h of context, %d h horizon",
                anchor, CONTEXT_HOURS, HORIZON_HOURS)
    context = context_frame(anchor, root=covariates)
    future = future_frame(anchor, now=now)
    quantiles = predict(context, future, forecaster=forecaster)
    issued_at = pd.Timestamp(issued_at or pd.Timestamp.now(tz="UTC"))
    # The weather first: a forecast whose inputs were not kept can never be re-examined,
    # and this is the only copy — a DWD forecast is gone the moment it is superseded.
    write_covariate_forecast(future, reference_time=anchor, issued_at=issued_at, root=root)
    write_forecast(quantiles, reference_time=anchor, kind=KIND_LIVE,
                   covariate_source="brightsky_mosmix", issued_at=issued_at,
                   model_id=f"{CHECKPOINT}@{MODEL_REVISION[:12]}",
                   root=root, store=ARCHIVE_STORE)
    return quantiles, context, future


def write_covariate_forecast(future: pd.DataFrame, *, reference_time, issued_at,
                             root: Path = DEFAULT_ROOT) -> Path:
    """Archive the known-future covariates as they were forecast, for later replay.

    `data/archive/weather/` already keeps Munich's, in the shape DUET needs. This keeps
    all seven in the shape TimesFM consumed them, which is what a replay of *this*
    forecast requires. Without it the candidate could only ever be re-scored against
    weather that had already happened, which is the flattery the whole research programme
    had to caveat.
    """
    from eisbach.archive import _partition_path, _read_partition, _write_partition

    incoming = future.reset_index()
    incoming.insert(0, "reference_time", pd.Timestamp(reference_time))
    incoming["issued_at"] = pd.Timestamp(issued_at)
    path = _partition_path(root, "covariate_forecasts", pd.Timestamp(reference_time))
    existing = _read_partition(path)
    combined = (pd.concat([existing, incoming], ignore_index=True)
                if not existing.empty else incoming)
    combined = (combined.sort_values("issued_at", na_position="first")
                .drop_duplicates(["reference_time", "target_time"], keep="last")
                .sort_values(["reference_time", "target_time"]))
    _write_partition(path, combined)
    logger.info("Archived %d forecast covariate hours for %s", len(incoming), reference_time)
    return path
