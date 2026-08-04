"""Run the forecast, and resolve each backtest to the most honest source available.

The main forecast is straightforward: feed the model everything up to the last real
water-temperature reading and predict forward.

The backtests are the interesting part. A backtest at −96 h asks *what would we have
predicted four days ago?*, and there are three ways to answer that, in descending order
of both honesty and cheapness:

1. **Look it up.** If we actually ran four days ago, the forecast is in the archive. It
   used the DWD forecast that was genuinely available then, so it carries the weather
   error too. Free, and the only fully honest answer.
2. **Replay it.** If we have the DWD forecast *as it was issued* around that time, we
   can rebuild the covariates the model would have seen and recompute. Costs one model
   run and is honest, but only works where a weather snapshot exists.
3. **Ask the oracle.** Otherwise, fall back to the weather that actually occurred. This
   hands the model a perfect forecast, so the result flatters us and must be labelled as
   such wherever it is shown.

Historical DWD *forecasts* are a paid product we do not have, which is why option 3
exists at all, and why option 2 only becomes useful once we have been archiving weather
snapshots for a while.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from eisbach import archive
from eisbach.data import assemble_long_frame
from eisbach.model import forecast, load_model, long_to_wide
from eisbach.model.checkpoint import CHECKPOINT_SHA256

logger = logging.getLogger(__name__)

BACKTEST_OFFSETS_HOURS = (96, 192, 288)

#: How far a stored forecast's reference time may sit from the one we asked for. Kept
#: tight: a forecast anchored elsewhere is a different forecast, because it knew a
#: different amount about the world.
ARCHIVE_TOLERANCE_HOURS = 2

#: How stale a weather snapshot may be for a replay. One-sided by construction — see
#: :func:`eisbach.archive.load_weather_snapshot`.
SNAPSHOT_MAX_AGE_HOURS = 12

MODEL_ID = CHECKPOINT_SHA256[:12]

MAIN_CSV_NAME = "Prediction.csv"


@dataclass(frozen=True)
class Backtest:
    """One backtest, together with how trustworthy it is."""

    offset_hours: int
    reference_time: pd.Timestamp
    forecast: pd.DataFrame
    kind: str
    covariate_source: str

    @property
    def is_honest(self) -> bool:
        """False for oracle backtests, which saw weather nobody could have known."""
        return self.kind != archive.KIND_ORACLE

    @property
    def label(self) -> str:
        suffix = "" if self.is_honest else ", perfect weather"
        return f"Backtest -{self.offset_hours}h ({self.kind}{suffix})"


def _predict(model, config, df_long: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Run the model on everything up to and including ``cutoff``."""
    truncated = df_long[df_long["date"] <= cutoff]
    return forecast(model, config, long_to_wide(truncated))


def _replay_weather(df_wetter: pd.DataFrame, snapshot: pd.DataFrame,
                    reference_time: pd.Timestamp) -> pd.DataFrame:
    """Splice observed weather with the forecast that was current at ``reference_time``.

    This reproduces what a live run would have seen: measurements up to the reference
    time, and beyond it a forecast — not hindsight.
    """
    observed = df_wetter.loc[df_wetter.index <= reference_time, ["lufttemperatur_c", "pressure"]]

    predicted = snapshot.copy()
    predicted["timestamp"] = pd.to_datetime(predicted["timestamp"], utc=True)
    predicted = (
        predicted.set_index("timestamp")
        .rename(columns={"temperature": "lufttemperatur_c", "pressure_msl": "pressure"})
        .loc[:, ["lufttemperatur_c", "pressure"]]
        .sort_index()
    )
    predicted = predicted[predicted.index > reference_time]

    spliced = pd.concat([observed, predicted])
    return spliced[~spliced.index.duplicated(keep="first")].sort_index()


def _resolve_backtest(
    offset_hours: int,
    reference_time: pd.Timestamp,
    *,
    model,
    config,
    df_long: pd.DataFrame,
    df_wetter: pd.DataFrame,
    df_wt: pd.DataFrame,
    archive_root: Path,
) -> Backtest:
    """Find the most honest available answer for one backtest offset."""

    # 1. A forecast we really made. Free and fully honest.
    hit = archive.load_forecast(
        reference_time,
        tolerance_hours=ARCHIVE_TOLERANCE_HOURS,
        kinds=[archive.KIND_LIVE, archive.KIND_REPLAY],
        root=archive_root,
    )
    if hit is not None:
        stored, meta = hit
        logger.info("Backtest -%dh: reusing archived %s forecast", offset_hours, meta["kind"])
        return Backtest(
            offset_hours=offset_hours,
            reference_time=meta["reference_time"],
            forecast=stored,
            kind=meta["kind"],
            covariate_source=meta["covariate_source"],
        )

    # 2. Rebuild it from the weather forecast that was current at the time.
    snapshot = archive.load_weather_snapshot(
        reference_time, max_age_hours=SNAPSHOT_MAX_AGE_HOURS, root=archive_root,
    )
    if snapshot is not None:
        rows, _issued_at = snapshot
        replay_long = assemble_long_frame(
            df_wt[df_wt["timestamp"] <= reference_time],
            _replay_weather(df_wetter, rows, reference_time),
        )
        result = _predict(model, config, replay_long, reference_time)
        kind, covariates = archive.KIND_REPLAY, archive.COVARIATE_DWD_ARCHIVED
        logger.info("Backtest -%dh: replayed from an archived weather forecast", offset_hours)
    else:
        # 3. Oracle. Optimistic by construction, so it is labelled everywhere it appears.
        result = _predict(model, config, df_long, reference_time)
        kind, covariates = archive.KIND_ORACLE, archive.COVARIATE_DWD_OBSERVED
        logger.info(
            "Backtest -%dh: no archived forecast or weather snapshot, falling back to "
            "observed weather. This flatters the model.", offset_hours,
        )

    archive.write_forecast(
        result,
        reference_time=reference_time,
        kind=kind,
        covariate_source=covariates,
        model_id=MODEL_ID,
        root=archive_root,
    )
    return Backtest(
        offset_hours=offset_hours,
        reference_time=reference_time,
        forecast=result,
        kind=kind,
        covariate_source=covariates,
    )


def run_inference(
    df_long: pd.DataFrame,
    df_wetter: pd.DataFrame,
    df_wt: pd.DataFrame,
    archive_root: Path = archive.DEFAULT_ROOT,
) -> tuple[pd.DataFrame, dict[int, Backtest]]:
    """Produce the main forecast and its backtests.

    Returns the forecast indexed by target time, and one :class:`Backtest` per offset.
    """
    df_long = df_long.copy()
    df_long["date"] = pd.to_datetime(df_long["date"])

    observed = df_long[df_long["cols"] == "wassertemp"].dropna(subset=["data"])
    if observed.empty:
        raise ValueError("no water temperature observations, cannot forecast")
    last_timestamp = observed["date"].max()

    logger.info("Forecasting from %s", last_timestamp)
    model, config = load_model()

    # The covariates are already shifted forward by 96 h, so truncating at the last
    # observation still leaves the model the full four days of weather it needs.
    df_inference = _predict(model, config, df_long, last_timestamp)

    archive.write_forecast(
        df_inference,
        reference_time=last_timestamp,
        kind=archive.KIND_LIVE,
        covariate_source=archive.COVARIATE_DWD_FORECAST,
        model_id=MODEL_ID,
        root=archive_root,
    )
    # Snapshot the weather forecast as issued, so this moment can be replayed later even
    # if the model output is ever lost.
    archive.write_weather_snapshot(df_wetter.reset_index(), root=archive_root)
    archive.write_observations(
        observed.set_index("date")[["data"]].rename(columns={"data": "wassertemp"}),
        root=archive_root,
    )

    backtests = {
        offset: _resolve_backtest(
            offset,
            last_timestamp - pd.Timedelta(hours=offset),
            model=model,
            config=config,
            df_long=df_long,
            df_wetter=df_wetter,
            df_wt=df_wt,
            archive_root=archive_root,
        )
        for offset in BACKTEST_OFFSETS_HOURS
    }

    _write_readable_csv(df_inference)
    return df_inference, backtests


def _write_readable_csv(df_inference: pd.DataFrame, path: str = MAIN_CSV_NAME) -> None:
    """Write the human-facing CSV in local time, without timezone offsets in the index."""
    local = df_inference.copy()
    if local.index.tzinfo is None:
        local.index = local.index.tz_localize("UTC")
    local.index = local.index.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M")
    local.to_csv(path)
    logger.info("Wrote %s", path)
