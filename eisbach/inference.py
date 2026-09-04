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
#: :func:`eisbach.archive.load_weather_snapshot`. Defined in ``archive`` because the
#: write path derives its row-trimming margin from it; re-exported here so the read and
#: write sides cannot drift apart.
SNAPSHOT_MAX_AGE_HOURS = archive.SNAPSHOT_MAX_AGE_HOURS

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


#: Bright Sky's own column names, as they appear in older archived snapshots.
_RAW_WEATHER_NAMES = {"temperature": "lufttemperatur_c", "pressure_msl": "pressure"}


def _declares_current_schema(snapshot: pd.DataFrame) -> bool:
    """True when every row of a snapshot states it is the current weather schema.

    Rows written before ``schema_version`` existed carry nothing, so absence is not a
    denial — it means "sniff the columns". A partition can hold both, but a snapshot
    handed here is always one anchor's worth of rows, so it is homogeneous in practice;
    requiring *every* row to agree keeps that assumption honest rather than assumed.
    """
    if "schema_version" not in snapshot.columns:
        return False
    declared = snapshot["schema_version"].dropna().astype(str)
    return len(declared) == len(snapshot) and set(declared) == {archive.WEATHER_SCHEMA_VERSION}


def _canonical_weather(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Reduce an archived weather snapshot to a timestamp index and the two covariates.

    Snapshots exist in two shapes: the raw Bright Sky payload, which is what the
    original five-daily archive stored, and the processed frame the pipeline works with.
    Accept both, and take the processed names where a row carries both — renaming blindly
    would produce two columns of the same name and fail on lookup.

    A snapshot that declares the current schema is taken at its word, so a missing
    canonical column raises instead of quietly falling back to a Bright Sky name that a
    normalised frame never had. Older snapshots say nothing about their shape and are
    sniffed as before; those partitions are irreplaceable and are never rewritten.
    """
    frame = snapshot.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()

    self_describing = _declares_current_schema(snapshot)

    columns = {}
    for raw, canonical in _RAW_WEATHER_NAMES.items():
        if canonical in frame.columns and frame[canonical].notna().any():
            columns[canonical] = frame[canonical]
        elif self_describing:
            raise ValueError(
                f"weather snapshot declares schema {archive.WEATHER_SCHEMA_VERSION!r} "
                f"but has no usable {canonical!r} column"
            )
        elif raw in frame.columns:
            columns[canonical] = frame[raw]
        else:
            raise ValueError(f"weather snapshot has neither {canonical!r} nor {raw!r}")

    return pd.DataFrame(columns).astype(float)


def _replay_weather(df_weather: pd.DataFrame, snapshot: pd.DataFrame,
                    reference_time: pd.Timestamp) -> pd.DataFrame:
    """Splice observed weather with the forecast that was current at ``reference_time``.

    This reproduces what a live run would have seen: measurements up to the reference
    time, and beyond it a forecast — not hindsight.
    """
    observed = df_weather.loc[df_weather.index <= reference_time, ["lufttemperatur_c", "pressure"]]

    predicted = _canonical_weather(snapshot)
    predicted = predicted[predicted.index > reference_time]

    spliced = pd.concat([observed, predicted])
    return spliced[~spliced.index.duplicated(keep="first")].sort_index()


#: Measured weather archived alongside the water temperature, under the same names the
#: live frame uses — a German domain name that must not be translated, and a column an
#: analyst will join against the weather store.
OBSERVED_WEATHER_COLUMNS = ["lufttemperatur_c", "pressure"]


def _observed_frame(observed: pd.DataFrame, df_weather: pd.DataFrame,
                    reference_time: pd.Timestamp) -> pd.DataFrame:
    """Assemble what was *measured* up to the run's anchor, for later verification.

    Weather snapshots keep only what the DWD forecast said, so the observed air
    temperature and pressure are recorded here instead. Bright Sky serves DWD's
    observation archive for free and indefinitely, so this is a convenience rather than a
    last copy — but without it a forecast error cannot be split into "the model was wrong
    about the river" and "DWD was wrong about the air", which is the question the
    live/replay/oracle distinction exists to answer.

    The observed part of the weather frame is everything at or before the anchor; past it
    the frame holds DWD's forecast, which is not an observation of anything. Joined onto
    the water observations rather than unioned with them, so the store keeps exactly one
    row per measured hour, as it always has.
    """
    water = observed.set_index("date")[["data"]].rename(columns={"data": "wassertemp"})
    water.index.name = "timestamp"

    measured = df_weather.loc[df_weather.index <= reference_time, OBSERVED_WEATHER_COLUMNS]
    return water.join(measured, how="left")


def _resolve_backtest(
    offset_hours: int,
    reference_time: pd.Timestamp,
    *,
    model,
    config,
    df_long: pd.DataFrame,
    df_weather: pd.DataFrame,
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
            _replay_weather(df_weather, rows, reference_time),
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
    df_weather: pd.DataFrame,
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
    # if the model output is ever lost. Keyed to the run's anchor rather than to the
    # fetch time — see archive.write_weather_snapshot.
    snapshot = df_weather.copy()
    # Name it here rather than trusting the caller's index name: the archive keys and
    # trims on `timestamp`, and a frame arriving with an unnamed index would otherwise
    # reach it as a column called `index`.
    snapshot.index.name = "timestamp"
    archive.write_weather_snapshot(
        snapshot.reset_index(), reference_time=last_timestamp, root=archive_root,
    )
    archive.write_observations(
        _observed_frame(observed, df_weather, last_timestamp), root=archive_root,
    )

    backtests = {
        offset: _resolve_backtest(
            offset,
            last_timestamp - pd.Timedelta(hours=offset),
            model=model,
            config=config,
            df_long=df_long,
            df_weather=df_weather,
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
