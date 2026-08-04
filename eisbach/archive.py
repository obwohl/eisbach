"""Provenance-tracked storage for forecasts, weather snapshots and observations.

The point of this module is to make one question answerable for any past moment:
*what did we actually predict then, and how honest was that prediction?*

Three kinds of forecast end up in the archive, and they are not equally trustworthy:

``live``
    Produced by a real run, using the DWD weather forecast that was genuinely
    available at that moment. It carries the weather forecast's own error, which is
    what makes it honest. It cannot be regenerated — if it was not stored, it is gone.

``replay``
    Reconstructed after the fact from an archived DWD forecast snapshot, so the model
    sees the same covariates a live run would have seen at that time. Reproducible,
    but only for moments where a weather snapshot exists.

``oracle``
    Reconstructed using the weather that *actually occurred*. This pretends the
    forecast was perfect, so it flatters the model and must never be read as evidence
    of real-world skill. Always reproducible. Historical DWD forecasts exist as a paid
    product; without them this is the fallback.

Because ``live`` rows are irreplaceable and the other two are not, writes obey a
precedence rule: a regenerable row may never overwrite a genuine one.

Storage is month-partitioned CSV under ``data/archive/``. CSV rather than a database
because the archive is append-mostly, is committed to git, and should stay readable by
hand; month partitions because rewriting one 900 KB file per month deltas well in git,
whereas rewriting one ever-growing file per run does not — that pattern put 171 MB of
revisions of a single 3 MB file into this repository's history.
Partitioning is by the month of ``reference_time``, so a single forecast never straddles
two files even when its horizon crosses a month boundary.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("data/archive")

KIND_LIVE = "live"
KIND_REPLAY = "replay"
KIND_ORACLE = "oracle"

#: Higher wins. A write may upgrade a row's kind but never downgrade it.
KIND_PRECEDENCE = {KIND_ORACLE: 0, KIND_REPLAY: 1, KIND_LIVE: 2}

#: Where the future covariates for a forecast came from.
COVARIATE_DWD_FORECAST = "dwd_forecast"  # available at run time — honest
COVARIATE_DWD_ARCHIVED = "dwd_forecast_archived"  # snapshot replayed — honest
COVARIATE_DWD_OBSERVED = "dwd_observed"  # what actually happened — oracle

#: Written ahead of the quantile columns, in this order.
METADATA_COLUMNS = [
    "reference_time",
    "target_time",
    "kind",
    "covariate_source",
    "issued_at",
    "model_id",
    "code_version",
]

_UNIQUE_KEY = ["reference_time", "target_time"]

logger = logging.getLogger(__name__)


def code_version() -> str:
    """Best-effort identifier of the code that produced a forecast.

    Prefers ``GITHUB_SHA`` so CI records the exact commit; falls back to asking git
    directly. Returns an empty string rather than raising — provenance is valuable but
    never worth failing a run over.
    """
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()[:12]
    except (subprocess.SubprocessError, OSError):
        return ""


def _partition_path(root: Path, store: str, when: pd.Timestamp) -> Path:
    return Path(root) / store / f"{pd.Timestamp(when).strftime('%Y-%m')}.csv"


def _as_utc(value) -> pd.Timestamp:
    """Normalise any timestamp to tz-aware UTC, so comparisons never mix naive/aware."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _as_utc_series(values) -> pd.Series:
    """Parse a column of timestamps to UTC, tolerating a column that is not there.

    Returns an all-NaT series for a missing column, so callers can treat "written before
    this field existed" the same as "this row has no value".
    """
    if values is None:
        return pd.Series(pd.NaT, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")


def _read_partition(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    # `archive_timestamp` is deliberately left as text: it is an identity key for a
    # weather snapshot, not a value to do arithmetic on.
    for col in ("reference_time", "target_time", "issued_at", "timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def _write_partition(path: Path, df: pd.DataFrame) -> None:
    """Write atomically, so an interrupted run cannot leave a half-written partition."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _resolve_precedence(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate (reference_time, target_time) rows, keeping the best kind.

    Ties on kind are broken by ``issued_at``, newest wins — re-running a live forecast
    for the same reference time should refresh it, not be ignored.
    """
    if df.empty:
        return df
    ranked = df.assign(_rank=df["kind"].map(KIND_PRECEDENCE).fillna(-1))
    ranked = ranked.sort_values(["_rank", "issued_at"], na_position="first")
    deduped = ranked.drop_duplicates(subset=_UNIQUE_KEY, keep="last")
    return deduped.drop(columns="_rank").sort_values(_UNIQUE_KEY).reset_index(drop=True)


def write_forecast(
    df_forecast: pd.DataFrame,
    *,
    reference_time,
    kind: str,
    covariate_source: str,
    issued_at=None,
    model_id: str = "",
    version: str | None = None,
    root: Path = DEFAULT_ROOT,
) -> Path:
    """Store one forecast with its provenance and return the partition written.

    ``df_forecast`` is indexed by target time, with one column per quantile
    (``wassertemp_q0.5`` and friends).

    A row that is already present at a higher precedence is left alone, so replaying an
    oracle backtest over a moment we hold a live forecast for is a no-op rather than a
    silent loss of real data.
    """
    if kind not in KIND_PRECEDENCE:
        raise ValueError(f"unknown kind {kind!r}, expected one of {sorted(KIND_PRECEDENCE)}")
    if df_forecast.empty:
        raise ValueError("refusing to archive an empty forecast")

    reference_time = _as_utc(reference_time)
    issued_at = _as_utc(issued_at if issued_at is not None else pd.Timestamp.now(tz="UTC"))

    incoming = df_forecast.copy()
    incoming.index = pd.to_datetime(incoming.index, utc=True)
    incoming.index.name = "target_time"
    incoming = incoming.reset_index()

    incoming.insert(0, "reference_time", reference_time)
    incoming["kind"] = kind
    incoming["covariate_source"] = covariate_source
    incoming["issued_at"] = issued_at
    incoming["model_id"] = model_id
    incoming["code_version"] = version if version is not None else code_version()

    quantile_columns = [c for c in incoming.columns if c not in METADATA_COLUMNS]
    incoming = incoming[METADATA_COLUMNS + quantile_columns]

    path = _partition_path(root, "forecasts", reference_time)
    existing = _read_partition(path)
    combined = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
    _write_partition(path, _resolve_precedence(combined))

    logger.info(
        "Archived %d rows (kind=%s, covariates=%s, ref=%s) to %s",
        len(incoming), kind, covariate_source, reference_time, path,
    )
    return path


def read_forecasts(root: Path = DEFAULT_ROOT, kinds: list[str] | None = None) -> pd.DataFrame:
    """Load every archived forecast, optionally filtered to certain kinds."""
    partitions = sorted(Path(root).glob("forecasts/*.csv"))
    frames = [df for df in (_read_partition(p) for p in partitions) if not df.empty]
    if not frames:
        return pd.DataFrame(columns=METADATA_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    if kinds is not None:
        df = df[df["kind"].isin(kinds)]
    return df.sort_values(_UNIQUE_KEY).reset_index(drop=True)


def load_forecast(
    reference_time,
    *,
    tolerance_hours: float = 2,
    kinds: list[str] | None = None,
    root: Path = DEFAULT_ROOT,
):
    """Return the archived forecast closest to ``reference_time``, or ``None``.

    Only reads the partitions that could plausibly contain a hit. When several
    reference times are equally close, the more trustworthy kind wins.

    Returns ``(df, metadata)`` where ``df`` is indexed by target time and carries only
    the quantile columns, so it is a drop-in replacement for a freshly computed
    forecast, and ``metadata`` records what kind of thing was actually found.
    """
    reference_time = _as_utc(reference_time)
    window = pd.Timedelta(hours=tolerance_hours)

    # A hit within tolerance can only sit in this month or an adjacent one.
    candidates = {
        _partition_path(root, "forecasts", reference_time - window),
        _partition_path(root, "forecasts", reference_time),
        _partition_path(root, "forecasts", reference_time + window),
    }
    frames = [df for df in (_read_partition(p) for p in sorted(candidates)) if not df.empty]
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    if kinds is not None:
        df = df[df["kind"].isin(kinds)]
    if df.empty:
        return None

    df = df.assign(
        _distance=(df["reference_time"] - reference_time).abs(),
        _rank=df["kind"].map(KIND_PRECEDENCE).fillna(-1),
    )
    df = df[df["_distance"] <= window]
    if df.empty:
        logger.info("No archived forecast within %sh of %s", tolerance_hours, reference_time)
        return None

    # Closest first; among equally close reference times, the most trustworthy kind.
    best = df.sort_values(["_distance", "_rank"], ascending=[True, False]).iloc[0]
    chosen = df[df["reference_time"] == best["reference_time"]]
    chosen = chosen[chosen["kind"] == best["kind"]]

    metadata = {
        "reference_time": best["reference_time"],
        "kind": best["kind"],
        "covariate_source": best["covariate_source"],
        "issued_at": best["issued_at"],
        "model_id": best["model_id"],
        "code_version": best["code_version"],
        "distance": best["_distance"],
    }
    logger.info(
        "Using archived %s forecast (ref=%s, %s away from %s)",
        metadata["kind"], metadata["reference_time"], metadata["distance"], reference_time,
    )

    quantile_columns = [c for c in chosen.columns if c not in METADATA_COLUMNS and not c.startswith("_")]
    out = chosen.set_index("target_time")[quantile_columns].sort_index()
    return out, metadata


def write_weather_snapshot(df_weather: pd.DataFrame, *, archived_at=None,
                           reference_time=None, root: Path = DEFAULT_ROOT) -> Path:
    """Store a DWD forecast exactly as it was issued.

    This is what makes ``replay`` backtests possible later: it is the only record of
    what the weather forecast *said*, as opposed to what the weather then did.

    Two timestamps are recorded, and the distinction matters:

    ``archive_timestamp``
        When the snapshot was actually fetched. A fact about the fetch.
    ``reference_time``
        The anchor of the run that fetched it — the last gauge observation. This is what
        a replay looks up by, because the question a replay asks is *which weather
        forecast did the run anchored here actually use?*

    They are never equal: the gauge reading is always a little older than the run that
    consumes it. Keying by fetch time alone would mean a run's own snapshot could never
    replay that run's anchor, which silently turns every replay into an oracle backtest.
    """
    if df_weather.empty:
        raise ValueError("refusing to archive an empty weather snapshot")

    archived_at = _as_utc(archived_at if archived_at is not None else pd.Timestamp.now(tz="UTC"))
    reference_time = _as_utc(reference_time) if reference_time is not None else archived_at

    snapshot = df_weather.copy()
    snapshot["archive_timestamp"] = archived_at.isoformat()
    snapshot["reference_time"] = reference_time

    # Partition by reference time, so a snapshot sits with the run it belongs to.
    path = _partition_path(root, "weather", reference_time)
    existing = _read_partition(path)
    if not existing.empty:
        # One snapshot per anchor. A retry before the gauge advances shares the anchor
        # but not the fetch time, so deduplicating on the fetch would keep both and leave
        # a replay to pick between two sets of the same timestamps arbitrarily. The
        # newest fetch for an anchor is the one that run actually forecast from.
        superseded = _as_utc_series(existing.get("reference_time")) == reference_time
        same_fetch = existing["archive_timestamp"].astype(str) == archived_at.isoformat()
        existing = existing[~(superseded | same_fetch)]
        snapshot = pd.concat([existing, snapshot], ignore_index=True)
    _write_partition(path, snapshot)

    logger.info(
        "Archived %d weather rows for reference %s (fetched %s) to %s",
        len(df_weather), reference_time, archived_at, path,
    )
    return path


def load_weather_snapshot(reference_time, *, max_age_hours: float = 12,
                          root: Path = DEFAULT_ROOT):
    """Return the DWD forecast that was current at ``reference_time``, or ``None``.

    Matching is on the snapshot's ``reference_time`` — the anchor of the run that
    fetched it — not on when the fetch happened. A run reads the gauge, then fetches the
    weather a moment later, so those differ by minutes; keying on the fetch time would
    mean a run's own snapshot could never replay that run's anchor.

    Eligibility is **one-sided**, and two clocks bound it, because a snapshot carries two
    kinds of knowledge:

    - its *anchor* bounds what the run knew about the river,
    - its *fetch time* bounds what it knew about the weather.

    A snapshot anchored exactly at ``reference_time`` is that run's own, so it is the
    forecast that run really used, whenever it happened to be fetched. Falling back to an
    older anchor is different: that snapshot only qualifies if it was also *fetched* by
    ``reference_time``. Otherwise a delayed gauge — a run at 16:00 still anchored to a
    09:00 reading — would hand a 10:00 replay a forecast issued six hours in its future,
    and label the result honest.

    An older snapshot merely makes the replay slightly more pessimistic, which is the
    safe way to be wrong.
    """
    reference_time = _as_utc(reference_time)
    window = pd.Timedelta(hours=max_age_hours)

    candidates = {
        _partition_path(root, "weather", reference_time - window),
        _partition_path(root, "weather", reference_time),
    }
    frames = [df for df in (_read_partition(p) for p in sorted(candidates)) if not df.empty]
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    fetched = _as_utc_series(df["archive_timestamp"])
    # Snapshots written before anchors existed fall back to their fetch time.
    anchor = _as_utc_series(df.get("reference_time")).fillna(fetched)

    in_window = anchor >= reference_time - window
    own_run = anchor == reference_time
    earlier_run = (anchor < reference_time) & (fetched <= reference_time)
    eligible_mask = in_window & (own_run | earlier_run)

    if not eligible_mask.any():
        logger.info("No weather snapshot usable for a replay at %s", reference_time)
        return None

    chosen_at = anchor[eligible_mask].max()
    at_anchor = eligible_mask & (anchor == chosen_at)
    # An anchor may hold more than one fetch in archives written before writes started
    # collapsing them. The newest is the one that run forecast from.
    latest_fetch = fetched[at_anchor].max()
    snapshot = df[at_anchor & (fetched == latest_fetch)].copy()

    logger.info(
        "Replaying weather forecast anchored at %s, fetched %s (%s before %s)",
        chosen_at, latest_fetch, reference_time - chosen_at, reference_time,
    )
    return snapshot, chosen_at


def write_observations(df_observations: pd.DataFrame, root: Path = DEFAULT_ROOT) -> list[Path]:
    """Store measured values, so verification never needs to re-scrape.

    GKD only serves a rolling window, so anything not captured here is eventually
    unrecoverable. Existing timestamps are updated in place rather than duplicated.
    """
    if df_observations.empty:
        return []

    df = df_observations.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df = df.reset_index()

    written = []
    months = df["timestamp"].dt.tz_convert(None).dt.to_period("M")
    for period, group in df.groupby(months):
        path = _partition_path(root, "observations", period.to_timestamp())
        existing = _read_partition(path)
        combined = pd.concat([existing, group], ignore_index=True) if not existing.empty else group
        combined = (
            combined.sort_values("timestamp")
            .drop_duplicates(subset="timestamp", keep="last")
            .reset_index(drop=True)
        )
        _write_partition(path, combined)
        written.append(path)
    return written


def migrate_legacy_forecasts(
    legacy_path: Path = Path("data/forecast_archive/water_temp_predictions_archive.csv"),
    root: Path = DEFAULT_ROOT,
) -> int:
    """Fold the old flat archive into the partitioned layout.

    Every row in the legacy file came from ``run_inference``'s main forecast, which only
    ever ran against the live DWD forecast — so all of it is genuinely ``live`` and must
    be labelled as such. It is also the only copy: these forecasts cannot be recomputed.
    """
    legacy_path = Path(legacy_path)
    if not legacy_path.exists():
        logger.info("No legacy archive at %s, nothing to migrate", legacy_path)
        return 0

    legacy = pd.read_csv(legacy_path, parse_dates=["reference_time", "target_time"])
    if legacy.empty:
        return 0

    legacy["reference_time"] = pd.to_datetime(legacy["reference_time"], utc=True)
    legacy["target_time"] = pd.to_datetime(legacy["target_time"], utc=True)

    migrated = 0
    for reference_time, group in legacy.groupby("reference_time"):
        quantiles = group.drop(columns=["reference_time"]).set_index("target_time")
        write_forecast(
            quantiles,
            reference_time=reference_time,
            kind=KIND_LIVE,
            covariate_source=COVARIATE_DWD_FORECAST,
            # The run's wall clock was never recorded; the reference time is the closest
            # honest approximation, and is at most a couple of hours early.
            issued_at=reference_time,
            model_id="",
            version="legacy",
            root=root,
        )
        migrated += len(group)

    logger.info("Migrated %d legacy rows from %s", migrated, legacy_path)
    return migrated
