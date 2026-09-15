"""Score past runs against what actually happened, and keep the score.

Every number in `docs/PRD.md` was recomputed from scratch, from an archive that only
grows. That is fine once; it is useless as a way to notice that the model got worse last
Tuesday. This module turns a one-off analysis into a series: one row per (run, kind, lead
bucket), written once, never revised.

**A run is scored only after its window has closed.** A forecast made at 09:00 says
something about the next 96 hours, and until those hours have been measured there is
nothing to score. So a run becomes scorable when the observation store reaches its last
target time, and at that point the score is final — which is what lets this store obey
the same append-only rule as the rest of `data/archive/`. A row is written once and is
never recomputed, so re-running the scorer is a no-op rather than a rewrite.

**The buckets do not overlap.** Leads are cut into four 24-hour buckets and nothing
stores a pooled "all leads" row, deliberately: with a total row in the same table, the
obvious `df["mae"].mean()` would count every observation twice. Pool with :func:`pool`,
which weights by `n` and knows that RMSE averages in squares.

**What is stored is the sufficient statistic, not the conclusion.** Calibration is kept
as the seven `pit_le_q*` columns — the fraction of observations at or below each
predicted quantile, which is the PIT histogram evaluated at the levels the model actually
emits. Every interval coverage falls out of a subtraction (`read_scores` does it), and
unlike a stored `cov_50` the knots pool correctly across runs.

**`kind` travels with every row.** An oracle backtest saw weather nobody could have
known, so its scores flatter the model and must never be pooled with the honest ones
(P2). :func:`read_scores` therefore returns `live` and `replay` only unless asked
otherwise, and pooling across `model_id` is a different lie the docstrings warn about.

Run it with ``python -m eisbach.verification``; the scheduled workflow does exactly that
before each forecast.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eisbach import archive, timesfm
from eisbach.data import COVARIATE_SHIFT_HOURS
from eisbach.model import QUANTILES

logger = logging.getLogger(__name__)

#: The channel that is scored. The covariate quantiles are on their way out of the
#: forecast store (PRD R4) and nothing reads them; water is what the product forecasts.
SCORED_CHANNEL = "wassertemp"


@dataclass(frozen=True)
class Scheme:
    """One model's scoring: which quantiles it emits, read from where, written to where.

    Two models run here and they do not report the same quantiles. DUET emits seven,
    TimesFM exactly nine deciles, and a CRPS integrated over [0.1, 0.9] is a smaller
    number than one integrated over [0.01, 0.99] for the same forecast — it omits more
    of the tails. So each model is scored on its own grid, into its own store, and the
    two `crps` columns are **not comparable**. Comparing them is what `compare` is for,
    and it does the interpolation onto a shared grid that makes it honest.

    Everything derived from the quantiles is derived here, so a store cannot drift from
    what its model emits.
    """

    name: str
    quantiles: tuple[float, ...]
    forecasts: str = "forecasts"
    verification: str = "verification"
    #: Intervals :func:`read_scores` materialises, as (name, lower, upper). Differences
    #: of PIT knots, so exact rather than re-derived from data.
    intervals: tuple[tuple[str, float, float], ...] = ()

    @property
    def quantile_columns(self) -> tuple[str, ...]:
        return tuple(f"{SCORED_CHANNEL}_q{q}" for q in self.quantiles)

    @property
    def pit_columns(self) -> tuple[str, ...]:
        return tuple(f"pit_le_q{q}" for q in self.quantiles)

    @property
    def columns(self) -> tuple[str, ...]:
        """One scored row, in the order it is stored."""
        return (
            "reference_time",
            "kind",
            "model_id",
            "code_version",
            "lead_lo",
            "lead_hi",
            "n",
            "n_forecast",
            "mae",
            "rmse",
            "bias",
            "crps",
            "mae_persistence",
            "mae_diurnal",
            "pit_mean",
            *self.pit_columns,
            "scored_at",
        )


#: The production model. Every default in this module is this one, so nothing about
#: DUET's series changed when the candidate arrived.
PRODUCTION = Scheme(
    name="duet",
    quantiles=tuple(QUANTILES),
    intervals=(("cov_50", 0.25, 0.75), ("cov_90", 0.05, 0.95), ("cov_98", 0.01, 0.99)),
)

#: The candidate. Deciles only, so its widest stored interval covers 80 % where the
#: production model's covers 98 %.
CANDIDATE = Scheme(
    name="timesfm",
    quantiles=timesfm.DECILES,
    forecasts=timesfm.ARCHIVE_STORE,
    verification=f"verification_{timesfm.ARCHIVE_STORE}",
    intervals=(("cov_60", 0.2, 0.8), ("cov_80", 0.1, 0.9)),
)

SCHEMES = (PRODUCTION, CANDIDATE)

QUANTILE_COLUMNS = PRODUCTION.quantile_columns
PIT_COLUMNS = PRODUCTION.pit_columns
SCORE_COLUMNS = PRODUCTION.columns

#: Width of one lead bucket. The horizon divides into four of these.
LEAD_BUCKET_HOURS = 24

#: How far the anchor observation may be from the reference time before the persistence
#: baseline is recorded as missing rather than guessed. A live run anchors on a real
#: gauge reading, so this only bites on a backtest whose anchor hour was never measured.
PERSISTENCE_ANCHOR_TOLERANCE = pd.Timedelta(hours=3)

#: The period the diurnal baseline repeats. The Eisbach's dominant signal at these
#: leads is the day/night cycle, and a baseline that cannot reproduce it is too weak to
#: prove anything against.
DIURNAL_PERIOD_HOURS = 24

#: Kinds whose scores may be read as evidence about the real world. An oracle backtest
#: is excluded by default rather than by convention — see the module docstring.
HONEST_KINDS = (archive.KIND_LIVE, archive.KIND_REPLAY)


def lead_buckets() -> list[tuple[int, int]]:
    """The lead ranges scored, as half-open ``(lo, hi]`` pairs in hours.

    Derived from the horizon rather than written out, so a change to
    ``COVARIATE_SHIFT_HOURS`` moves the buckets with it instead of silently leaving the
    last hours unscored.
    """
    edges = list(range(0, COVARIATE_SHIFT_HOURS, LEAD_BUCKET_HOURS))
    edges.append(COVARIATE_SHIFT_HOURS)
    return list(zip(edges[:-1], edges[1:], strict=True))


def _quantile_matrix(forecast: pd.DataFrame, scheme: Scheme = PRODUCTION) -> np.ndarray:
    """The quantile columns as a sorted array, one row per target hour.

    Sorting repairs quantile crossing — a predicted q0.75 below its own q0.5. The
    checkpoint in use has never produced one (0 of 5 398 scored rows), but PIT
    interpolation reads the columns as a CDF, and a CDF that goes backwards is not one.
    """
    return np.sort(forecast[list(scheme.quantile_columns)].to_numpy(dtype=float), axis=1)


def crps(values: np.ndarray, truth: np.ndarray, levels=None) -> np.ndarray:
    """CRPS per observation, from the quantiles the model emits.

    The continuous ranked probability score is twice the integral of the pinball loss
    over the quantile level, so with a discrete set of levels it is a trapezoid over
    them. Integrating rather than averaging the levels matters here: they are unevenly
    spaced (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), and an equal-weight mean would give
    the two tails the same say as the whole middle.

    The integral covers the levels the model reports — [0.01, 0.99] for DUET — so the
    value is a slight under-estimate of the true CRPS: it omits the two tails beyond.
    That is a constant of the method, not of the forecast, so the series stays comparable
    **with itself**. It is not comparable across models that report different levels: the
    candidate's deciles stop at 0.1 and 0.9, which omits far more tail and would hand it
    a smaller number for the same forecast. See :func:`compare`.
    """
    levels = np.asarray(QUANTILES if levels is None else levels, dtype=float)
    residual = truth[:, None] - values
    pinball = np.where(residual >= 0, levels * residual, (levels - 1) * residual)
    widths = np.diff(levels)
    return 2.0 * ((pinball[:, :-1] + pinball[:, 1:]) * widths / 2.0).sum(axis=1)


def pit(values: np.ndarray, truth: np.ndarray, levels=None) -> np.ndarray:
    """Probability integral transform, interpolated between the reported quantiles.

    An observation outside the reported range is clamped to the outermost level rather
    than extrapolated: the model says nothing about where in the far tail the value sits,
    and inventing a 0 or a 1 would be a claim it never made. Well calibrated means
    uniform on [0, 1], so a mean near 0.5 says the median is unbiased and says nothing at
    all about the spread — that is what the PIT knots are for.
    """
    levels = np.asarray(QUANTILES if levels is None else levels, dtype=float)
    return np.array([np.interp(y, row, levels) for y, row in zip(truth, values, strict=True)])


def _persistence(actuals: pd.Series, reference_time: pd.Timestamp) -> float:
    """The observation the run anchored on, held flat — the baseline worth beating.

    ``nan`` when no measurement sits close enough behind the anchor, so a run without a
    baseline records that rather than a fabricated one.
    """
    prior = actuals.loc[:reference_time]
    if prior.empty or reference_time - prior.index[-1] > PERSISTENCE_ANCHOR_TOLERANCE:
        return float("nan")
    return float(prior.iloc[-1])


def _diurnal(actuals: pd.Series, targets: pd.DatetimeIndex, leads: np.ndarray) -> np.ndarray:
    """The same hour, on the most recent whole day the run had already seen.

    Flat persistence is a weak baseline for a signal whose largest component is the
    daily cycle: it is wrong by most of that cycle's amplitude before the forecast has
    done anything. This one repeats the last observed day instead, so beating it means
    beating "tomorrow looks like yesterday" rather than "tomorrow looks like right now".

    Causal by construction: the lag is the smallest whole number of days that lands at or
    before the anchor, so a lead of 30 h reads 48 h back, not 24. An hour the gauge never
    measured yields ``nan`` and is left out of the mean rather than filled in.
    """
    lag = DIURNAL_PERIOD_HOURS * np.ceil(leads / DIURNAL_PERIOD_HOURS)
    lagged = targets - pd.to_timedelta(lag, unit="h")
    return actuals.reindex(lagged).to_numpy(dtype=float)


def score_forecast(
    forecast: pd.DataFrame,
    actuals: pd.Series,
    *,
    reference_time,
    kind: str,
    model_id: str = "",
    code_version: str = "",
    scored_at=None,
    scheme: Scheme = PRODUCTION,
) -> pd.DataFrame:
    """Score one forecast against measured water temperature, one row per lead bucket.

    ``forecast`` is indexed by target time and carries ``scheme``'s water-temperature
    quantile columns; ``actuals`` is the measured series. A bucket with no observation to
    compare against yields no row at all — an empty row would be indistinguishable from a
    perfect one once pooled.
    """
    missing = [c for c in scheme.quantile_columns if c not in forecast.columns]
    if missing:
        raise ValueError(f"cannot score a forecast missing {', '.join(missing)}")

    reference_time = pd.Timestamp(reference_time)
    scored_at = pd.Timestamp(scored_at) if scored_at is not None else pd.Timestamp.now(tz="UTC")
    truth_all = actuals.dropna()
    baseline = _persistence(truth_all, reference_time)

    forecast = forecast.sort_index()
    leads = (forecast.index - reference_time).total_seconds() / 3600.0

    rows = []
    for low, high in lead_buckets():
        in_bucket = forecast[(leads > low) & (leads <= high)]
        overlap = in_bucket.index.intersection(truth_all.index)
        if len(overlap) == 0:
            continue

        values = _quantile_matrix(in_bucket.loc[overlap], scheme)
        truth = truth_all.loc[overlap].to_numpy(dtype=float)
        median = values[:, scheme.quantiles.index(0.5)]
        error = median - truth
        diurnal = _diurnal(
            truth_all, overlap, (overlap - reference_time).total_seconds() / 3600.0
        )

        row = {
            "reference_time": reference_time,
            "kind": kind,
            "model_id": model_id,
            "code_version": code_version,
            "lead_lo": low,
            "lead_hi": high,
            # `n_forecast` is how many hours the run predicted in this bucket; `n` is how
            # many of them were ever measured. They differ exactly when the gauge had a
            # gap, which is a reason to distrust the row rather than to drop it.
            "n": len(overlap),
            "n_forecast": len(in_bucket),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt((error**2).mean())),
            "bias": float(error.mean()),
            "crps": float(crps(values, truth, scheme.quantiles).mean()),
            "mae_persistence": (
                float(np.abs(baseline - truth).mean()) if np.isfinite(baseline) else float("nan")
            ),
            # `nanmean` of an all-NaN slice warns and returns NaN, which is the answer
            # wanted without the warning.
            "mae_diurnal": (
                float(np.nanmean(np.abs(diurnal - truth))) if np.isfinite(diurnal).any()
                else float("nan")
            ),
            "pit_mean": float(pit(values, truth, scheme.quantiles).mean()),
        }
        for level, column in enumerate(scheme.pit_columns):
            row[column] = float((truth <= values[:, level]).mean())
        row["scored_at"] = scored_at
        rows.append(row)

    return pd.DataFrame(rows, columns=list(scheme.columns))


def score_archive(root: Path = archive.DEFAULT_ROOT, *, scored_at=None,
                  scheme: Scheme = PRODUCTION) -> pd.DataFrame:
    """Score every archived run whose window has closed and that has no score yet.

    Returns the rows written, so a caller can log how much the series grew. Reads the
    whole forecast store — a few tens of MB of CSV — which is cheap next to a model run
    and keeps this correct when a partition is backfilled out of order.

    Both models are scored against the same measured series: the candidate forecasts the
    same river, so anything else would make the two incomparable before they started.
    """
    forecasts = archive.read_forecasts(root=root, store=scheme.forecasts)
    if forecasts.empty:
        logger.info("No archived %s forecasts to score", scheme.name)
        return pd.DataFrame(columns=list(scheme.columns))

    observations = archive.read_observations(root=root)
    actuals = (
        observations[SCORED_CHANNEL].dropna()
        if SCORED_CHANNEL in observations.columns
        else pd.Series(dtype=float)
    )
    if actuals.empty:
        logger.info("No archived %s observations, so nothing can be scored yet", SCORED_CHANNEL)
        return pd.DataFrame(columns=list(scheme.columns))
    last_measured = actuals.index.max()

    existing = archive.read_verification(root=root, store=scheme.verification)
    already_scored = set()
    if not existing.empty:
        already_scored = set(zip(existing["reference_time"], existing["kind"], strict=True))

    scored, skipped_open, skipped_done = [], 0, 0
    for (reference_time, kind), group in forecasts.groupby(["reference_time", "kind"], sort=True):
        if (reference_time, kind) in already_scored:
            skipped_done += 1
            continue
        if group["target_time"].max() > last_measured:
            # The window is still open. Scoring it now would freeze a partial answer into
            # a store that is never revised.
            skipped_open += 1
            continue

        indexed = group.set_index("target_time")
        rows = score_forecast(
            indexed,
            actuals,
            reference_time=reference_time,
            kind=kind,
            model_id=_single(group, "model_id"),
            code_version=_single(group, "code_version"),
            scored_at=scored_at,
            scheme=scheme,
        )
        if not rows.empty:
            scored.append(rows)

    if not scored:
        logger.info(
            "Nothing new to score for %s (%d runs already scored, %d still open)",
            scheme.name, skipped_done, skipped_open,
        )
        return pd.DataFrame(columns=list(scheme.columns))

    fresh = pd.concat(scored, ignore_index=True)
    archive.write_verification(fresh, root=root, store=scheme.verification)
    logger.info(
        "Scored %d %s runs (%d rows); %d already scored, %d still open",
        fresh["reference_time"].nunique(), scheme.name, len(fresh), skipped_done, skipped_open,
    )
    return fresh


def _single(group: pd.DataFrame, column: str) -> str:
    """The one value a run should carry for ``column``, as text.

    A run is one model on one commit, so a group with several is malformed rather than
    interesting; keep the first and say so, because the alternative is a row that claims
    a provenance it does not have.
    """
    values = group[column].dropna().astype(str).unique()
    if len(values) > 1:
        logger.warning("Run %s carries %d values for %s: %s", group["reference_time"].iloc[0],
                       len(values), column, ", ".join(values))
    return values[0] if len(values) else ""


def read_scores(root: Path = archive.DEFAULT_ROOT,
                kinds: tuple[str, ...] | None = HONEST_KINDS,
                *, scheme: Scheme = PRODUCTION) -> pd.DataFrame:
    """The verification table, with interval coverage materialised.

    Defaults to the honest kinds. An oracle backtest saw the weather that actually
    occurred, so pooling its scores with the rest produces a number that describes no
    system that has ever run; pass ``kinds=None`` to get everything, and then keep the
    ``kind`` column in whatever you report.

    Beware of pooling across ``model_id`` as well. The eras in this archive do not
    overlap in time, so a difference between them is code change perfectly confounded
    with season — see the PRD.
    """
    df = archive.read_verification(root=root, store=scheme.verification)
    if df.empty:
        return df
    # 77 of the live runs in this archive predate `model_id` being recorded and carry a
    # blank, which `read_csv` turns into NaN — and a NaN key is dropped by a default
    # `groupby`. Grouping by era would then quietly answer for one era while looking like
    # it answered for both, so the blank stays a blank.
    for column in ("model_id", "code_version"):
        if column in df.columns:
            df[column] = df[column].fillna("")
    if kinds is not None:
        df = df[df["kind"].isin(kinds)].reset_index(drop=True)
    for name, low, high in scheme.intervals:
        df[name] = df[f"pit_le_q{high}"] - df[f"pit_le_q{low}"]
    return df


#: Columns that pool as an ``n``-weighted mean. Both schemes' PIT knots are listed:
#: `pool` keeps only the ones a given table actually carries.
_MEAN_COLUMNS = (
    "mae", "bias", "crps", "mae_persistence", "mae_diurnal", "pit_mean",
    *dict.fromkeys(PRODUCTION.pit_columns + CANDIDATE.pit_columns),
)


#: What :func:`pool` groups by when not told otherwise. Era and kind are in it because
#: the module's own warnings say they must be: a default that pools a legacy run with a
#: current one, or a replay with a live, hands back a number describing no system that
#: has ever run. Collapsing them has to be asked for.
DEFAULT_POOL_KEYS = ["model_id", "kind", "lead_lo"]


def pool(df: pd.DataFrame, by: list[str] | None = None, *,
         scheme: Scheme = PRODUCTION) -> pd.DataFrame:
    """Aggregate scored rows correctly, weighting each by the hours it covers.

    A plain ``groupby.mean()`` is wrong twice over: it weights a bucket with two
    observations like one with twenty-four, and it averages RMSE instead of the squares
    it is a root of. This does both properly, and re-derives interval coverage from the
    pooled PIT knots.

    ``by`` defaults to :data:`DEFAULT_POOL_KEYS`, which keeps the eras and the backtest
    kinds apart. Pass ``[]`` to collapse to a single row, or any other list of columns —
    but if that list drops ``model_id`` or ``kind``, know what you are mixing.
    """
    if df.empty:
        return df
    by = list(DEFAULT_POOL_KEYS) if by is None else by

    columns = [c for c in _MEAN_COLUMNS if c in df.columns]
    weights = df["n"].to_numpy(dtype=float)
    weighted = df.assign(
        **{c: df[c] * weights for c in columns},
        _sq=df["rmse"] ** 2 * weights,
        # Each column carries its own denominator, because a row can be missing one and
        # not the others: a backtest whose anchor hour was never measured has no
        # persistence baseline but a perfectly good MAE. Dividing every column by the
        # same total would quietly shrink whichever one had a gap.
        **{f"_w_{c}": np.where(df[c].notna(), weights, 0.0) for c in columns},
        _w_rmse=np.where(df["rmse"].notna(), weights, 0.0),
    )
    # `dropna=False`: a run whose provenance was never recorded still happened, and a
    # grouping that discards it reports on a subset while looking like it reports on all.
    grouped = (
        weighted.groupby(by, sort=True, dropna=False) if by
        else weighted.groupby(lambda _: 0, sort=True)
    )
    totals = grouped[["n", "n_forecast"]].sum()
    sums = grouped[columns + ["_sq"]].sum()
    denominators = grouped[[f"_w_{c}" for c in columns] + ["_w_rmse"]].sum()

    out = pd.DataFrame(index=sums.index)
    for column in columns:
        out[column] = sums[column] / denominators[f"_w_{column}"].replace(0.0, np.nan)
    out["rmse"] = np.sqrt(sums["_sq"] / denominators["_w_rmse"].replace(0.0, np.nan))
    out.insert(0, "n", totals["n"])
    out.insert(1, "n_forecast", totals["n_forecast"])
    out["runs"] = grouped["reference_time"].nunique()
    for name, low, high in scheme.intervals:
        out[name] = out[f"pit_le_q{high}"] - out[f"pit_le_q{low}"]
    return out.reset_index(drop=not by)


#: The grid the two models are compared on. TimesFM emits exactly these; DUET's seven
#: quantiles are interpolated onto them. Every decile lies inside [0.01, 0.99], so the
#: interpolation never extrapolates — which is why the shared grid is the candidate's and
#: not the union of both.
SHARED_GRID = CANDIDATE.quantiles


def to_grid(values: np.ndarray, levels, grid=SHARED_GRID) -> np.ndarray:
    """Interpolate a quantile function reported at ``levels`` onto ``grid``.

    Linear in the quantile level, which is what makes the two CRPS numbers mean the same
    thing. Integrating each model over its own range instead would hand the candidate a
    smaller number for free: its levels stop at 0.1 and 0.9 and omit far more of the
    tails than DUET's 0.01 and 0.99.
    """
    levels = np.asarray(levels, dtype=float)
    grid = np.asarray(grid, dtype=float)
    return np.array([np.interp(grid, levels, row) for row in np.sort(values, axis=1)])


def compare(root: Path = archive.DEFAULT_ROOT, *, kind: str = archive.KIND_LIVE,
            schemes: tuple[Scheme, ...] = SCHEMES) -> pd.DataFrame:
    """Both models on the hours where both really forecast, scored on one grid.

    This is the measurement the whole research programme had to caveat. Every covariate
    number in ``experiments/timesfm/`` used the weather that actually occurred, because
    archived DWD forecasts began in May 2026 and only for Munich. Here both models saw
    only what was knowable at the time, and they are compared on identical hours: an
    inner join on ``(reference_time, target_time)``, so a run where the two picked
    different anchors drops out of both sides rather than being compared against a
    different window.

    Nothing is stored. The forecast archives are kept forever and this is a pure function
    of them, so it can be recomputed whenever the question is asked — and a change to the
    scoring cannot quietly rewrite a history it disagrees with.

    Only closed windows count, the same rule `score_archive` writes rows under: a run
    whose 96 hours have not all been measured is left out entirely rather than
    contributing its first few hours to the first lead bucket and nothing to the rest.

    Returns one row per (model, lead bucket). It will be empty until both models have
    published over the same closed windows, which is the point: the candidate's archive
    starts the day it went live.
    """
    observations = archive.read_observations(root=root)
    actuals = (observations[SCORED_CHANNEL].dropna() if SCORED_CHANNEL in observations.columns
               else pd.Series(dtype=float))
    if actuals.empty:
        return pd.DataFrame()

    frames = {}
    for scheme in schemes:
        stored = archive.read_forecasts(root=root, store=scheme.forecasts, kinds=[kind])
        if stored.empty:
            logger.info("No %s forecasts of kind %s to compare", scheme.name, kind)
            return pd.DataFrame()
        frames[scheme.name] = stored.set_index(["reference_time", "target_time"]).sort_index()

    shared = None
    for indexed in frames.values():
        keys = indexed.index.unique()
        shared = keys if shared is None else shared.intersection(keys)

    # Only closed windows, the same rule `score_archive` writes rows under. An hour of a
    # run still unfolding has been measured and would otherwise be kept, but a run that
    # is three hours old contributes to the first lead bucket and to none of the others.
    # At three runs a day several are open at once, so the early buckets would carry more
    # — and more recent — runs than the late ones, and the reported comparison would
    # drift as observations arrived rather than as the models changed. Nothing is stored
    # here, so this is not a frozen partial answer; it is a misleading live one.
    last_measured = actuals.index.max()
    closes = None
    for indexed in frames.values():
        ends = indexed.index.to_frame(index=False).groupby("reference_time").target_time.max()
        closes = ends if closes is None else pd.concat([closes, ends], axis=1).max(axis=1)
    open_runs = closes.index[closes > last_measured]
    if len(open_runs):
        logger.info("Leaving out %d run(s) whose window has not closed yet", len(open_runs))
        shared = shared[~shared.get_level_values(0).isin(open_runs)]

    shared = shared[shared.get_level_values(1).isin(actuals.index)]
    if len(shared) == 0:
        logger.info("No closed window has been forecast by every model and then measured")
        return pd.DataFrame()

    truth = actuals.reindex(shared.get_level_values(1)).to_numpy(dtype=float)
    leads = ((shared.get_level_values(1) - shared.get_level_values(0))
             .total_seconds().to_numpy() / 3600.0)
    rows = []
    for scheme in schemes:
        indexed = frames[scheme.name].loc[shared]
        values = to_grid(indexed[list(scheme.quantile_columns)].to_numpy(dtype=float),
                         scheme.quantiles)
        median = values[:, SHARED_GRID.index(0.5)]
        for low, high in lead_buckets():
            sel = (leads > low) & (leads <= high)
            if not sel.any():
                continue
            error = median[sel] - truth[sel]
            rows.append({
                "model": scheme.name,
                "lead_lo": low,
                "lead_hi": high,
                "n": int(sel.sum()),
                "runs": shared[sel].get_level_values(0).nunique(),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt((error ** 2).mean())),
                "bias": float(error.mean()),
                "crps": float(crps(values[sel], truth[sel], SHARED_GRID).mean()),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=archive.DEFAULT_ROOT,
                        help="archive root (default: %(default)s)")
    parser.add_argument("--report", action="store_true",
                        help="print the scores for the honest kinds, by era, kind and "
                             "lead bucket, and exit")
    parser.add_argument("--compare", action="store_true",
                        help="print both models on the hours both really forecast, "
                             "scored on one shared decile grid, and exit")
    parser.add_argument("--model", choices=[s.name for s in SCHEMES],
                        help="score or report only this model (default: all of them)")
    args = parser.parse_args(argv)
    schemes = tuple(s for s in SCHEMES if args.model in (None, s.name))

    if args.compare:
        head_to_head = compare(root=args.root)
        if head_to_head.empty:
            logger.info("Nothing both models have forecast and that has since been measured")
            return 0
        print(head_to_head.to_string(index=False))
        return 0

    if args.report:
        for scheme in schemes:
            scores = read_scores(root=args.root, scheme=scheme)
            if scores.empty:
                logger.info("No %s scores stored yet", scheme.name)
                continue
            print(f"== {scheme.name} ==")
            print(pool(scores, scheme=scheme).to_string())
        return 0

    for scheme in schemes:
        try:
            score_archive(root=args.root, scheme=scheme)
        except Exception:
            if scheme is PRODUCTION:
                raise
            # Same rule as the forecast itself: the candidate never takes the run down.
            # This step runs before `main.py` in the scheduled workflow, so raising here
            # would cost a forecast cycle over a model nobody is relying on yet.
            logger.exception("Scoring the %s candidate failed; the production scores stand",
                             scheme.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
