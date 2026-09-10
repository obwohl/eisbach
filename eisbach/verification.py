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
from pathlib import Path

import numpy as np
import pandas as pd

from eisbach import archive
from eisbach.data import COVARIATE_SHIFT_HOURS
from eisbach.model import QUANTILES

logger = logging.getLogger(__name__)

#: The channel that is scored. The covariate quantiles are on their way out of the
#: forecast store (PRD R4) and nothing reads them; water is what the product forecasts.
SCORED_CHANNEL = "wassertemp"

QUANTILE_COLUMNS = tuple(f"{SCORED_CHANNEL}_q{q}" for q in QUANTILES)

#: Fraction of observations at or below each predicted quantile.
PIT_COLUMNS = tuple(f"pit_le_q{q}" for q in QUANTILES)

#: One scored row, in the order it is stored. Named from :data:`QUANTILES` so the store
#: cannot drift from what the model emits, which is why this lives here rather than in
#: ``archive``: that module stores what it is handed and never imports the model.
SCORE_COLUMNS = (
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
    *PIT_COLUMNS,
    "scored_at",
)

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

#: Intervals :func:`read_scores` materialises, as (name, lower quantile, upper quantile).
#: These are differences of PIT knots, so they are exact rather than re-derived from data.
COVERAGE_INTERVALS = (
    ("cov_50", 0.25, 0.75),
    ("cov_90", 0.05, 0.95),
    ("cov_98", 0.01, 0.99),
)

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


def _quantile_matrix(forecast: pd.DataFrame) -> np.ndarray:
    """The quantile columns as a sorted array, one row per target hour.

    Sorting repairs quantile crossing — a predicted q0.75 below its own q0.5. The
    checkpoint in use has never produced one (0 of 5 398 scored rows), but PIT
    interpolation reads the columns as a CDF, and a CDF that goes backwards is not one.
    """
    return np.sort(forecast[list(QUANTILE_COLUMNS)].to_numpy(dtype=float), axis=1)


def crps(values: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """CRPS per observation, from the quantiles the model emits.

    The continuous ranked probability score is twice the integral of the pinball loss
    over the quantile level, so with a discrete set of levels it is a trapezoid over
    them. Integrating rather than averaging the levels matters here: they are unevenly
    spaced (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), and an equal-weight mean would give
    the two tails the same say as the whole middle.

    The integral covers [0.01, 0.99] because that is what the model reports, so the value
    is a slight under-estimate of the true CRPS — it omits the two tails beyond. That is
    a constant of the method, not of the forecast, so the series stays comparable with
    itself.
    """
    levels = np.asarray(QUANTILES, dtype=float)
    residual = truth[:, None] - values
    pinball = np.where(residual >= 0, levels * residual, (levels - 1) * residual)
    widths = np.diff(levels)
    return 2.0 * ((pinball[:, :-1] + pinball[:, 1:]) * widths / 2.0).sum(axis=1)


def pit(values: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Probability integral transform, interpolated between the reported quantiles.

    An observation outside the reported range is clamped to the outermost level rather
    than extrapolated: the model says nothing about where in the far tail the value sits,
    and inventing a 0 or a 1 would be a claim it never made. Well calibrated means
    uniform on [0, 1], so a mean near 0.5 says the median is unbiased and says nothing at
    all about the spread — that is what the PIT knots are for.
    """
    levels = np.asarray(QUANTILES, dtype=float)
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
) -> pd.DataFrame:
    """Score one forecast against measured water temperature, one row per lead bucket.

    ``forecast`` is indexed by target time and carries the seven water-temperature
    quantile columns; ``actuals`` is the measured series. A bucket with no observation to
    compare against yields no row at all — an empty row would be indistinguishable from a
    perfect one once pooled.
    """
    missing = [c for c in QUANTILE_COLUMNS if c not in forecast.columns]
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

        values = _quantile_matrix(in_bucket.loc[overlap])
        truth = truth_all.loc[overlap].to_numpy(dtype=float)
        median = values[:, QUANTILES.index(0.5)]
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
            "crps": float(crps(values, truth).mean()),
            "mae_persistence": (
                float(np.abs(baseline - truth).mean()) if np.isfinite(baseline) else float("nan")
            ),
            # `nanmean` of an all-NaN slice warns and returns NaN, which is the answer
            # wanted without the warning.
            "mae_diurnal": (
                float(np.nanmean(np.abs(diurnal - truth))) if np.isfinite(diurnal).any()
                else float("nan")
            ),
            "pit_mean": float(pit(values, truth).mean()),
        }
        for level, column in enumerate(PIT_COLUMNS):
            row[column] = float((truth <= values[:, level]).mean())
        row["scored_at"] = scored_at
        rows.append(row)

    return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))


def score_archive(root: Path = archive.DEFAULT_ROOT, *, scored_at=None) -> pd.DataFrame:
    """Score every archived run whose window has closed and that has no score yet.

    Returns the rows written, so a caller can log how much the series grew. Reads the
    whole forecast store — a few tens of MB of CSV — which is cheap next to a model run
    and keeps this correct when a partition is backfilled out of order.
    """
    forecasts = archive.read_forecasts(root=root)
    if forecasts.empty:
        logger.info("No archived forecasts to score")
        return pd.DataFrame(columns=list(SCORE_COLUMNS))

    observations = archive.read_observations(root=root)
    actuals = (
        observations[SCORED_CHANNEL].dropna()
        if SCORED_CHANNEL in observations.columns
        else pd.Series(dtype=float)
    )
    if actuals.empty:
        logger.info("No archived %s observations, so nothing can be scored yet", SCORED_CHANNEL)
        return pd.DataFrame(columns=list(SCORE_COLUMNS))
    last_measured = actuals.index.max()

    existing = archive.read_verification(root=root)
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
        )
        if not rows.empty:
            scored.append(rows)

    if not scored:
        logger.info(
            "Nothing new to score (%d runs already scored, %d still open)",
            skipped_done, skipped_open,
        )
        return pd.DataFrame(columns=list(SCORE_COLUMNS))

    fresh = pd.concat(scored, ignore_index=True)
    archive.write_verification(fresh, root=root)
    logger.info(
        "Scored %d runs (%d rows); %d already scored, %d still open",
        fresh["reference_time"].nunique(), len(fresh), skipped_done, skipped_open,
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
                kinds: tuple[str, ...] | None = HONEST_KINDS) -> pd.DataFrame:
    """The verification table, with interval coverage materialised.

    Defaults to the honest kinds. An oracle backtest saw the weather that actually
    occurred, so pooling its scores with the rest produces a number that describes no
    system that has ever run; pass ``kinds=None`` to get everything, and then keep the
    ``kind`` column in whatever you report.

    Beware of pooling across ``model_id`` as well. The eras in this archive do not
    overlap in time, so a difference between them is code change perfectly confounded
    with season — see the PRD.
    """
    df = archive.read_verification(root=root)
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
    for name, low, high in COVERAGE_INTERVALS:
        df[name] = df[f"pit_le_q{high}"] - df[f"pit_le_q{low}"]
    return df


#: Columns that pool as an ``n``-weighted mean.
_MEAN_COLUMNS = (
    "mae", "bias", "crps", "mae_persistence", "mae_diurnal", "pit_mean", *PIT_COLUMNS,
)


def pool(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Aggregate scored rows correctly, weighting each by the hours it covers.

    A plain ``groupby.mean()`` is wrong twice over: it weights a bucket with two
    observations like one with twenty-four, and it averages RMSE instead of the squares
    it is a root of. This does both properly, and re-derives interval coverage from the
    pooled PIT knots.

    ``by`` defaults to the lead bucket. Pass ``[]`` to collapse to a single row, or e.g.
    ``["model_id", "lead_lo"]`` to keep the eras apart.
    """
    if df.empty:
        return df
    by = ["lead_lo"] if by is None else by

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
    for name, low, high in COVERAGE_INTERVALS:
        out[name] = out[f"pit_le_q{high}"] - out[f"pit_le_q{low}"]
    return out.reset_index(drop=not by)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=archive.DEFAULT_ROOT,
                        help="archive root (default: %(default)s)")
    parser.add_argument("--report", action="store_true",
                        help="print the pooled table for the honest kinds and exit")
    args = parser.parse_args(argv)

    if args.report:
        scores = read_scores(root=args.root)
        if scores.empty:
            logger.info("No scores stored yet")
            return 0
        print(pool(scores).to_string())
        return 0

    score_archive(root=args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
