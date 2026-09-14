"""Benchmark harness for zero-shot TimesFM 3.0 on the Eisbach.

Side project. Nothing here is imported by ``main.py`` or ``eisbach/``; it reads the
production archive read-only and never writes to it.

Two rules carried over from the production pipeline, because breaking them is how a
benchmark lies:

* **Scores are computed on one shared quantile grid.** The production CRPS integrates
  the pinball loss over the levels the model reports. DUET reports [0.01 .. 0.99],
  TimesFM reports [0.1 .. 0.9]; integrating each over its own range would hand TimesFM a
  smaller number for free. Everything below integrates over the deciles, and DUET's
  quantiles are interpolated onto them.
* **Weather that was not knowable is labelled.** A run fed observed air temperature over
  its horizon is an ``oracle`` run and is never presented as real-world skill.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "experiments"
ARCHIVE = REPO / "data" / "archive"

HORIZON = 96
LEAD_BUCKETS = ((0, 24), (24, 48), (48, 72), (72, 96))

#: Six-hour resolution over the first day, because that is where an upstream gauge can
#: still be ahead of the river: Beuerberg is hours away, not half a day. Coarse after
#: that, where the travel time has been outrun and nothing changes within a bucket.
FINE_BUCKETS = ((0, 6), (6, 12), (12, 18), (18, 24), (24, 48), (48, 72), (72, 96))

#: The shared grid. TimesFM emits exactly these; DUET is interpolated onto them.
DECILES = np.round(np.arange(0.1, 0.91, 0.1), 2)

#: What the production model stores.
DUET_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

#: Water temperatures outside this are instrument faults, not weather. The gauge has
#: produced 65.8 C (hourly mean) and the production archive holds 154.4 C for the same
#: hour, so this is a real failure mode rather than a hypothetical one.
PLAUSIBLE_RANGE = (-1.0, 32.0)


# --------------------------------------------------------------------------- data

def load_dataset(*, clean: bool = True) -> pd.DataFrame:
    """The long hourly frame, on a gapless hourly index (missing values stay NaN)."""
    df = pd.read_csv(CACHE / "long_hourly.csv", index_col=0, parse_dates=[0])
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    df = df.asfreq("1h")
    if clean:
        for col in ("eisbach", "isar"):
            bad = df[col].notna() & ~df[col].between(*PLAUSIBLE_RANGE)
            if bad.any():
                logger.info("masking %d implausible %s readings", int(bad.sum()), col)
                df.loc[bad, col] = np.nan
    return df


def observations() -> pd.Series:
    """The production observation store — what the archived forecasts were scored on."""
    frames = [pd.read_csv(p) for p in sorted((ARCHIVE / "observations").glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    s = df.set_index("timestamp")["wassertemp"].sort_index()
    return s[~s.index.duplicated(keep="last")]


def archived_forecasts(kind: str = "live") -> pd.DataFrame:
    """Archived production forecasts of the water channel."""
    frames = [pd.read_csv(p) for p in sorted((ARCHIVE / "forecasts").glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["kind"] == kind]
    for col in ("reference_time", "target_time"):
        df[col] = pd.to_datetime(df[col], utc=True)
    keep = ["reference_time", "target_time", "model_id"] + [f"wassertemp_q{q}" for q in DUET_QUANTILES]
    return df[keep].sort_values(["reference_time", "target_time"])


# ------------------------------------------------------------------------ scoring

def to_deciles(values: np.ndarray, levels) -> np.ndarray:
    """Interpolate a quantile function reported at ``levels`` onto the deciles.

    Linear in the quantile level. Every decile lies inside [0.01, 0.99], so this never
    extrapolates into a tail the model said nothing about.
    """
    levels = np.asarray(levels, dtype=float)
    return np.stack([np.interp(DECILES, levels, row) for row in values])


def crps(values: np.ndarray, truth: np.ndarray, levels=DECILES) -> np.ndarray:
    """CRPS per observation: twice the trapezoid of the pinball loss over the levels.

    Same definition as ``eisbach.verification.crps``, but with the grid passed in so
    every model is integrated over the same range.
    """
    levels = np.asarray(levels, dtype=float)
    residual = truth[:, None] - values
    pinball = np.where(residual >= 0, levels * residual, (levels - 1) * residual)
    widths = np.diff(levels)
    return 2.0 * ((pinball[:, :-1] + pinball[:, 1:]) * widths / 2.0).sum(axis=1)


def pit(values: np.ndarray, truth: np.ndarray, levels=DECILES) -> np.ndarray:
    levels = np.asarray(levels, dtype=float)
    return np.array([np.interp(y, row, levels) for y, row in zip(truth, values, strict=True)])


@dataclass
class Run:
    """One forecast of one window, with everything needed to score it."""
    label: str
    reference_time: pd.Timestamp
    target_times: pd.DatetimeIndex
    quantiles: np.ndarray          # (H, len(DECILES)) on the shared grid
    truth: np.ndarray              # (H,) NaN where the gauge never measured
    extra: dict = field(default_factory=dict)

    @property
    def median(self) -> np.ndarray:
        return self.quantiles[:, len(DECILES) // 2]


def score(run: Run, *, diurnal: np.ndarray | None = None,
          persistence: float | None = None, buckets=LEAD_BUCKETS) -> list[dict]:
    """One row per lead bucket, mirroring the production verification schema.

    ``buckets`` must be disjoint and cover the horizon, or :func:`pool` will count some
    hours twice — the same trap the production verification store documents.
    """
    leads = np.arange(1, len(run.truth) + 1)
    rows = []
    for lo, hi in buckets:
        sel = (leads > lo) & (leads <= hi) & np.isfinite(run.truth)
        if not sel.any():
            continue
        truth = run.truth[sel]
        q = run.quantiles[sel]
        row = {
            "label": run.label,
            "reference_time": run.reference_time,
            "lead_lo": lo, "lead_hi": hi,
            "n": int(sel.sum()),
            "mae": float(np.mean(np.abs(q[:, len(DECILES) // 2] - truth))),
            "rmse": float(np.sqrt(np.mean((q[:, len(DECILES) // 2] - truth) ** 2))),
            "bias": float(np.mean(q[:, len(DECILES) // 2] - truth)),
            "crps": float(np.mean(crps(q, truth))),
            "pit_mean": float(np.mean(pit(q, truth))),
            "cov_80": float(np.mean((truth >= q[:, 0]) & (truth <= q[:, -1]))),
            "width_80": float(np.mean(q[:, -1] - q[:, 0])),
        }
        if persistence is not None and np.isfinite(persistence):
            row["mae_persistence"] = float(np.mean(np.abs(persistence - truth)))
        if diurnal is not None:
            d = diurnal[sel]
            row["mae_diurnal"] = (float(np.nanmean(np.abs(d - truth)))
                                  if np.isfinite(d).any() else float("nan"))
        row.update(run.extra)
        rows.append(row)
    return rows


def pool(scores: pd.DataFrame, by=("label",)) -> pd.DataFrame:
    """Aggregate weighted by the hours each row covers; RMSE averaged in squares."""
    by = list(by)
    out = []
    for key, g in scores.groupby(by, dropna=False):
        w = g["n"].to_numpy(dtype=float)
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
        rec["n"] = int(w.sum())
        rec["runs"] = g["reference_time"].nunique()
        for col in ("mae", "bias", "crps", "pit_mean", "cov_80", "width_80",
                    "mae_persistence", "mae_diurnal"):
            if col in g:
                v = g[col].to_numpy(dtype=float)
                ok = np.isfinite(v)
                rec[col] = float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan
        v = g["rmse"].to_numpy(dtype=float)
        rec["rmse"] = float(np.sqrt((v ** 2 * w).sum() / w.sum()))
        out.append(rec)
    return pd.DataFrame(out).sort_values("mae").reset_index(drop=True)


# ---------------------------------------------------------------------- baselines

def diurnal_baseline(series: pd.Series, targets: pd.DatetimeIndex) -> np.ndarray:
    """Same hour, most recent whole day already seen. Production's honest baseline."""
    leads = np.arange(1, len(targets) + 1)
    lag = 24 * np.ceil(leads / 24)
    return series.reindex(targets - pd.to_timedelta(lag, unit="h")).to_numpy(dtype=float)


def per_run(scores: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """One value per (label, run), pooling the lead buckets weighted by their hours."""
    num = scores[metric] * scores["n"]
    grouped = scores.assign(_num=num).groupby(["label", "reference_time"])[["_num", "n"]].sum()
    return (grouped["_num"] / grouped["n"]).unstack(0)


def paired(scores: pd.DataFrame, reference: str, *, metric: str = "mae",
           draws: int = 8000, seed: int = 0) -> pd.DataFrame:
    """Every label against one reference on the windows both ran, with a bootstrap CI.

    Paired, because every combination is evaluated on the same windows and the windows
    differ from each other far more than the combinations do — an unpaired interval here
    is about ten times too wide to see anything.

    The interval is unadjusted. These sweeps test many variants against one reference, so
    with twenty candidates roughly one will clear a 5 % bar by chance; ``n_tested`` is
    carried on every row so a reader cannot forget how many there were.
    """
    values = per_run(scores, metric)
    if reference not in values:
        raise KeyError(f"{reference!r} is not among {sorted(values.columns)}")
    rng = np.random.default_rng(seed)
    base = values[reference]
    rows = []
    others = [c for c in values.columns if c != reference]
    for label in others:
        delta = (values[label] - base).dropna().to_numpy()
        if delta.size == 0:
            continue
        boot = np.array([rng.choice(delta, delta.size, replace=True).mean()
                         for _ in range(draws)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "label": label, "n_runs": delta.size,
            metric: float(values[label].mean()),
            "delta": float(delta.mean()),
            "pct": 100.0 * float(delta.mean()) / float(base.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "better_in": float((delta < 0).mean()),
            "verdict": "better" if hi < 0 else ("worse" if lo > 0 else "—"),
            "n_tested": len(others),
        })
    return pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)


def load_forecaster(checkpoint: str = "google/timesfm-3.0-pytorch"):
    """The TimesFM forecaster, on MLX where that is available and asked for.

    ``TIMESFM_BACKEND=mlx`` selects the Apple-silicon backend, which mirrors the PyTorch
    interface — ``predict`` / ``predict_batch``, univariate or multivariate, both kinds of
    covariate — and is numerically matched to it on this checkpoint to within about 2e-6
    on the quantiles. On a MacBook it is several times faster than PyTorch on CPU, which
    is the whole reason the switch exists.

    Anything else, including no setting at all, loads the PyTorch backend.
    """
    import os

    backend = os.environ.get("TIMESFM_BACKEND", "torch").lower()
    if backend == "mlx":
        from timesfm3.mlx import TimesFM3Forecaster as MlxForecaster

        logger.info("TimesFM on the MLX backend")
        return MlxForecaster.from_pretrained(checkpoint)

    from timesfm3 import TimesFM3Forecaster

    logger.info("TimesFM on the PyTorch backend")
    return TimesFM3Forecaster.from_pretrained(checkpoint)


def prepare_inputs(target: np.ndarray, past_only: np.ndarray | None,
                   past_future: np.ndarray | None, *, force: bool = False):
    """Fill missing model inputs the way the PyTorch backend does internally.

    PyTorch trims leading missing target values and linearly interpolates gaps before the
    network sees anything. MLX 3.0.2 does not: it passes NaN through, and the forecast
    comes back all NaN. Found by the local run of exp11, whose first attempt produced
    nothing but NaN quantiles.

    Off by default, and deliberately so. On PyTorch the backend already does this, and
    doing it here instead changes the numbers slightly — measured at 1.8e-5 °C hourly and
    4.6e-3 °C quarter-hourly on one window — which would silently break comparability with
    every score already recorded in ``data/experiments``. It switches on for MLX, where it
    is the difference between a forecast and a NaN, or with ``force``.

    ``target`` may be one series or a stack of them, ``(variates, time)``, since TimesFM
    3.0 forecasts several at once — exp9 hands it a stack. The trim is then over time, at
    the first step where every variate is present: slicing a stack as if it were one
    series would cut away variates rather than leading NaN, and quietly forecast the
    wrong thing.

    Never touches the arrays it is given, and never the truth used for scoring.
    """
    import os

    if not force and os.environ.get("TIMESFM_BACKEND", "torch").lower() != "mlx":
        return target, past_only, past_future

    target = np.array(target, dtype=np.float32, copy=True)
    past_only = None if past_only is None else np.array(past_only, dtype=np.float32, copy=True)
    past_future = (None if past_future is None
                   else np.array(past_future, dtype=np.float32, copy=True))

    # Validity per time step: for a stack, a step counts only if every variate has it.
    valid = ~np.isnan(target)
    if valid.ndim > 1:
        valid = valid.all(axis=0)
    if valid.any():
        first = int(np.argmax(valid))
        target = target[..., first:]
        if past_only is not None:
            past_only = past_only[:, first:]
        if past_future is not None:
            past_future = past_future[:, first:]
    else:
        target[...] = 0.0

    for arr in (target, past_only, past_future):
        if arr is None:
            continue
        for row in np.atleast_2d(arr):
            missing = np.isnan(row)
            if not missing.any():
                continue
            present = ~missing
            row[missing] = (np.interp(np.flatnonzero(missing), np.flatnonzero(present),
                                      row[present]) if present.any() else 0.0)
    return target, past_only, past_future
