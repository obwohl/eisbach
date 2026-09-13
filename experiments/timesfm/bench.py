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
          persistence: float | None = None) -> list[dict]:
    """One row per lead bucket, mirroring the production verification schema."""
    leads = np.arange(1, len(run.truth) + 1)
    rows = []
    for lo, hi in LEAD_BUCKETS:
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
