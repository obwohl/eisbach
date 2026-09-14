"""Experiment 16 — how much of all this survives a real forecast?

Every covariate result in this tree is an **oracle**: the known-future covariates are the
weather that actually occurred, because archived DWD forecasts begin only in May 2026 and
historical ones are a paid product. Every report says so, and then quotes the number
anyway. This measures the gap instead of flagging it.

Two parts, because only one of them can be a real measurement.

**Part A, the replay.** For the reference times where the production archive holds a
weather snapshot issued *before* the run, the air temperature is replayed exactly as DWD
issued it, through the production lookup — so its one-sidedness comes along and a snapshot
from after the reference time is never eligible. The drop from oracle to replay is then a
real number, on a real forecast, for the covariate that matters most.

The archive also holds the forecast **global radiation** for Munich: May to July under the
old schema, and again from mid-September, with a gap in August when the field was dropped.
That gap is unrecoverable and is not a bug to fix, only a limit to state. Where solar is
present it is replayed too, which is the only honest evidence available about whether the
radiation result in exp13 is reachable in production at all.

**Part B, the extrapolation.** `t_catchment` has no archived forecast and never will for
the past. So its forecast error is *simulated*: the empirical error of the Munich air
temperature forecast, measured by lead hour from the archive in Part A, is resampled in
blocks — blocks, because a forecast that is two degrees warm at hour 30 is still warm at
hour 31, and independent noise would flatter the covariate — and injected into the
catchment temperature. This is a simulation and a substitute for evidence. It is here to
bound the risk, not to license a claim.

What this can establish: how much of a covariate's measured advantage is an artefact of
knowing the future exactly. What it cannot: that the replayed windows, all within a few
months of one season, represent the whole record.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bench  # noqa: E402
from exp4_replay import replay_airtemp  # noqa: E402
from exp6_upstream import TARGET  # noqa: E402
from exp8_catchment_weather import load_all  # noqa: E402

from eisbach import archive  # noqa: E402

logger = logging.getLogger(__name__)

CONTEXT = 8760
BATCH = 8
#: Blocks long enough to carry a forecast's bias across a whole day of the horizon.
ERROR_BLOCK_HOURS = 24
DRAWS = 20


def replay_field(reference_time: pd.Timestamp, targets: pd.DatetimeIndex,
                 candidates: tuple[str, ...]) -> np.ndarray | None:
    """A forecast field for ``targets``, as issued before ``reference_time``."""
    found = archive.load_weather_snapshot(reference_time)
    if found is None:
        return None
    snap, _anchor = found
    if "timestamp" not in snap or snap["timestamp"].isna().all():
        return None
    s = snap.copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True, errors="coerce")
    s = s.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp")
    col = next((c for c in candidates if c in s), None)
    if col is None:
        return None
    values = pd.to_numeric(s[col], errors="coerce").reindex(targets).to_numpy(dtype=float)
    return values if np.isfinite(values).all() else None


def forecast_error_by_lead(df: pd.DataFrame, anchors: list[pd.Timestamp]) -> pd.DataFrame:
    """What the DWD air-temperature forecast actually got wrong, hour by hour."""
    idx = df.index
    rows = []
    for ts in anchors:
        pos = idx.get_loc(ts)
        targets = idx[pos + 1: pos + 1 + bench.HORIZON]
        forecast = replay_airtemp(ts, targets)
        if forecast is None:
            continue
        observed = df["airtemp"].reindex(targets).to_numpy(dtype=float)
        for lead, (f, o) in enumerate(zip(forecast, observed, strict=True), start=1):
            if np.isfinite(f) and np.isfinite(o):
                rows.append({"reference_time": ts, "lead": lead, "error": f - o})
    return pd.DataFrame(rows)


def error_pool(errors: pd.DataFrame) -> dict[int, np.ndarray]:
    """Errors grouped by lead hour, to resample from."""
    return {lead: g["error"].to_numpy() for lead, g in errors.groupby("lead")}


def degrade(values: np.ndarray, pool: dict[int, np.ndarray],
            rng: np.random.Generator) -> np.ndarray:
    """Add a block-resampled forecast error to a horizon of known-future values.

    One error trace is drawn per block of ``ERROR_BLOCK_HOURS``, and within a block the
    error is taken from consecutive lead hours of one real forecast. Drawing every hour
    independently would average the error away and make the covariate look better than a
    forecast could ever make it.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    for start in range(0, len(out), ERROR_BLOCK_HOURS):
        stop = min(start + ERROR_BLOCK_HOURS, len(out))
        pick = rng.integers(0, max(len(pool.get(start + 1, [0.0])), 1))
        for lead in range(start, stop):
            samples = pool.get(lead + 1)
            if samples is None or not len(samples):
                continue
            out[lead] += samples[pick % len(samples)]
    return out


def score_variant(fc, df, truth, usable, *, label: str, future: dict[str, np.ndarray | None],
                  past_only: list[str]) -> list[dict]:
    """``future`` maps a column name to a horizon override, or None to use the observed."""
    idx = df.index
    rows = []
    items = list(usable)
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        contexts, po_list, pf_list, metas = [], [], [], []
        for ts, overrides in chunk:
            pos = idx.get_loc(ts)
            lo = pos - CONTEXT + 1
            targets = idx[pos + 1: pos + 1 + bench.HORIZON]
            contexts.append(truth.iloc[lo:pos + 1].to_numpy(dtype=np.float32))
            po_list.append(
                np.stack([df[c].iloc[lo:pos + 1].to_numpy(dtype=np.float32)
                          for c in past_only]) if past_only else None)
            stacked = []
            for col in future:
                past = df[col].iloc[lo:pos + 1].to_numpy(dtype=np.float32)
                horizon = overrides.get(col)
                if horizon is None:
                    horizon = df[col].reindex(targets).to_numpy(dtype=np.float32)
                stacked.append(np.concatenate([past, np.asarray(horizon, dtype=np.float32)]))
            pf_list.append(np.stack(stacked) if stacked else None)
            metas.append((ts, targets))
        for j in range(len(contexts)):
            contexts[j], po_list[j], pf_list[j] = bench.prepare_inputs(
                contexts[j], po_list[j], pf_list[j])
        outs = list(fc.predict_batch(contexts, horizon=bench.HORIZON,
                                     past_only_covariates=po_list,
                                     past_future_covariates=pf_list,
                                     return_quantiles=True))
        for (ts, targets), o in zip(metas, outs, strict=True):
            q = o.quantiles if o.quantiles.ndim == 2 else o.quantiles[0]
            run = bench.Run(label=label, reference_time=ts, target_times=targets,
                            quantiles=q, truth=truth.reindex(targets).to_numpy(dtype=float))
            rows.extend(bench.score(run, buckets=bench.FINE_BUCKETS))
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("eisbach.archive").setLevel(logging.WARNING)

    df = load_all()
    truth = df[TARGET]
    idx = df.index

    # Every hour the archive could possibly replay: a snapshot exists, the context fits,
    # and the horizon is observed.
    candidates = [ts for ts in idx
                  if idx.get_loc(ts) >= CONTEXT
                  and idx.get_loc(ts) + bench.HORIZON < len(idx)
                  and ts >= pd.Timestamp("2026-05-01", tz="UTC")]
    usable = []
    for ts in candidates:
        targets = idx[idx.get_loc(ts) + 1: idx.get_loc(ts) + 1 + bench.HORIZON]
        if not np.isfinite(truth.reindex(targets).to_numpy(dtype=float)).sum() >= 48:
            continue
        air = replay_airtemp(ts, targets)
        if air is None:
            continue
        solar = replay_field(ts, targets, ("solar",))
        usable.append((ts, {"airtemp": air, "_solar": solar}))
    if not usable:
        raise RuntimeError("no replayable reference times in the archive")
    with_solar = sum(1 for _, o in usable if o["_solar"] is not None)
    logger.info("%d replaybare Referenzzeiten, %s .. %s; davon %d mit Strahlungsprognose",
                len(usable), usable[0][0], usable[-1][0], with_solar)

    errors = forecast_error_by_lead(df, [ts for ts, _ in usable])
    print("\n=== Was die DWD-Lufttemperaturprognose wirklich danebenlag ===")
    by_bucket = errors.assign(
        bucket=pd.cut(errors["lead"], [0, 6, 12, 18, 24, 48, 72, 96])).groupby(
        "bucket", observed=True)["error"]
    print(pd.DataFrame({"bias": by_bucket.mean(), "MAE": by_bucket.apply(lambda e: e.abs().mean()),
                        "sd": by_bucket.std(), "n": by_bucket.size()}).to_string())

    fc = bench.load_forecaster()
    pool = error_pool(errors)
    rng = np.random.default_rng(0)

    future = {"airtemp": None, "t_catchment": None}
    past = ["isar_toelz"]
    oracle = [(ts, {}) for ts, _ in usable]
    replayed = [(ts, {"airtemp": o["airtemp"]}) for ts, o in usable]
    degraded = [(ts, {"airtemp": o["airtemp"],
                      "t_catchment": degrade(
                          df["t_catchment"].reindex(
                              idx[idx.get_loc(ts) + 1: idx.get_loc(ts) + 1 + bench.HORIZON]
                          ).to_numpy(dtype=float), pool, rng)})
                for ts, o in usable]

    rows = []
    for label, items in [("Orakel-Wetter", oracle),
                         ("echte DWD-Prognose (Luft)", replayed),
                         ("Prognose Luft + simulierter Fehler im Einzugsgebiet", degraded)]:
        t0 = time.time()
        rows += score_variant(fc, df, truth, items, label=label, future=future,
                              past_only=past)
        pd.DataFrame(rows).to_csv(bench.CACHE / "exp16_honesty.csv", index=False)
        logger.info("%-52s %.0fs", label, time.time() - t0)

    scores = pd.DataFrame(rows)
    print(f"\n=== {len(usable)} Fenster, {CONTEXT} h Kontext ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "crps"]].to_string(index=False))
    for metric in ("mae", "crps"):
        p = bench.paired(scores, "Orakel-Wetter", metric=metric)
        print(f"\n--- {metric.upper()} gegen das Orakel, gepaart ---")
        print(p[["label", metric, "pct", "ci_lo", "ci_hi", "better_in", "verdict"]]
              .to_string(index=False))
    print("\nDer Abstand zum Orakel ist der Aufschlag, den jede Zahl in diesem Baum trägt.")


if __name__ == "__main__":
    main()
