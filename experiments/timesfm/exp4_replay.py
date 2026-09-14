"""Experiment 4 — the honest head to head.

Experiment 3 fed TimesFM the air temperature that actually occurred. That is an oracle,
and the production PRD is explicit that an oracle backtest is not evidence of skill. This
one hands TimesFM the **DWD forecast as it was issued**, read through the production
snapshot lookup, so both models see the same imperfect weather that DUET really had.

The lookup is the production one precisely so its one-sidedness comes along: a snapshot
issued after the reference time is never eligible.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bench  # noqa: E402
from exp3_headtohead import duet_runs  # noqa: E402

from eisbach import archive  # noqa: E402

logger = logging.getLogger(__name__)

CONTEXT = 8760
BATCH = 4


def replay_airtemp(reference_time: pd.Timestamp, targets: pd.DatetimeIndex) -> np.ndarray | None:
    """The forecast air temperature for ``targets``, as DWD issued it before the run."""
    found = archive.load_weather_snapshot(reference_time)
    if found is None:
        return None
    snap, _anchor = found
    if "timestamp" not in snap or snap["timestamp"].isna().all():
        return None  # one of the seven lost snapshots
    s = snap.copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True, errors="coerce")
    s = s.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp")
    # Snapshots written before the rename carry "temperature"; newer ones carry the
    # production name.
    col = next((c for c in ("lufttemperatur_c", "temperature") if c in s), None)
    if col is None:
        return None
    values = s[col].reindex(targets).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return values


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("eisbach.archive").setLevel(logging.WARNING)
    from timesfm3 import TimesFM3Forecaster

    df = bench.load_dataset()
    truth = df["eisbach"]
    idx = df.index

    duet = duet_runs(truth)
    usable = []
    for r in duet:
        ts = r.reference_time
        if ts not in idx or idx.get_loc(ts) < CONTEXT:
            continue
        if idx.get_loc(ts) + bench.HORIZON >= len(idx):
            continue
        if np.isfinite(r.truth).sum() < 48:
            continue
        air = replay_airtemp(ts, r.target_times)
        if air is None:
            continue
        usable.append((r, air))
    logger.info("%d of %d archived runs have a replayable weather snapshot", len(usable), len(duet))

    fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

    def variant(label: str, future_air):
        """future_air(run, replayed) -> (H,) array or None for no future covariate."""
        runs = []
        for i in range(0, len(usable), BATCH):
            chunk = usable[i:i + BATCH]
            contexts, pf_list, metas = [], [], []
            for r, air in chunk:
                pos = idx.get_loc(r.reference_time)
                lo = pos - CONTEXT + 1
                contexts.append(truth.iloc[lo:pos + 1].to_numpy(dtype=np.float32))
                fut = future_air(r, air)
                if fut is None:
                    pf_list.append(None)
                else:
                    hist = df["airtemp"].iloc[lo:pos + 1].to_numpy(dtype=np.float32)
                    pf_list.append(np.concatenate([hist, fut])[None, :].astype(np.float32))
                metas.append(r)
            outs = list(fc.predict_batch(contexts, horizon=bench.HORIZON,
                                         past_future_covariates=pf_list,
                                         return_quantiles=True))
            for r, o in zip(metas, outs, strict=True):
                q = o.quantiles if o.quantiles.ndim == 2 else o.quantiles[0]
                runs.append(bench.Run(label=label, reference_time=r.reference_time,
                                      target_times=r.target_times, quantiles=q, truth=r.truth))
        return runs

    all_runs = [r for r, _ in usable]
    all_runs += variant("timesfm_air_replay", lambda r, air: air)
    logger.info("replay done")
    all_runs += variant("timesfm_air_oracle",
                        lambda r, air: df["airtemp"].reindex(r.target_times).to_numpy(dtype=np.float32))
    logger.info("oracle done")
    all_runs += variant("timesfm_uni", lambda r, air: None)
    logger.info("univariate done")

    rows = []
    for r in all_runs:
        rows.extend(bench.score(
            r, diurnal=bench.diurnal_baseline(truth, r.target_times),
            persistence=float(truth.loc[r.reference_time])
            if r.reference_time in truth.index and np.isfinite(truth.loc[r.reference_time])
            else float("nan")))
    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "exp4_replay.csv", index=False)

    print(f"\n=== honest head to head on {len(usable)} identical windows ===")
    print(bench.pool(scores).to_string(index=False))
    print("\n=== MAE by lead bucket ===")
    print(bench.pool(scores, by=["label", "lead_lo"]).pivot(
        index="label", columns="lead_lo", values="mae").round(3).to_string())
    era = scores[scores.reference_time >= "2026-08-04"]
    print(f"\n=== the PRD era only ({era.reference_time.nunique()} runs) ===")
    print(bench.pool(era).to_string(index=False))


if __name__ == "__main__":
    main()
