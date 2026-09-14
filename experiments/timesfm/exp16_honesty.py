"""Experiment 16 — audited weather replay and forecast-error sensitivity.

The entry point delegates to run_exp16_audited: strict fetch-time eligibility,
read-only legacy parsing fixes, weather-free and Munich-only controls, actual solar
replay, provenance-bound checkpoints, and twenty repeated simulation draws.

The original 24-hour degrade() is retained unchanged for the explicit comparison
with resampling a whole 96-hour error trace. Neither is an observed southern forecast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bench  # noqa: E402

CONTEXT = 8760
BATCH = 8
#: Blocks long enough to carry a forecast's bias across a whole day of the horizon.
ERROR_BLOCK_HOURS = 24
DRAWS = 20


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
    # The audited driver retains score_variant and the original degrade() above.
    from run_exp16_audited import main as audited_main
    audited_main()


if __name__ == "__main__":
    main()
