"""Experiment 6 — what is the river upstream worth?

The Eisbach branches off the Isar in Munich, so the water arriving tomorrow is water that
passed Bad Tölz today. Nothing local can know that; an upstream gauge can. This sweeps
which upstream series are worth handing to TimesFM.

Three honesty rules, all inherited:

* **Upstream gauges are past-only covariates.** We do not know tomorrow's temperature at
  Bad Tölz any more than we know tomorrow's at the Eisbach, so they are never given over
  the horizon. Air temperature is the only past-and-future covariate.
* **Air temperature over the horizon is observed, so every run here is an oracle run.**
  There are no archived DWD forecasts before 2026-05. `exp4_replay` measured the cost of
  that substitution on the windows where both exist — 0.285 against 0.290 — so it is a
  small and known bias, but it is a bias, and the label says so.
* **Every combination is scored on the same windows.** A window enters only if every
  candidate series is complete enough over it, so a combination can never look good by
  being evaluated somewhere easier.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from exp5_regimes import swing  # noqa: E402

logger = logging.getLogger(__name__)

CONTEXT = 8760
BATCH = 8
TARGET = "eisbach"

#: Upstream temperature gauges, ordered downstream.
CHAIN_T = ["isar_mittenwald", "rissbach_klamm", "isar_lenggries", "isar_toelz",
           "isar_puppling", "loisach_eschenlohe", "loisach_beuerberg", "isar_muenchen"]
#: Discharge, same ordering.
CHAIN_Q = ["q_mittenwald", "q_rissbachdueker", "q_rissbachklamm", "q_sylvenstein",
           "q_lenggries", "q_toelz_kw", "q_puppling", "q_loisach_kochel",
           "q_loisach_beuerberg", "q_isar_muenchen", "q_eisbach"]
LOCAL = ["schwabinger_bach", "q_schwabinger_bach"]


def load() -> pd.DataFrame:
    """Stations joined with the weather from the long dataset."""
    st = pd.read_csv(bench.CACHE / "stations_hourly.csv", index_col=0, parse_dates=[0])
    st.index = pd.DatetimeIndex(st.index).tz_convert("UTC")
    weather = bench.load_dataset()[["airtemp", "pressure"]]
    df = st.join(weather, how="left").asfreq("1h")
    bad = df[TARGET].notna() & ~df[TARGET].between(*bench.PLAUSIBLE_RANGE)
    if bad.any():
        logger.info("masking %d implausible %s readings", int(bad.sum()), TARGET)
        df.loc[bad, TARGET] = np.nan
    return df


def pick_anchors(df: pd.DataFrame, required: list[str], *, n: int,
                 min_complete: float = 0.95) -> list[pd.Timestamp]:
    """Anchors where every candidate series is complete enough, spread over the record.

    Stratified by calendar month so a combination cannot be judged on one season.
    """
    idx = df.index
    candidates = []
    for ts in pd.date_range(idx.min() + pd.Timedelta(hours=CONTEXT), idx.max()
                            - pd.Timedelta(hours=bench.HORIZON), freq="13h", tz="UTC"):
        if ts not in idx:
            continue
        pos = idx.get_loc(ts)
        block = df.iloc[pos - CONTEXT + 1: pos + 1 + bench.HORIZON]
        if block[required].notna().mean().min() < min_complete:
            continue
        target = df[TARGET].iloc[pos + 1: pos + 1 + bench.HORIZON]
        if target.notna().mean() < 0.9 or not np.isfinite(df[TARGET].iloc[pos]):
            continue
        # The horizon of the air covariate must be complete: it is handed over as known.
        if df["airtemp"].iloc[pos + 1: pos + 1 + bench.HORIZON].isna().any():
            continue
        candidates.append(ts)

    if not candidates:
        return []
    by_month: dict[int, list] = {}
    for ts in candidates:
        by_month.setdefault(ts.month, []).append(ts)
    per_month = max(1, n // 12)
    picked = []
    for month in sorted(by_month):
        group = by_month[month]
        step = max(1, len(group) // per_month)
        picked.extend(group[::step][:per_month])
    return sorted(picked)


def evaluate(fc, df: pd.DataFrame, anchors, truth: pd.Series, *, label: str,
             past_only: list[str], with_air: bool = True) -> list[dict]:
    idx = df.index
    rows = []
    for i in range(0, len(anchors), BATCH):
        chunk = anchors[i:i + BATCH]
        contexts, po_list, pf_list, metas = [], [], [], []
        for ts in chunk:
            pos = idx.get_loc(ts)
            lo = pos - CONTEXT + 1
            contexts.append(truth.iloc[lo:pos + 1].to_numpy(dtype=np.float32))
            po_list.append(
                np.stack([df[c].iloc[lo:pos + 1].to_numpy(dtype=np.float32) for c in past_only])
                if past_only else None)
            if with_air:
                air = df["airtemp"].iloc[lo:pos + 1 + bench.HORIZON].to_numpy(dtype=np.float32)
                pf_list.append(air[None, :])
            else:
                pf_list.append(None)
            metas.append((ts, idx[pos + 1: pos + 1 + bench.HORIZON]))
        outs = list(fc.predict_batch(contexts, horizon=bench.HORIZON,
                                     past_only_covariates=po_list,
                                     past_future_covariates=pf_list,
                                     return_quantiles=True))
        for (ts, targets), o in zip(metas, outs, strict=True):
            q = o.quantiles if o.quantiles.ndim == 2 else o.quantiles[0]
            run = bench.Run(label=label, reference_time=ts, target_times=targets,
                            quantiles=q,
                            truth=truth.reindex(targets).to_numpy(dtype=float),
                            extra={"n_covariates": len(past_only)})
            rows.extend(bench.score(run, diurnal=bench.diurnal_baseline(truth, targets),
                                    persistence=float(truth.loc[ts])))
    return rows


def report(scores: pd.DataFrame, df: pd.DataFrame, title: str) -> pd.DataFrame:
    pooled = bench.pool(scores)
    print(f"\n=== {title} ===")
    print(pooled[["label", "n", "runs", "mae", "rmse", "crps", "cov_80", "width_80"]]
          .to_string(index=False))

    swings = {ts: swing(df[TARGET], pd.Timestamp(ts)) for ts in scores.reference_time.unique()}
    s = scores.copy()
    s["abs_swing"] = s.reference_time.map(swings).abs()
    cut = s.abs_swing.quantile(0.75)
    volatile = bench.pool(s[s.abs_swing >= cut])
    print(f"\n--- the 25 % of windows that move most (|swing| >= {cut:.2f} °C) ---")
    print(volatile[["label", "runs", "mae", "crps"]].to_string(index=False))

    print("\n--- MAE by lead bucket ---")
    print(bench.pool(scores, by=["label", "lead_lo"]).pivot(
        index="label", columns="lead_lo", values="mae").round(3).to_string())
    return pooled


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from timesfm3 import TimesFM3Forecaster

    df = load()
    truth = df[TARGET]
    candidates = [c for c in CHAIN_T + CHAIN_Q + LOCAL if c in df.columns]
    logger.info("%d candidate series: %s", len(candidates), ", ".join(candidates))

    anchors = pick_anchors(df, candidates + [TARGET, "airtemp"], n=120)
    logger.info("%d anchors, %s .. %s", len(anchors),
                anchors[0] if anchors else "-", anchors[-1] if anchors else "-")
    if not anchors:
        raise RuntimeError("No window has every candidate series complete enough")

    fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

    # Stage 1 — what each series is worth on its own, on top of water + air.
    rows = []
    rows += evaluate(fc, df, anchors, truth, label="water only", past_only=[], with_air=False)
    rows += evaluate(fc, df, anchors, truth, label="+ air", past_only=[])
    for c in candidates:
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=f"+ air + {c}", past_only=[c])
        logger.info("%-34s %.0fs", c, time.time() - t0)
    single = pd.DataFrame(rows)
    single.to_csv(bench.CACHE / "exp6_single.csv", index=False)
    pooled = report(single, df, f"one upstream series at a time, {len(anchors)} windows")

    # Stage 2 — combinations worth testing on physical grounds.
    best_single = [lab.replace("+ air + ", "") for lab in pooled.label
                   if lab.startswith("+ air + ")][:3]
    logger.info("best single additions: %s", best_single)
    sets = {
        "+ air + Isar München (T)": ["isar_muenchen"],
        "+ air + Isar München (T+Q)": ["isar_muenchen", "q_isar_muenchen"],
        "+ air + Eisbach Q": ["q_eisbach"],
        "+ air + Isar München T+Q + Eisbach Q": ["isar_muenchen", "q_isar_muenchen", "q_eisbach"],
        "+ air + whole T chain": [c for c in CHAIN_T if c in df.columns],
        "+ air + whole T chain + their Q": ([c for c in CHAIN_T if c in df.columns]
                                            + [c for c in CHAIN_Q if c in df.columns]),
        "+ air + best three singly": best_single,
        "+ air + everything": candidates,
    }
    rows2 = [r for r in rows if r["label"] in ("water only", "+ air")]
    for label, cols in sets.items():
        t0 = time.time()
        rows2 += evaluate(fc, df, anchors, truth, label=label, past_only=cols)
        logger.info("%-42s %2d series  %.0fs", label, len(cols), time.time() - t0)
    combos = pd.DataFrame(rows2)
    combos.to_csv(bench.CACHE / "exp6_combinations.csv", index=False)
    report(combos, df, f"combinations, {len(anchors)} windows")


if __name__ == "__main__":
    main()
