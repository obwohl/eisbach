"""Experiment 12 — does a gauge's temperature need its discharge to be worth anything?

A screen, deliberately cheap. Measured throughput with six covariates: 5.0 s per window at
a year of context, 0.42 s at 1024 hours — twelve times faster. Ranking twenty covariate
sets at a year of context costs four hours; at 1024 hours it costs half of one. So this
**ranks** at a short context and the survivors are **confirmed** at the full year, rather
than pretending one run can do both jobs.

That is a real trade, not a free lunch: exp1 measured 1024 hours of context as about 10 %
worse in absolute MAE than a year. The claim here is only that the *ordering* of covariate
sets carries over, which is the assumption the confirmation stage exists to check.

The question is the one the single-covariate scan could not answer. A temperature on its
own can look useless because the model cannot tell what it means without knowing how much
water is behind it: 50 m³/s at 11 °C and 200 m³/s at 11 °C arrive as very different things
downstream. So every gauge is tried three ways — temperature alone, discharge alone, and
the pair.

Stations are the ones the LfU's own topology puts on the path to Munich, with catchment
areas from the HND gauge pages: Mittenwald 401.7 km² (above the Krün diversion, so most of
its water leaves the Isar), Sylvenstein 1137.9, Lenggries 1295.5, Bad Tölz KW 1453.5,
Puppling 1620.3, Munich 2838.4. The Loisach joins between Puppling and Munich — Puppling's
catchment does not contain it — which is why the Loisach gauges are tested as their own
branch rather than as more of the same river.
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
from exp6_upstream import TARGET, pick_anchors  # noqa: E402
from exp8_catchment_weather import load_all  # noqa: E402

logger = logging.getLogger(__name__)

SCREEN_CONTEXT = 1024
BATCH = 16
FUTURE = ["airtemp", "t_catchment"]

#: (label, temperature, discharge) at the same gauge, ordered down the catchment.
GAUGES = [
    ("Mittenwald", "isar_mittenwald", "q_mittenwald"),
    ("Rißbachklamm", "rissbach_klamm", "q_rissbachklamm"),
    ("Lenggries", "isar_lenggries", "q_lenggries"),
    ("Bad Tölz", "isar_toelz", "q_toelz_kw"),
    ("Puppling", "isar_puppling", "q_puppling"),
    ("Loisach Beuerberg", "loisach_beuerberg", "q_loisach_beuerberg"),
]


def evaluate(fc, df, anchors, truth, *, label: str, past_only: list[str],
             context: int) -> list[dict]:
    idx = df.index
    rows = []
    for i in range(0, len(anchors), BATCH):
        chunk = anchors[i:i + BATCH]
        contexts, po_list, pf_list, metas = [], [], [], []
        for ts in chunk:
            pos = idx.get_loc(ts)
            lo = pos - context + 1
            contexts.append(truth.iloc[lo:pos + 1].to_numpy(dtype=np.float32))
            po_list.append(
                np.stack([df[c].iloc[lo:pos + 1].to_numpy(dtype=np.float32) for c in past_only])
                if past_only else None)
            pf_list.append(np.stack([
                df[c].iloc[lo:pos + 1 + bench.HORIZON].to_numpy(dtype=np.float32)
                for c in FUTURE]))
            metas.append((ts, idx[pos + 1: pos + 1 + bench.HORIZON]))
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


def report(scores: pd.DataFrame, title: str) -> pd.DataFrame:
    print(f"\n=== {title} ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "crps"]].to_string(index=False))
    out = {}
    for metric in ("mae", "crps"):
        p = bench.paired(scores, "weather only", metric=metric)
        out[metric] = p
        print(f"\n--- {metric.upper()} against weather only, paired "
              f"({int(p.n_tested.iloc[0])} variants tested) ---")
        print(p[["label", metric, "pct", "ci_lo", "ci_hi", "better_in", "verdict"]]
              .to_string(index=False))
    return out["mae"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from timesfm3 import TimesFM3Forecaster

    df = load_all()
    truth = df[TARGET]
    series = [c for _, t, q in GAUGES for c in (t, q)]
    anchors = pick_anchors(df, [*FUTURE, *series, TARGET], n=250)
    logger.info("%d anchors, %s .. %s", len(anchors), anchors[0], anchors[-1])
    logger.info("per year: %s", pd.Series(anchors).dt.year.value_counts().sort_index().to_dict())

    fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

    variants = [("weather only", [])]
    for name, t, q in GAUGES:
        variants += [(f"{name} T", [t]), (f"{name} Q", [q]), (f"{name} T+Q", [t, q])]

    rows = []
    for label, cols in variants:
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, past_only=cols,
                         context=SCREEN_CONTEXT)
        logger.info("%-22s %.0fs", label, time.time() - t0)

    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "exp12_pairs.csv", index=False)
    ranked = report(scores, f"screen at {SCREEN_CONTEXT} h of context, {len(anchors)} windows")

    print("\n--- does the discharge earn its place next to the temperature? ---")
    per = bench.per_run(scores, "mae")
    rng = np.random.default_rng(0)
    for name, _t, _q in GAUGES:
        a, b = f"{name} T+Q", f"{name} T"
        if a not in per or b not in per:
            continue
        d = (per[a] - per[b]).dropna().to_numpy()
        boot = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(8000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = "hilft" if hi < 0 else ("schadet" if lo > 0 else "—")
        print(f"   {name:<20} T+Q vs T: {d.mean():+.4f} "
              f"({100 * d.mean() / per[b].mean():+5.1f} %)  CI [{lo:+.4f},{hi:+.4f}]  {verdict}")

    print("\nShortlist for confirmation at a full year of context:")
    print("   " + ", ".join(ranked.head(4).label))


if __name__ == "__main__":
    main()
