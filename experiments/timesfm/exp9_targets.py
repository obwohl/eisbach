"""Experiment 9 — covariate or target? They are not the same thing.

Everything so far handed the upstream gauges to TimesFM as **covariates**: the Eisbach
was the only series it was asked to forecast. TimesFM 3.0 is natively multivariate, so
the same gauges can instead be additional **targets** — forecast alongside the Eisbach,
with variate attention running between them.

Two guesses point opposite ways, which is why this is worth measuring rather than
arguing. As targets the gauges get the full attention path rather than the covariate
path, which might carry more. But they also spend model capacity and horizon patches on
series nobody asked about, which in a zero-shot model could pull focus off the Eisbach.

The Eisbach is always variate 0, and only its forecast is scored.
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
from exp6_upstream import CONTEXT, TARGET, pick_anchors  # noqa: E402
from exp8_catchment_weather import load_all  # noqa: E402

logger = logging.getLogger(__name__)

BATCH = 2

#: What experiments 6 to 8 left standing, in the order they earned their place.
UPSTREAM = ["isar_lenggries", "isar_toelz", "q_lenggries", "q_toelz_kw"]
WEATHER = ["airtemp", "t_catchment"]


def evaluate(fc, df, anchors, truth, *, label: str, targets: list[str],
             past_only: list[str], future: list[str]) -> list[dict]:
    """``targets`` are forecast alongside the Eisbach; only the Eisbach is scored."""
    idx = df.index
    rows = []
    for i in range(0, len(anchors), BATCH):
        chunk = anchors[i:i + BATCH]
        contexts, po_list, pf_list, metas = [], [], [], []
        for ts in chunk:
            pos = idx.get_loc(ts)
            lo = pos - CONTEXT + 1
            block = [truth.iloc[lo:pos + 1].to_numpy(dtype=np.float32)]
            block += [df[c].iloc[lo:pos + 1].to_numpy(dtype=np.float32) for c in targets]
            contexts.append(np.stack(block) if targets else block[0])
            po_list.append(
                np.stack([df[c].iloc[lo:pos + 1].to_numpy(dtype=np.float32) for c in past_only])
                if past_only else None)
            pf_list.append(
                np.stack([df[c].iloc[lo:pos + 1 + bench.HORIZON].to_numpy(dtype=np.float32)
                          for c in future]) if future else None)
            metas.append((ts, idx[pos + 1: pos + 1 + bench.HORIZON]))
        outs = list(fc.predict_batch(contexts, horizon=bench.HORIZON,
                                     past_only_covariates=po_list,
                                     past_future_covariates=pf_list,
                                     return_quantiles=True))
        for (ts, target_times), o in zip(metas, outs, strict=True):
            q = o.quantiles
            # (V, H, Q) when there is more than one target; the Eisbach is variate 0.
            q = q[0] if q.ndim == 3 else q
            run = bench.Run(label=label, reference_time=ts, target_times=target_times,
                            quantiles=q,
                            truth=truth.reindex(target_times).to_numpy(dtype=float))
            rows.extend(bench.score(run, buckets=bench.FINE_BUCKETS))
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from timesfm3 import TimesFM3Forecaster

    df = load_all()
    truth = df[TARGET]
    anchors = pick_anchors(df, WEATHER + UPSTREAM + [TARGET], n=150)
    logger.info("%d anchors, %s .. %s", len(anchors), anchors[0], anchors[-1])

    fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

    temps = ["isar_lenggries", "isar_toelz"]
    flows = ["q_lenggries", "q_toelz_kw"]
    variants = [
        ("baseline: air only", [], [], ["airtemp"]),
        ("upstream as covariates", [], UPSTREAM, WEATHER),
        ("upstream temps as targets", temps, flows, WEATHER),
        ("all four upstream as targets", UPSTREAM, [], WEATHER),
        ("temps as targets, none as covariates", temps, [], WEATHER),
    ]

    rows = []
    for label, targets, past_only, future in variants:
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, targets=targets,
                         past_only=past_only, future=future)
        logger.info("%-38s %d targets, %d past-only  %.0fs",
                    label, 1 + len(targets), len(past_only), time.time() - t0)

    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "exp9_targets.csv", index=False)

    print(f"\n=== covariate or target, {len(anchors)} windows ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "crps", "cov_80", "width_80"]]
          .to_string(index=False))
    for metric in ("mae", "crps"):
        print(f"\n--- {metric.upper()} against 'upstream as covariates', paired ---")
        p = bench.paired(scores, "upstream as covariates", metric=metric)
        print(p[["label", metric, "pct", "ci_lo", "ci_hi", "better_in", "verdict"]]
              .to_string(index=False))
    print("\n--- MAE by lead bucket ---")
    print(bench.pool(scores, by=["label", "lead_lo"]).pivot(
        index="label", columns="lead_lo", values="mae").round(3).to_string())


if __name__ == "__main__":
    main()
