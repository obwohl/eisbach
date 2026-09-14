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
import pathlib
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

BATCH = 8

#: Screen cheaply, confirm at the context exp1 and exp11 both found best. The first
#: attempt at this experiment ran every variant at 8760 h and was still on its baseline
#: after an hour.
SCREEN_CONTEXT = 1024
CONFIRM_CONTEXT = 8760

#: What exp12 left standing. It walked six gauges and found every main-stem *temperature*
#: below the Krün diversion beating the weather baseline, every *discharge* alone
#: indistinguishable from zero, and the discharge next to its own temperature worth
#: nothing at any of the six — six intervals, six containing zero. So the discharges are
#: gone from this design, and the question is asked of the series that actually carry.
UPSTREAM = ["isar_toelz", "isar_lenggries"]
WEATHER = ["airtemp", "t_catchment"]


def evaluate(fc, df, anchors, truth, *, label: str, targets: list[str],
             past_only: list[str], future: list[str], context: int) -> list[dict]:
    """``targets`` are forecast alongside the Eisbach; only the Eisbach is scored."""
    idx = df.index
    rows = []
    for i in range(0, len(anchors), BATCH):
        chunk = anchors[i:i + BATCH]
        contexts, po_list, pf_list, metas = [], [], [], []
        for ts in chunk:
            pos = idx.get_loc(ts)
            lo = pos - context + 1
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

    df = load_all()
    truth = df[TARGET]
    anchors = pick_anchors(df, WEATHER + UPSTREAM + [TARGET], n=150)
    logger.info("%d anchors, %s .. %s", len(anchors), anchors[0], anchors[-1])

    fc = bench.load_forecaster()

    best = ["isar_toelz"]  # exp12's strongest single gauge: -4.2 % MAE, -4.3 % CRPS.
    variants = [
        ("baseline: air only", [], [], ["airtemp"]),
        # The reference both framings are measured against.
        ("upstream as covariates", [], UPSTREAM, WEATHER),
        ("upstream as targets", UPSTREAM, [], WEATHER),
        # One gauge, both ways. Two targets may simply cost more capacity than one gauge
        # is worth, which would confound the framing question with a count question.
        ("Bad Tölz only, as covariate", [], best, WEATHER),
        ("Bad Tölz only, as target", best, [], WEATHER),
        # Both at once: forecast alongside *and* handed over. TimesFM allows it, and if
        # the two paths carry different things it should beat either alone.
        ("upstream as targets and covariates", UPSTREAM, UPSTREAM, WEATHER),
    ]

    def sweep(context: int, subset, path: pathlib.Path) -> pd.DataFrame:
        rows = []
        for label, targets, past_only, future in subset:
            t0 = time.time()
            rows += evaluate(fc, df, anchors, truth, label=label, targets=targets,
                             past_only=past_only, future=future, context=context)
            pd.DataFrame(rows).to_csv(path, index=False)
            logger.info("ctx=%d %-38s %d targets  %.0fs",
                        context, label, 1 + len(targets), time.time() - t0)
        return pd.DataFrame(rows)

    def show(scores: pd.DataFrame, title: str) -> pd.DataFrame:
        print(f"\n=== {title} ===")
        print(bench.pool(scores)[["label", "n", "runs", "mae", "crps", "cov_80", "width_80"]]
              .to_string(index=False))
        out = None
        for metric in ("mae", "crps"):
            p = bench.paired(scores, "upstream as covariates", metric=metric)
            out = out if out is not None else p
            print(f"\n--- {metric.upper()} against 'upstream as covariates', paired ---")
            print(p[["label", metric, "pct", "ci_lo", "ci_hi", "better_in", "verdict"]]
                  .to_string(index=False))
        print("\n--- MAE by lead bucket ---")
        print(bench.pool(scores, by=["label", "lead_lo"]).pivot(
            index="label", columns="lead_lo", values="mae").round(3).to_string())
        return out

    screened = sweep(SCREEN_CONTEXT, variants, bench.CACHE / "exp9_targets.csv")
    ranked = show(screened, f"screen at {SCREEN_CONTEXT} h, {len(anchors)} windows")

    keep = ["upstream as covariates"] + [lab for lab in ranked.label
                                         if lab != "baseline: air only"][:2]
    subset = [v for v in variants if v[0] in keep]
    confirmed = sweep(CONFIRM_CONTEXT, subset, bench.CACHE / "exp9_confirm.csv")
    show(confirmed, f"confirmation at {CONFIRM_CONTEXT} h, {len(anchors)} windows")


if __name__ == "__main__":
    main()
