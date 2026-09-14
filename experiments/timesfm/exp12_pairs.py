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
#: Where the screen's survivors are re-measured. exp1 found this the best hourly context.
CONFIRM_CONTEXT = 8760
BATCH = 8
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
             context: int, future: list[str] | None = None) -> list[dict]:
    future = FUTURE if future is None else future
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
                for c in future]) if future else None)
            metas.append((ts, idx[pos + 1: pos + 1 + bench.HORIZON]))
        # Per window, not per batch. A no-op on PyTorch, which does this internally; on
        # MLX it is the difference between a forecast and a column of NaN.
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


def _resume(path: Path, wanted: list[str]) -> list[dict]:
    """Rows for variants a previous run finished, so a restart does not redo them.

    A variant is only taken from the checkpoint if it is one we still want *and* it is
    not the last label in the file: the run may have been killed mid-variant, and a
    half-scored variant is worse than no variant at all.
    """
    if not path.exists():
        return []
    prev = pd.read_csv(path, parse_dates=["reference_time"])
    if prev.empty:
        return []
    order = list(dict.fromkeys(prev["label"]))
    complete = [lab for lab in order[:-1] if lab in wanted]
    kept = prev[prev["label"].isin(complete)]
    if complete:
        logger.info("resuming %s: %d variants already done (%s)",
                    path.name, len(complete), ", ".join(complete))
    return kept.to_dict("records")


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

    # Checkpoint after every variant, and resume from it. The first attempt was killed
    # silently after six of nineteen and lost all of them, because results were only
    # written at the end; the second lost eight more to a restart, because writing a
    # checkpoint nobody reads back only records the loss.
    path = bench.CACHE / "exp12_pairs.csv"
    rows = _resume(path, [label for label, _ in variants])
    done = {r["label"] for r in rows}
    for label, cols in variants:
        if label in done:
            continue
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, past_only=cols,
                         context=SCREEN_CONTEXT)
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("%-22s %.0fs", label, time.time() - t0)

    scores = pd.DataFrame(rows)
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

    # Confirmation, and a test of the screen itself. exp11 measured a year of context as
    # worth 16 % MAE over 160 days, independently of exp1 — so context dominates, and the
    # assumption that a short-context ranking survives at full context is worth checking
    # rather than asserting.
    shortlist = [lab for lab in ranked.label if lab != "weather only"][:4]
    print(f"\nConfirming at {CONFIRM_CONTEXT} h of context: {', '.join(shortlist)}")
    confirm_path = bench.CACHE / "exp12_confirm.csv"
    confirm_rows = _resume(confirm_path, ["weather only", *shortlist])
    confirm_done = {r["label"] for r in confirm_rows}
    for label in ["weather only", *shortlist]:
        if label in confirm_done:
            continue
        cols = dict(variants)[label]
        t0 = time.time()
        confirm_rows += evaluate(fc, df, anchors, truth, label=label, past_only=cols,
                                 context=CONFIRM_CONTEXT)
        pd.DataFrame(confirm_rows).to_csv(confirm_path, index=False)
        logger.info("confirm %-22s %.0fs", label, time.time() - t0)

    confirmed = pd.DataFrame(confirm_rows)
    report(confirmed, f"confirmation at {CONFIRM_CONTEXT} h, {len(anchors)} windows")

    print("\n--- did the ranking survive the change of context? ---")
    a = bench.paired(scores, "weather only").set_index("label")["pct"]
    b = bench.paired(confirmed, "weather only").set_index("label")["pct"]
    common = [c for c in shortlist if c in b.index]
    print(f"{'Variante':<22}{'Screen %':>11}{'Voll %':>10}")
    for lab in common:
        print(f"{lab:<22}{a[lab]:>+10.1f}%{b[lab]:>+9.1f}%")
    if len(common) > 2:
        rank_screen = a[common].rank()
        rank_full = b[common].rank()
        rho = rank_screen.corr(rank_full, method="spearman")
        print(f"\n   Spearman der Rangfolge über {len(common)} Varianten: {rho:+.2f}")
        print("   (bei so wenigen Varianten grob; ein negativer Wert widerlegt das "
              "Screen-Protokoll, ein positiver bestätigt es nicht.)")


if __name__ == "__main__":
    main()
