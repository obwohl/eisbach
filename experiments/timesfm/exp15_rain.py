"""Experiment 15 — the last constructed series.

`rain_catchment_24h` is a 24-hour rolling sum of the mean of four rain gauges. Two
constructions stacked on one another, and the only ones left after `t_catchment` turned
out to be four copies of a single station rather than a mean of four.

The rain series, unlike the temperatures, really are four measurements: pairwise r
between 0.14 and 0.53 on hours when at least one of them is wet. So the mean is doing
something, and the question is whether it is doing something *useful* — or whether
handing the model the four raw series and letting it weight them does better. That is a
covariate-selection question, which is the only kind still in scope.

Three things are separated here, because the settled series confounds them:

* **the averaging** — four raw stations against their mean, both as hourly values;
* **the accumulation** — the hourly mean against its 24-hour sum;
* **both at once** — four raw stations, each as its own 24-hour sum.

If the raw stations match or beat the construction, the construction goes, and nothing
in the covariate set is hand-built any more. If the 24-hour sum is what carries, that is
worth knowing plainly rather than inheriting it from an experiment that never tested it
against its own ingredients.

Same protocol as exp12 and exp14: screen at 1024 hours of context, confirm at a year.
Everything is measured on top of the settled air temperatures and Bad Tölz, not against
bare weather — exp14 showed a branch can beat bare weather and still add nothing beside
Bad Tölz, and rain itself was the original example of a covariate that is worthless alone
and worth having in company.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from checkpoint import bind, resume, save  # noqa: E402
from exp6_upstream import TARGET, pick_anchors  # noqa: E402
from exp8_catchment_weather import SOUTH, load_all  # noqa: E402
from exp12_pairs import CONFIRM_CONTEXT, SCREEN_CONTEXT, evaluate  # noqa: E402

logger = logging.getLogger(__name__)

#: Air temperatures plus exp12's strongest gauge. The rain variants are measured on top.
BASE_FUTURE = ["airtemp", "t_catchment"]
BASE_PAST = ["isar_toelz"]

RAW = [f"rain_{s}" for s in SOUTH]
RAW_24H = [f"{c}_24h" for c in RAW]


def load() -> pd.DataFrame:
    df = load_all()
    for col in RAW:
        df[f"{col}_24h"] = df[col].rolling(24, min_periods=24).sum()
    df["rain_catchment_hourly"] = df["rain_catchment"]
    return df


def report(scores: pd.DataFrame, title: str, reference: str) -> None:
    print(f"\n=== {title} ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "crps"]].to_string(index=False))
    for metric in ("mae", "crps"):
        p = bench.paired(scores, reference, metric=metric)
        print(f"\n--- {metric.upper()} gegen '{reference}', gepaart ---")
        print(p[["label", metric, "pct", "ci_lo", "ci_hi", "better_in", "verdict"]]
              .to_string(index=False))


def sweep(fc, df, truth, anchors, variants, path, *, context: int) -> list[dict]:
    bind(path, df, anchors, context, [list(v) for v in variants])
    rows = resume(path, [label for label, _, _ in variants], anchors)
    done = {r["label"] for r in rows}
    for label, future, past_only in variants:
        if label in done:
            continue
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, past_only=past_only,
                         context=context, future=future)
        save(path, rows)
        logger.info("ctx=%d %-44s %.0fs", context, label, time.time() - t0)
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = load()
    truth = df[TARGET]
    needed = [*BASE_FUTURE, *BASE_PAST, *RAW, *RAW_24H,
              "rain_catchment_hourly", "rain_catchment_24h"]
    anchors = pick_anchors(df, [*needed, TARGET], n=250)
    if not anchors:
        raise RuntimeError("no anchors where every rain series is complete enough")
    logger.info("%d Anker, %s .. %s", len(anchors), anchors[0], anchors[-1])
    logger.info("pro Jahr: %s",
                pd.Series(anchors).dt.year.value_counts().sort_index().to_dict())

    fc = bench.load_forecaster()

    reference = "ohne Regen"
    variants = [
        (reference, BASE_FUTURE, BASE_PAST),
        # The settled construction, so the others are measured against something real.
        ("Mittel, 24h-Summe (gesetzt)", [*BASE_FUTURE, "rain_catchment_24h"], BASE_PAST),
        # Averaging, held at hourly resolution.
        ("Mittel, stündlich", [*BASE_FUTURE, "rain_catchment_hourly"], BASE_PAST),
        ("vier Rohstationen, stündlich", [*BASE_FUTURE, *RAW], BASE_PAST),
        # Accumulation, applied to the raw stations instead of to their mean.
        ("vier Rohstationen, je 24h-Summe", [*BASE_FUTURE, *RAW_24H], BASE_PAST),
        # One station, to see whether four are needed at all.
        ("nur München, 24h-Summe", [*BASE_FUTURE, "rain_muenchen_24h"], BASE_PAST),
    ]

    rows = sweep(fc, df, truth, anchors, variants, bench.CACHE / "exp15_rain.csv",
                 context=SCREEN_CONTEXT)
    report(pd.DataFrame(rows),
           f"Screen bei {SCREEN_CONTEXT} h Kontext, {len(anchors)} Fenster", reference)

    confirm = [v for v in variants if v[0] in (
        reference, "Mittel, 24h-Summe (gesetzt)", "vier Rohstationen, je 24h-Summe",
        "vier Rohstationen, stündlich")]
    confirm_rows = sweep(fc, df, truth, anchors, confirm,
                         bench.CACHE / "exp15_confirm.csv", context=CONFIRM_CONTEXT)
    report(pd.DataFrame(confirm_rows),
           f"Bestätigung bei {CONFIRM_CONTEXT} h, {len(anchors)} Fenster", reference)


if __name__ == "__main__":
    main()
