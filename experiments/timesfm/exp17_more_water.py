"""Experiment 17 — do the other water temperatures add anything beside Bad Tölz?

Three of the nine water-temperature series beat bare weather on their own and were never
tested **next to Bad Tölz**, which is the only test that decides whether they earn a
variate slot: `isar_lenggries` (−3.9 %), `isar_puppling` (−2.3 %) and `loisach_beuerberg`
(−2.0 %). `loisach_eschenlohe` failed exactly this test — it beat bare weather and then
added +0.25 % beside Bad Tölz — so the standalone number is no guide.

Note the names. `isar_lenggries` is the **water temperature** at the Lenggries measuring
point; `q_lenggries` is the **discharge** at the same place. Every discharge is gone from
the set, so nothing here is a discharge.

**Deliberately a direction, not a verdict.** At a year of context one variant costs ten
minutes and this would run an hour; at 1024 hours it costs one. exp12 measured that the
screen ranking transfers — Spearman +0.80 over four variants, with the effects slightly
*larger* at full context, not smaller — so a short-context screen is a fair instrument for
ordering. What it cannot do is give the size of the effect at the context we actually use.

So this runs every combination at 1024 hours over 250 windows, and confirms at a year only
if something looks worth the ten minutes. The reference is Bad Tölz alone plus the settled
weather, so every number is the **incremental** value of adding a second water temperature.
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
from exp8_catchment_weather import load_all  # noqa: E402
from exp12_pairs import SCREEN_CONTEXT, evaluate  # noqa: E402

logger = logging.getLogger(__name__)

#: The settled known-future weather. Radiation is left out: it is still being confirmed,
#: and adding an unsettled covariate would confound this question with that one.
WEATHER = ["airtemp", "t_catchment"]
BEST = "isar_toelz"
CANDIDATES = ["isar_lenggries", "isar_puppling", "loisach_beuerberg"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = load_all()
    truth = df[TARGET]
    needed = [*WEATHER, BEST, *CANDIDATES]
    anchors = pick_anchors(df, [*needed, TARGET], n=250)
    if not anchors:
        raise RuntimeError("no anchors where every water temperature is complete enough")
    logger.info("%d Anker, %s .. %s", len(anchors), anchors[0], anchors[-1])
    logger.info("pro Jahr: %s",
                pd.Series(anchors).dt.year.value_counts().sort_index().to_dict())

    print("\n=== wie ähnlich sind sie Bad Tölz? ===")
    sub = df[[BEST, *CANDIDATES, TARGET]].dropna()
    for col in CANDIDATES:
        print(f"   {col:20s} r mit {BEST} = {sub[col].corr(sub[BEST]):.4f}, "
              f"mittlerer Abstand {(sub[col] - sub[BEST]).abs().mean():.3f} °C, "
              f"r mit {TARGET} = {sub[col].corr(sub[TARGET]):.4f}")

    fc = bench.load_forecaster()

    reference = "nur Bad Tölz"
    variants = [(reference, [BEST])]
    variants += [(f"+ {c}", [BEST, c]) for c in CANDIDATES]
    variants.append(("+ alle drei", [BEST, *CANDIDATES]))
    # Without Bad Tölz at all, to see whether it is the right one to build on.
    variants += [(f"{c} statt Bad Tölz", [c]) for c in CANDIDATES]

    path = bench.CACHE / "exp17_more_water.csv"
    bind(path, df, anchors, SCREEN_CONTEXT, [list(v) for v in variants])
    rows = resume(path, [label for label, _ in variants], anchors)
    done = {r["label"] for r in rows}
    for label, cols in variants:
        if label in done:
            continue
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, past_only=cols,
                         context=SCREEN_CONTEXT, future=WEATHER)
        save(path, rows)
        logger.info("%-28s %.0fs", label, time.time() - t0)

    scores = pd.DataFrame(rows)
    print(f"\n=== Screen bei {SCREEN_CONTEXT} h Kontext, {len(anchors)} Fenster ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "crps"]].to_string(index=False))
    for metric in ("mae", "crps"):
        p = bench.paired(scores, reference, metric=metric)
        print(f"\n--- {metric.upper()} gegen '{reference}', gepaart ---")
        print(p[["label", metric, "pct", "ci_lo", "ci_hi", "better_in", "verdict"]]
              .to_string(index=False))
    print("\nEine Richtung, keine Größe: bei 1024 h Kontext gemessen, nicht bei einem Jahr.")


if __name__ == "__main__":
    main()
