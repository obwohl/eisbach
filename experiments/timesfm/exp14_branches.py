"""Experiment 14 — the gauges fetched after the hydrology reading and never tested.

exp12 walked the Isar's main stem. This walks everything else the LfU's topology puts on
the path to Munich: the reservoir release, the Krün diversion itself, the Walchensee
system, the small tributaries at Bad Tölz, and the Loisach's canals.

exp12's verdict frames this. Every main-stem *temperature* below the Krün diversion beat
the weather baseline; every main-stem *discharge* was indistinguishable from zero, and
adding a discharge next to its own temperature moved nothing at any of six gauges. So the
prior on a bank of discharge series is poor, and this is mostly a check of that rather
than a hunt. Two things here are genuinely new rather than more of the same:

* ``loisach_eschenlohe`` is a **temperature**, and temperatures are what worked. It is the
  only one in the tree never measured.
* ``q_rissbachdueker`` is not a river but the **diversion**: the water being taken out of
  the Isar system at Krün and sent to the Walchensee. Every other discharge says how much
  river there is; this one says how much is being removed. That is a different question,
  and it is the one discharge with a reason to behave unlike the rest.

River names below are read off the LfU gauge pages, not inferred from the gauge names:
Peternerbrücke measures the **Jachen**, Walchen the **Walchen**, Gaißach the **Große
Gaißach**, Bad Tölz the **Ellbach**, Bruggen the **Loisach-Isar-Kanal**.

Two carry the LfU's own warnings, and both are kept as labelled variants so the exclusion
is visible and can be argued with rather than taken on trust:

* **Bruggen** (``q_loisach_isar_kanal``): *"Der Pegel wird durch Baumaßnahmen unterhalb
  beeinflusst (Rückstau). Entsprechend sind die Abflusswerte nicht korrekt."* The operator
  says the discharge is wrong. Expect nothing, and treat a gain here as a warning sign
  about the harness rather than a finding.
* **Peternerbrücke** (``q_jachen``): data transmission disturbed, last value stale as of
  this writing. The archive covers 99.7 % of hours, so the history is usable even though
  the live feed is not — which also means it could not be used in production as it stands.

Same protocol as exp12: screen at 1024 hours of context, confirm the survivors at a year.
The bar is the relaxed one — a covariate stays if its paired effect trends in our favour
and the interval does not show it doing harm.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from exp6_upstream import TARGET, pick_anchors  # noqa: E402
from exp8_catchment_weather import load_all  # noqa: E402
from exp12_pairs import CONFIRM_CONTEXT, SCREEN_CONTEXT, _resume, evaluate  # noqa: E402

logger = logging.getLogger(__name__)

#: The weather exp12 measured its gauges against, so the two experiments are comparable.
FUTURE = ["airtemp", "t_catchment"]

#: (label, columns). Ordered by what they are, not by how promising they look.
BRANCHES = [
    ("Loisach Eschenlohe T", ["loisach_eschenlohe"]),
    ("Rißbach-Düker (Ableitung)", ["q_rissbachdueker"]),
    ("Sylvensteinsee Abgabe", ["q_sylvensteinsee_ab"]),
    ("Sylvenstein Zufluss", ["q_sylvenstein"]),
    ("Jachen (Peternerbrücke)", ["q_jachen"]),
    ("Walchen", ["q_walchen"]),
    ("Große Gaißach", ["q_gaissach"]),
    ("Ellbach", ["q_ellbach"]),
    ("Loisach Kochel", ["q_loisach_kochel"]),
    ("Loisach-Beuerberg-Kanal", ["q_loisach_beuerberg_kanal"]),
    ("Loisach-Isar-Kanal (LfU: Werte nicht korrekt)", ["q_loisach_isar_kanal"]),
    # The Walchensee system as one thing: what goes in, what comes out, what was diverted.
    ("Walchensee-System", ["q_rissbachdueker", "q_walchen", "q_jachen"]),
    # The small tributaries joining around Bad Tölz.
    ("Zuflüsse bei Bad Tölz", ["q_gaissach", "q_ellbach"]),
    # Everything at once, minus the gauge whose operator says its values are wrong.
    ("alle Zweige", ["loisach_eschenlohe", "q_rissbachdueker", "q_sylvensteinsee_ab",
                     "q_jachen", "q_walchen", "q_gaissach", "q_ellbach",
                     "q_loisach_kochel"]),
]


def report(scores: pd.DataFrame, title: str) -> pd.DataFrame:
    print(f"\n=== {title} ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "crps"]].to_string(index=False))
    out = None
    for metric in ("mae", "crps"):
        p = bench.paired(scores, "weather only", metric=metric)
        out = p if out is None else out
        print(f"\n--- {metric.upper()} gegen 'weather only', gepaart ---")
        print(p[["label", metric, "pct", "ci_lo", "ci_hi", "better_in", "verdict"]]
              .to_string(index=False))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = load_all()
    truth = df[TARGET]
    series = sorted({c for _, cols in BRANCHES for c in cols})
    anchors = pick_anchors(df, [*FUTURE, *series, TARGET], n=250)
    if not anchors:
        raise RuntimeError("no anchors where every branch gauge is complete enough")
    logger.info("%d anchors, %s .. %s", len(anchors), anchors[0], anchors[-1])
    logger.info("per year: %s",
                pd.Series(anchors).dt.year.value_counts().sort_index().to_dict())

    fc = bench.load_forecaster()

    variants = [("weather only", []), *BRANCHES]
    path = bench.CACHE / "exp14_branches.csv"
    rows = _resume(path, [label for label, _ in variants])
    done = {r["label"] for r in rows}
    for label, cols in variants:
        if label in done:
            continue
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, past_only=cols,
                         context=SCREEN_CONTEXT, future=FUTURE)
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("%-46s %.0fs", label, time.time() - t0)

    scores = pd.DataFrame(rows)
    ranked = report(scores, f"Screen bei {SCREEN_CONTEXT} h Kontext, {len(anchors)} Fenster")

    # exp12's best main-stem gauge, as the yardstick a branch has to be worth adding next
    # to. A branch that only repeats Bad Tölz is not worth a variate slot.
    print("\n--- zum Vergleich: exp12 maß Bad Tölz T bei -4.2 % MAE, -4.3 % CRPS ---")

    shortlist = [lab for lab in ranked.label if lab != "weather only"][:3]
    print(f"\nBestätigung bei {CONFIRM_CONTEXT} h Kontext: {', '.join(shortlist)}")
    spec = dict(variants)
    confirm_path = bench.CACHE / "exp14_confirm.csv"
    confirm_rows = _resume(confirm_path, ["weather only", *shortlist])
    confirm_done = {r["label"] for r in confirm_rows}
    for label in ["weather only", *shortlist]:
        if label in confirm_done:
            continue
        t0 = time.time()
        confirm_rows += evaluate(fc, df, anchors, truth, label=label,
                                 past_only=spec[label], context=CONFIRM_CONTEXT,
                                 future=FUTURE)
        pd.DataFrame(confirm_rows).to_csv(confirm_path, index=False)
        logger.info("confirm %-38s %.0fs", label, time.time() - t0)

    report(pd.DataFrame(confirm_rows),
           f"Bestätigung bei {CONFIRM_CONTEXT} h, {len(anchors)} Fenster")


if __name__ == "__main__":
    main()
