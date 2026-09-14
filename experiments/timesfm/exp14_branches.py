"""Experiment 14 — the gauges fetched after the hydrology reading and never tested.

exp12 walked the Isar's main stem. This tests the remaining supplied reservoir,
tributary and Loisach series. Several legacy labels overstate what the gauges measure;
the station-identity corrections below are part of the experiment's result.

exp12's verdict frames this. Every main-stem *temperature* below the Krün diversion beat
the weather baseline; every main-stem *discharge* was indistinguishable from zero, and
adding a discharge next to its own temperature moved nothing at any of six gauges. So the
prior on a bank of discharge series is poor. ``loisach_eschenlohe`` is the previously
untested temperature. The original special rationale for ``q_rissbachdueker`` was
incorrect: station 16001303 is on the Isar, not a measurement of diverted water.
Likewise station 16002500 (Sylvenstein) is downstream of the reservoir, and virtual
Beuerberg 16408506 is Loisach WITH its canal, not the canal alone. Official sources
and the interpretation consequences are recorded in REPORT_exp9_exp14.md. Legacy
variant keys below are retained so cached measurements remain traceable.

River names below are read off the LfU gauge pages, not inferred from the gauge names:
Peternerbrücke measures the **Jachen**, Walchen the **Walchen**, Gaißach the **Große
Gaißach**, Bad Tölz the **Ellbach**, Bruggen the **Loisach-Isar-Kanal**.

Two carry the LfU's own warnings, and both are kept as labelled variants so the exclusion
is visible and can be argued with rather than taken on trust:

* **Bruggen** (``q_loisach_isar_kanal``): *"Der Pegel wird durch Baumaßnahmen unterhalb
  beeinflusst (Rückstau). Entsprechend sind die Abflusswerte nicht korrekt."* The operator
  says the discharge is wrong. An apparent gain cannot establish a reliable covariate;
  a biased series can still correlate with useful conditions.
* **Peternerbrücke** (``q_jachen``): the handoff reports disturbed transmission and a
  stale last value; the present live-feed status was not reverified. The archive covers
  99.7 % of hours, but historical coverage alone does not establish live reliability.

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
from checkpoint import bind, resume, save  # noqa: E402
from exp6_upstream import TARGET, pick_anchors  # noqa: E402
from exp8_catchment_weather import load_all  # noqa: E402
from exp12_pairs import CONFIRM_CONTEXT, SCREEN_CONTEXT, evaluate  # noqa: E402

logger = logging.getLogger(__name__)

#: The weather exp12 measured its gauges against, so the two experiments are comparable.
FUTURE = ["airtemp", "t_catchment"]

#: (legacy result key, columns). See the report for corrected station descriptions.
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
    # Legacy group: Isar at Rißbachdüker + Walchen + Jachen; not a measured water balance.
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
    # A known-invalid diagnostic must not choose the sample for usable candidates.
    # Its 2023 outage otherwise removes almost that entire year from every comparison.
    series = sorted({c for _, cols in BRANCHES for c in cols}
                    - {"q_loisach_isar_kanal"})
    anchors = pick_anchors(df, [*FUTURE, *series, "isar_toelz", TARGET], n=250)
    if not anchors:
        raise RuntimeError("no anchors where every branch gauge is complete enough")
    logger.info("%d anchors, %s .. %s", len(anchors), anchors[0], anchors[-1])
    logger.info("per year: %s",
                pd.Series(anchors).dt.year.value_counts().sort_index().to_dict())

    fc = bench.load_forecaster()

    variants = [("weather only", []), ("Bad Tölz T", ["isar_toelz"]), *BRANCHES]
    path = bench.CACHE / "exp14_branches.csv"
    bind(path, df, anchors, SCREEN_CONTEXT, variants)
    rows = resume(path, [label for label, _ in variants], anchors)
    done = {r["label"] for r in rows}
    for label, cols in variants:
        if label in done:
            continue
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, past_only=cols,
                         context=SCREEN_CONTEXT, future=FUTURE)
        save(path, rows)
        logger.info("%-46s %.0fs", label, time.time() - t0)

    scores = pd.DataFrame(rows)
    ranked = report(scores, f"Screen bei {SCREEN_CONTEXT} h Kontext, {len(anchors)} Fenster")

    # exp12's best main-stem gauge, as the yardstick a branch has to be worth adding next
    # to. A branch that only repeats Bad Tölz is not worth a variate slot.
    print("\n--- zum Vergleich: exp12 maß Bad Tölz T bei -4.2 % MAE, -4.3 % CRPS ---")

    eligible = {label for label, _ in BRANCHES if "LfU:" not in label}
    shortlist = [lab for lab in ranked.label if lab in eligible][:3]
    print(f"\nBestätigung bei {CONFIRM_CONTEXT} h Kontext: {', '.join(shortlist)}")
    spec = dict(variants)
    confirm_path = bench.CACHE / "exp14_confirm.csv"
    # Measure incremental value on exactly the same windows as Bad Tölz.
    confirm_labels = ["weather only", "Bad Tölz T", *shortlist]
    for label in shortlist:
        combined = f"Bad Tölz T + {label}"
        spec[combined] = ["isar_toelz", *spec[label]]
        confirm_labels.append(combined)
    bind(confirm_path, df, anchors, CONFIRM_CONTEXT,
         [(label, spec[label]) for label in confirm_labels])
    confirm_rows = resume(confirm_path, confirm_labels, anchors)
    confirm_done = {r["label"] for r in confirm_rows}
    for label in confirm_labels:
        if label in confirm_done:
            continue
        t0 = time.time()
        confirm_rows += evaluate(fc, df, anchors, truth, label=label,
                                 past_only=spec[label], context=CONFIRM_CONTEXT,
                                 future=FUTURE)
        save(confirm_path, confirm_rows)
        logger.info("confirm %-38s %.0fs", label, time.time() - t0)

    report(pd.DataFrame(confirm_rows),
           f"Bestätigung bei {CONFIRM_CONTEXT} h, {len(anchors)} Fenster")


if __name__ == "__main__":
    main()
