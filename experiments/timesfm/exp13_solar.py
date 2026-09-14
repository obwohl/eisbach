"""Experiment 13 — is global radiation worth a covariate slot?

Radiation is the energy that actually warms the water. Air temperature is its proxy, and
a lossy one: a clear March day and an overcast May day can sit at the same 12 °C and
deliver very different heat to the surface of a river. So the question is not whether
radiation matters physically — it plainly does — but whether the model can still read
something out of it that the air temperature has not already told it.

**One station, chosen by name.** The DWD measures global radiation at far fewer sites than
it measures temperature: of the stations within reach of the Isar's headwaters, Mittenwald,
Jachenau-Tannern and Holzkirchen carry it in under half a percent of hours, and Kreuth —
the nearest one to Lenggries, and the one a coordinate lookup silently reaches for after
September 2020 — in 59 %, with 37 of its 73 months below nine tenths. That leaves two, and
`build_solar.py` fetches both by DWD id so no seam can creep in:

* ``hohenpeissenberg`` 977 m, unbroken in 93 of 93 months, but on the Ammer watershed —
  its own water reaches the Isar at Moosburg, *below* Munich. Not in the catchment.
* ``garmisch`` 719 m, 97.3 %, in the Loisach valley, which does feed the Isar above Munich.

Reliability points one way and catchment membership the other, so both are measured and
the data decides. Only the winner is kept: this adds **one** column, not a bank of them.

Radiation is offered as a known-future covariate, which it honestly is — Bright Sky serves
`solar` from MOSMIX for future hours, so the production pipeline could ask for it. The
past-only variant is here to check whether the future half is where the value sits.

Two caveats worth carrying into any conclusion. Like every experiment in this tree, the
future covariates are the weather that *actually occurred*, not the forecast that would
have been available — the numbers are an upper bound. And the forecast that would serve
this in production comes from MOSMIX, not from Hohenpeißenberg's pyranometer, so train and
serve would not see quite the same instrument.

The bar is the relaxed one: a covariate stays if its paired effect trends in our favour
and the interval does not show it doing harm. Significance is a bonus, not a gate.
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
from exp12_pairs import SCREEN_CONTEXT, _resume, evaluate  # noqa: E402

logger = logging.getLogger(__name__)

CONFIRM_CONTEXT = 8760

#: What the covariate search has settled on so far, all of it known-future.
SETTLED = ["airtemp", "t_catchment", "rain_catchment_24h"]

SOLAR = ["solar_hohenpeissenberg", "solar_garmisch"]

#: Radiation gaps are short and the value between two measured hours is not in doubt, but
#: a NaN handed to the model over the horizon is. Anything longer than this is left as a
#: gap and the anchor filter drops the window instead of inventing a day of sunshine.
MAX_GAP_HOURS = 3


def load() -> pd.DataFrame:
    df = load_all()
    solar = pd.read_csv(bench.CACHE / "solar_stations.csv", index_col=0, parse_dates=[0])
    solar.index = pd.DatetimeIndex(solar.index).tz_convert("UTC")
    df = df.join(solar, how="left")
    for col in SOLAR:
        before = df[col].isna().sum()
        df[col] = df[col].interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
        logger.info("%-26s %6d hours missing, %6d still missing after filling gaps "
                    "of up to %d h", col, before, df[col].isna().sum(), MAX_GAP_HOURS)
    return df


def complete_anchors(df: pd.DataFrame, anchors: list[pd.Timestamp],
                     cols: list[str]) -> list[pd.Timestamp]:
    """Anchors whose whole window — context and horizon — carries every column.

    ``pick_anchors`` tolerates 5 % missing, which is right for a past-only covariate and
    wrong for one handed over as known: a single NaN in the horizon poisons the forecast
    rather than degrading it.
    """
    idx = df.index
    kept = []
    for ts in anchors:
        pos = idx.get_loc(ts)
        block = df[cols].iloc[pos - CONFIRM_CONTEXT + 1: pos + 1 + bench.HORIZON]
        if not block.isna().to_numpy().any():
            kept.append(ts)
    return kept


def solar_vs_air(df: pd.DataFrame) -> None:
    """What radiation might add that the air temperature has not already said."""
    print("\n=== was die Strahlung überhaupt Neues sagen kann ===")
    sub = df[[*SETTLED, *SOLAR]].dropna()
    print(f"{len(sub)} gemeinsame Stunden")
    for col in SOLAR:
        r_air = sub[col].corr(sub["airtemp"])
        r_catch = sub[col].corr(sub["t_catchment"])
        # What is left of the radiation once both air temperatures are regressed out.
        x = np.column_stack([np.ones(len(sub)), sub["airtemp"], sub["t_catchment"]])
        beta, *_ = np.linalg.lstsq(x, sub[col].to_numpy(), rcond=None)
        resid = sub[col].to_numpy() - x @ beta
        share = resid.var() / sub[col].var()
        print(f"   {col:26s} r(airtemp)={r_air:+.3f}  r(t_catchment)={r_catch:+.3f}  "
              f"unabhängiger Anteil {share:.0%}")
    a, b = SOLAR
    print(f"   {a[6:]} vs {b[6:]}: r={sub[a].corr(sub[b]):+.4f}, "
          f"mittlerer Abstand {np.abs(sub[a] - sub[b]).mean():.4f} kWh/m²")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from timesfm3 import TimesFM3Forecaster

    df = load()
    truth = df[TARGET]
    solar_vs_air(df)

    anchors = pick_anchors(df, [*SETTLED, *SOLAR, TARGET], n=250)
    anchors = complete_anchors(df, anchors, [*SETTLED, *SOLAR])
    logger.info("%d anchors, %s .. %s", len(anchors), anchors[0], anchors[-1])
    logger.info("per year: %s", pd.Series(anchors).dt.year.value_counts().sort_index().to_dict())

    fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

    hp, ga = SOLAR
    #: (label, known-future covariates, past-only covariates)
    variants = [
        ("ohne Strahlung", SETTLED, []),
        ("+ Hohenpeißenberg", [*SETTLED, hp], []),
        ("+ Garmisch", [*SETTLED, ga], []),
        ("+ Hohenpeißenberg, nur Vergangenheit", SETTLED, [hp]),
        # The isolation check. Rain looked worthless alone and earned its place in the
        # full set, so a covariate measured on its own is measured on the wrong question.
        ("nur Luft + Hohenpeißenberg", ["airtemp", hp], []),
        ("nur Luft", ["airtemp"], []),
    ]

    path = bench.CACHE / "exp13_solar.csv"
    rows = _resume(path, [label for label, _, _ in variants])
    done = {r["label"] for r in rows}
    for label, future, past_only in variants:
        if label in done:
            continue
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, past_only=past_only,
                         context=SCREEN_CONTEXT, future=future)
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("%-38s %.0fs", label, time.time() - t0)

    scores = pd.DataFrame(rows)
    report(scores, f"Screen bei {SCREEN_CONTEXT} h Kontext, {len(anchors)} Fenster")

    shortlist = ["+ Hohenpeißenberg", "+ Garmisch"]
    confirm_path = bench.CACHE / "exp13_confirm.csv"
    confirm_rows = _resume(confirm_path, ["ohne Strahlung", *shortlist])
    confirm_done = {r["label"] for r in confirm_rows}
    spec = {label: (future, past_only) for label, future, past_only in variants}
    for label in ["ohne Strahlung", *shortlist]:
        if label in confirm_done:
            continue
        future, past_only = spec[label]
        t0 = time.time()
        confirm_rows += evaluate(fc, df, anchors, truth, label=label, past_only=past_only,
                                 context=CONFIRM_CONTEXT, future=future)
        pd.DataFrame(confirm_rows).to_csv(confirm_path, index=False)
        logger.info("confirm %-30s %.0fs", label, time.time() - t0)

    report(pd.DataFrame(confirm_rows),
           f"Bestätigung bei {CONFIRM_CONTEXT} h, {len(anchors)} Fenster")


def report(scores: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "crps"]].to_string(index=False))
    for metric in ("mae", "crps"):
        p = bench.paired(scores, "ohne Strahlung", metric=metric)
        print(f"\n--- {metric.upper()} gegen 'ohne Strahlung', gepaart ---")
        print(p[["label", metric, "pct", "ci_lo", "ci_hi", "better_in", "verdict"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
