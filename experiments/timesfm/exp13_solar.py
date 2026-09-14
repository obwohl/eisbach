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
#: A day is the limit: Hohenpeißenberg's longest gap is four hours, so it fills entirely.
#: Garmisch's longest is 1329 hours — 55 days — which no interpolation should touch.
MAX_GAP_HOURS = 24


def load() -> pd.DataFrame:
    df = load_all()
    # The coordinate-derived solar columns go first, and by name rather than by being
    # quietly overwritten: they are the stitched ones, four of them identical and with a
    # source change in September 2020. One of them is even called solar_garmisch, which
    # is not the Garmisch station but whatever was nearest to Garmisch that month.
    stitched = [c for c in df.columns if c.startswith("solar_")]
    df = df.drop(columns=stitched)
    logger.info("dropped %d coordinate-derived solar columns: %s",
                len(stitched), ", ".join(stitched))
    solar = pd.read_csv(bench.CACHE / "solar_stations.csv", index_col=0, parse_dates=[0])
    solar.index = pd.DatetimeIndex(solar.index).tz_convert("UTC")
    df = df.join(solar, how="left")
    for col in SOLAR:
        before = df[col].isna().sum()
        df[col] = df[col].interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
        logger.info("%-26s %6d hours missing, %6d still missing after filling gaps "
                    "of up to %d h", col, before, df[col].isna().sum(), MAX_GAP_HOURS)
    df[f"{SOLAR[0]}_24h"] = df[SOLAR[0]].rolling(24, min_periods=24).sum()
    # airtemp carries one 69-hour hole of its own, and it is a known-future covariate in
    # every variant here, so it gets the same treatment rather than silently dropping a
    # year of anchors around it.
    df["airtemp"] = df["airtemp"].interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
    return df


def usable_anchors(df: pd.DataFrame, anchors: list[pd.Timestamp],
                   cols: list[str], *, context: int = CONFIRM_CONTEXT) -> list[pd.Timestamp]:
    """Anchors whose window carries every column, context and horizon alike.

    The first version of this demanded a spotless *year* of context for every column and
    left 15 anchors out of 250, all of them inside an eighteen-month window — a paired
    comparison on nothing. The cause was Garmisch's 1329-hour gap: every anchor within a
    year of it died, whether or not the variant under test used Garmisch at all.

    So gaps up to ``MAX_GAP_HOURS`` are filled first and only what survives that is
    required to be clean. The horizon is the part that truly cannot carry a NaN — a
    known-future covariate with a hole in it poisons the forecast rather than degrading
    it — but a hole in the context is not obviously handled either, since the backend's
    own interpolation is documented for the target and not for the covariates. Demanding
    both is the safe reading; the cost is only the windows near a genuine multi-day gap.
    """
    idx = df.index
    kept = []
    for ts in anchors:
        pos = idx.get_loc(ts)
        block = df[cols].iloc[pos - context + 1: pos + 1 + bench.HORIZON]
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

    df = load()
    truth = df[TARGET]
    solar_vs_air(df)

    hp, ga = SOLAR
    # Two anchor sets, because the two questions do not need the same data. Whether
    # radiation helps at all is asked of Hohenpeißenberg, which is missing eleven hours in
    # eight years; which of the two stations to use is asked only where both are present.
    main_cols = [*SETTLED, hp, f"{hp}_24h"]
    candidates = pick_anchors(df, [*main_cols, TARGET], n=250)
    anchors = usable_anchors(df, candidates, main_cols)
    logger.info("%d Anker, %s .. %s", len(anchors), anchors[0], anchors[-1])
    logger.info("pro Jahr: %s",
                pd.Series(anchors).dt.year.value_counts().sort_index().to_dict())

    station_anchors = usable_anchors(df, anchors, [*main_cols, ga])
    logger.info("%d davon auch mit Garmisch vollständig (%s .. %s)", len(station_anchors),
                station_anchors[0] if station_anchors else "-",
                station_anchors[-1] if station_anchors else "-")

    fc = bench.load_forecaster()

    #: (label, known-future covariates, past-only covariates)
    variants = [
        ("ohne Strahlung", SETTLED, []),
        ("+ Hohenpeißenberg", [*SETTLED, hp], []),
        ("+ Hohenpeißenberg, nur Vergangenheit", SETTLED, [hp]),
        # The isolation check. Rain looked worthless alone and earned its place in the
        # full set, so a covariate measured on its own is measured on the wrong question.
        ("nur Luft + Hohenpeißenberg", ["airtemp", hp], []),
        ("nur Luft", ["airtemp"], []),
        # Constructed, and flagged as such. An hour of radiation is mostly the time of
        # day — zero every night — and correlates only +0.27 with the water; the running
        # day-sum correlates +0.66. It is the same transformation `rain_catchment_24h`
        # already uses and the same objection applies to both: a uniform 24-hour window
        # is a weighting, however plain. It is here as one labelled variant so the
        # question is visible rather than decided by leaving it out.
        ("+ Hohenpeißenberg als 24h-Summe (konstruiert)", [*SETTLED, f"{hp}_24h"], []),
    ]

    rows = sweep(fc, df, truth, anchors, variants, bench.CACHE / "exp13_solar.csv",
                 context=SCREEN_CONTEXT)
    report(pd.DataFrame(rows),
           f"Screen bei {SCREEN_CONTEXT} h Kontext, {len(anchors)} Fenster")

    # Which station, on the windows where the question is answerable at all.
    if len(station_anchors) >= 50:
        station_variants = [
            ("ohne Strahlung", SETTLED, []),
            ("+ Hohenpeißenberg", [*SETTLED, hp], []),
            ("+ Garmisch", [*SETTLED, ga], []),
        ]
        st_rows = sweep(fc, df, truth, station_anchors, station_variants,
                        bench.CACHE / "exp13_stations.csv", context=SCREEN_CONTEXT)
        report(pd.DataFrame(st_rows),
               f"Stationswahl, {len(station_anchors)} Fenster mit beiden Stationen")
    else:
        logger.warning("nur %d Fenster mit beiden Stationen — Stationsvergleich "
                       "übersprungen, Hohenpeißenberg gewinnt kampflos auf Abdeckung",
                       len(station_anchors))

    # Confirmation of the main question at the context exp1 and exp11 both found best.
    confirm = [v for v in variants
               if v[0] in ("ohne Strahlung", "+ Hohenpeißenberg",
                           "+ Hohenpeißenberg als 24h-Summe (konstruiert)")]
    confirm_rows = sweep(fc, df, truth, anchors, confirm,
                         bench.CACHE / "exp13_confirm.csv", context=CONFIRM_CONTEXT)
    report(pd.DataFrame(confirm_rows),
           f"Bestätigung bei {CONFIRM_CONTEXT} h, {len(anchors)} Fenster")


def sweep(fc, df, truth, anchors, variants, path, *, context: int) -> list[dict]:
    rows = _resume(path, [label for label, _, _ in variants])
    done = {r["label"] for r in rows}
    for label, future, past_only in variants:
        if label in done:
            continue
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, past_only=past_only,
                         context=context, future=future)
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("ctx=%d %-46s %.0fs", context, label, time.time() - t0)
    return rows


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
