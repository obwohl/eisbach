"""Experiment 10 — two covariates the hydrology says should exist.

Written after reading how the Isar actually reaches Munich, which turned out to explain
the earlier results rather than merely decorate them.

**The water splits at Krün.** Up to 25 m³/s of the upper Isar and up to 12 m³/s of the
Rißbach are diverted into the Walchensee; below Krün only 3 to 4.8 m³/s of residual flow
stays in the bed. The diverted water crosses the Walchensee, the power station and the
Kochelsee, enters the Loisach, and rejoins the Isar at Wolfratshausen — **below** the
Puppling gauge. So Munich's water arrives on two branches, and the gauges see one each:

* the direct branch, regulated by the Sylvenstein reservoir: Lenggries, Bad Tölz, Puppling;
* the detour, buffered by two large lakes: Loisach at Eschenlohe and Beuerberg.

Measured here, the Loisach carries a **median 55 %** of what reaches Munich — more than
the Isar at Puppling does. That is why Mittenwald, which sits above Krün and whose water
mostly leaves the Isar, was the worst temperature covariate of all, and why no single
gauge is the upstream signal.

So: `t_mix`, the discharge-weighted temperature of the two branches — the temperature of
the water actually entering Munich. It correlates 0.9929 with the Eisbach against 0.9740
and 0.9905 for its parts. The mass balance closes exactly: Puppling plus Beuerberg is
66 m³/s median, and Munich measures 41 in the Isar plus 25 in the Eisbach.

**And solar radiation.** Air temperature is a proxy for the energy that actually heats the
water; Bright Sky serves global radiation at any coordinate, 100 % covered historically
and in the forecast, so it is a known-future covariate like air temperature. The Eisbach's
open reach takes roughly 15 MW of sun on a summer afternoon, so this is not a subtle term.
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
from exp8_catchment_weather import evaluate, load_all  # noqa: E402

logger = logging.getLogger(__name__)


def add_hydrology(df: pd.DataFrame) -> pd.DataFrame:
    """The mixed inflow, and the discharge share it is mixed with."""
    q_i, t_i = df["q_puppling"], df["isar_puppling"]
    q_l, t_l = df["q_loisach_beuerberg"], df["loisach_beuerberg"]
    total = q_i + q_l
    df["t_mix"] = (q_i * t_i + q_l * t_l) / total
    df["q_mix"] = total
    df["loisach_share"] = q_l / total
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from timesfm3 import TimesFM3Forecaster

    df = add_hydrology(load_all())
    truth = df[TARGET]
    chain = ["isar_lenggries", "isar_toelz", "q_lenggries", "q_toelz_kw"]
    needed = ["airtemp", "t_catchment", "t_mix", "q_mix", *chain, TARGET]
    anchors = pick_anchors(df, needed, n=150)
    logger.info("%d anchors, %s .. %s", len(anchors), anchors[0], anchors[-1])

    fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")
    base_future = ["airtemp", "t_catchment"]

    variants = [
        ("air + catchment air (reference)", base_future, []),
        ("+ upstream chain", base_future, chain),
        ("+ mixed inflow T", base_future, ["t_mix"]),
        ("+ mixed inflow T+Q", base_future, ["t_mix", "q_mix"]),
        ("+ mixed inflow T+Q + chain", base_future, ["t_mix", "q_mix", *chain]),
        ("+ both branches separately", base_future,
         ["isar_puppling", "q_puppling", "loisach_beuerberg", "q_loisach_beuerberg"]),
        ("+ solar München", [*base_future, "solar_muenchen"], []),
        ("+ solar München + mixed inflow", [*base_future, "solar_muenchen"], ["t_mix", "q_mix"]),
        ("everything", [*base_future, "solar_muenchen"], ["t_mix", "q_mix", *chain]),
    ]

    rows = []
    for label, future, past_only in variants:
        if any(c not in df.columns for c in future + past_only):
            logger.warning("skipping %s: missing %s", label,
                           [c for c in future + past_only if c not in df.columns])
            continue
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, future=future, past_only=past_only)
        logger.info("%-36s %.0fs", label, time.time() - t0)

    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "exp10_hydrology.csv", index=False)
    print(f"\n=== hydrology-led covariates, {len(anchors)} windows ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "crps", "cov_80", "width_80"]]
          .to_string(index=False))
    for metric in ("mae", "crps"):
        print(f"\n--- {metric.upper()} against the reference, paired ---")
        p = bench.paired(scores, "air + catchment air (reference)", metric=metric)
        print(p[["label", metric, "pct", "ci_lo", "ci_hi", "better_in", "verdict"]]
              .to_string(index=False))
    print("\n--- MAE by lead bucket ---")
    print(bench.pool(scores, by=["label", "lead_lo"]).pivot(
        index="label", columns="lead_lo", values="mae").round(3).to_string())


if __name__ == "__main__":
    main()
