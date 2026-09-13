"""Does the weather forecast make the live model *better*, or only make it move?

The probe next door shows the forecast moves when the DWD forecast changes. Moving is
not the same as being right. This scores the same windows against what actually happened,
with the real weather forecast and with a wrong one — the same calendar hours from a week
earlier, which is a plausible-looking forecast for the wrong days.

If the real weather is doing work, the real one scores better. If the model is only being
jostled by an input it cannot interpret, the two score the same.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "timesfm"))

import bench  # noqa: E402
from probe import windows  # noqa: E402

from eisbach.data import COVARIATE_SHIFT_HOURS  # noqa: E402
from eisbach.model import QUANTILES, forecast, load_model  # noqa: E402

logger = logging.getLogger(__name__)

WATER_COLUMNS = [f"wassertemp_q{q}" for q in QUANTILES]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = bench.load_dataset()
    truth = df["eisbach"]
    blocks = windows(df, n=60, every_hours=61)
    logger.info("%d windows", len(blocks))

    model, config = load_model()
    air_shifted = df["airtemp"].shift(-COVARIATE_SHIFT_HOURS)

    rows = []
    for block in blocks:
        anchor = block.index[-1]
        targets = pd.date_range(anchor + pd.Timedelta(hours=1), periods=96, freq="1h", tz="UTC")
        y = truth.reindex(targets).to_numpy(dtype=float)
        if np.isfinite(y).sum() < 48:
            continue

        # The wrong forecast: the same hours, one week earlier.
        wrong = air_shifted.reindex(block.index - pd.Timedelta(days=7)).to_numpy()
        if not np.isfinite(wrong).all():
            continue

        for label, mutate in (
            ("real DWD forecast", None),
            ("weather from a week earlier",
             lambda b, w=wrong: b.__setitem__("airtemp_96", w)),
            ("air held at the window mean",
             lambda b: b.__setitem__("airtemp_96", float(b["airtemp_96"].mean()))),
        ):
            b = block.copy()
            if mutate is not None:
                mutate(b)
            fc = forecast(model, config, b)[WATER_COLUMNS].to_numpy()
            run = bench.Run(label=label, reference_time=anchor, target_times=targets,
                            quantiles=bench.to_deciles(fc, QUANTILES), truth=y)
            rows.extend(bench.score(run, diurnal=bench.diurnal_baseline(truth, targets)))

    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "duet_usefulness.csv", index=False)
    print(f"\n=== DUET on {scores.reference_time.nunique()} windows, same truth, decile grid ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "rmse", "crps", "bias",
                              "mae_diurnal"]].to_string(index=False))
    print("\n=== MAE by lead bucket ===")
    print(bench.pool(scores, by=["label", "lead_lo"]).pivot(
        index="label", columns="lead_lo", values="mae").round(3).to_string())


if __name__ == "__main__":
    main()
