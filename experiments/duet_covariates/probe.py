"""Does the live model use the weather forecast, or only look like it does?

PRD R6 says the covariates are "barely connected to the water head" and gives three
measurements. This asks the blunter question the project actually cares about: if the
DWD forecast for the next four days had said something completely different, would the
published forecast have been different?

The input is 384 hours wide and ``airtemp_96`` is the air temperature shifted 96 hours
back, so the **last 96 rows of that channel are the weather forecast** — the only part of
the input that is not already history. Substituting exactly those rows is the experiment;
everything else is a control.

Nothing here touches the pipeline. It loads the production checkpoint read-only and runs
it on windows built from the long experiment dataset.
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

from eisbach.data import COVARIATE_SHIFT_HOURS  # noqa: E402
from eisbach.model import SERIES_ORDER, forecast, load_model  # noqa: E402

logger = logging.getLogger(__name__)

SEQ_LEN = 384
HORIZON = 96
MEDIAN = "wassertemp_q0.5"


def windows(df: pd.DataFrame, n: int = 30, every_hours: int = 121) -> list[pd.DataFrame]:
    """Model-input frames, each ``SEQ_LEN`` rows of the three trained channels."""
    wide = pd.DataFrame({
        "wassertemp": df["eisbach"],
        "airtemp_96": df["airtemp"].shift(-COVARIATE_SHIFT_HOURS),
        "pressure_96": df["pressure"].shift(-COVARIATE_SHIFT_HOURS),
    })[list(SERIES_ORDER)]

    out = []
    for end in pd.date_range("2025-06-01", "2026-09-01", freq=f"{every_hours}h", tz="UTC"):
        if end not in wide.index:
            continue
        stop = wide.index.get_loc(end)
        block = wide.iloc[stop - SEQ_LEN + 1: stop + 1]
        if len(block) < SEQ_LEN or block.notna().all(axis=None) is np.False_:
            continue
        if block.isna().any(axis=None):
            continue
        out.append(block)
        if len(out) >= n:
            break
    return out


def hot_and_cold(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """A real heatwave and a real cold spell, 96 hours each, to substitute in."""
    air = df["airtemp"].dropna()
    daily = air.resample("1D").mean()
    hot_day = daily.idxmax()
    cold_day = daily.idxmin()
    hot = air.loc[hot_day: hot_day + pd.Timedelta(hours=95)].to_numpy()[:96]
    cold = air.loc[cold_day: cold_day + pd.Timedelta(hours=95)].to_numpy()[:96]
    return hot, cold


def run(model, config, block: pd.DataFrame) -> np.ndarray:
    return forecast(model, config, block)[MEDIAN].to_numpy()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = bench.load_dataset()
    blocks = windows(df)
    logger.info("%d complete windows", len(blocks))
    hot, cold = hot_and_cold(df)
    logger.info("substitute weather: heatwave mean %.1f C, cold spell mean %.1f C",
                hot.mean(), cold.mean())

    model, config = load_model()

    def variant(name: str, mutate) -> dict:
        deltas = []
        for block in blocks:
            base = run(model, config, block)
            changed = block.copy()
            mutate(changed)
            deltas.append(run(model, config, changed) - base)
        d = np.stack(deltas)
        return {"variant": name,
                "mean_abs_delta": float(np.abs(d).mean()),
                "max_abs_delta": float(np.abs(d).max()),
                "delta_h24": float(d[:, 23].mean()),
                "delta_h96": float(d[:, -1].mean())}

    def set_future(column: str, values):
        """Replace only the horizon part of a covariate — the forecast, not the history."""
        def mutate(b):
            b.iloc[-HORIZON:, b.columns.get_loc(column)] = values
        return mutate

    def add_to_future(column: str, offset: float):
        def mutate(b):
            b.iloc[-HORIZON:, b.columns.get_loc(column)] += offset
        return mutate

    def replace_whole(column: str, fn):
        def mutate(b):
            b[column] = fn(b[column])
        return mutate


    rows = [
        # The question itself: replace the four-day DWD forecast, keep all history.
        variant("future air -> a real heatwave", set_future("airtemp_96", hot)),
        variant("future air -> a real cold spell", set_future("airtemp_96", cold)),
        variant("future air +5 C", add_to_future("airtemp_96", 5.0)),

        # Controls, to separate "ignores the level" from "ignores the covariate".
        variant("whole air channel +3 C", replace_whole("airtemp_96", lambda c: c + 3.0)),
        variant("whole air channel +10 C", replace_whole("airtemp_96", lambda c: c + 10.0)),
        variant("whole air channel -> its own mean",
                replace_whole("airtemp_96", lambda c: pd.Series(c.mean(), index=c.index))),
        variant("air daily swing doubled, mean kept",
                replace_whole("airtemp_96", lambda c: c.mean() + 2.0 * (c - c.mean()))),
        variant("whole pressure channel -> its own mean",
                replace_whole("pressure_96", lambda c: pd.Series(c.mean(), index=c.index))),
    ]

    out = pd.DataFrame(rows)
    out.to_csv(bench.CACHE / "duet_covariate_probe.csv", index=False)
    print(f"\n=== DUET, {len(blocks)} real windows: how far the water median moves (°C) ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    # For scale: how much does the water forecast move between consecutive real runs?
    spread = np.std([run(model, config, b)[-1] for b in blocks])
    print(f"\nFor scale: the h=96 median varies by {spread:.2f} °C (sd) across these windows.")


if __name__ == "__main__":
    main()
