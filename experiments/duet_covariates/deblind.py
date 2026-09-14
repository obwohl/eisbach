"""Can the live model's level blindness be taken away without retraining it?

The blindness is arithmetic, not something the weights learned. ``RevIN._get_statistics``
takes the per-channel **median over the 384-hour window** as the location and the RMS
about it as the scale, and divides them out before any weight sees the input. So the
absolute level of a covariate is deleted upstream of the model entirely.

That rules out one whole family of fixes. **No transform that is constant within the
window can put the level back**, because RevIN removes it again by construction — and a
climatological anomaly is very nearly constant across sixteen days. Variant ``anomaly``
below is here to show that, not because it is expected to work.

What can work is changing the statistics themselves: give the two covariate channels a
**climatological** location and scale instead of their window's own, so that a warm
fortnight normalises to a higher number than a cold one. The water channel keeps its
window statistics — the output distribution is denormalised with them, so touching them
would corrupt the forecast rather than inform it.

The catch, and the reason this is a measurement rather than a fix: the model was trained
with window statistics. Feeding it climatologically normalised covariates is
out-of-distribution input to a router that selects experts from the channel mean. Whether
the level it can now see is worth more than the distribution shift costs is exactly what
the table at the bottom answers.

The vendored tree stays byte-identical, as it must: the override is a runtime patch on
the loaded instance, applied and removed here.
"""
from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "timesfm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench  # noqa: E402
from probe import windows  # noqa: E402

from eisbach.data import COVARIATE_SHIFT_HOURS  # noqa: E402
from eisbach.model import QUANTILES, forecast, load_model  # noqa: E402

logger = logging.getLogger(__name__)

WATER_COLUMNS = [f"wassertemp_q{q}" for q in QUANTILES]

#: Channel slots whose statistics may be overridden. Slot 0 is wassertemp, whose stats
#: the output is denormalised with.
COVARIATE_SLOTS = (1, 2)


def climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day-of-year location and scale for each covariate, from the whole record.

    Location is the median and scale the RMS about it, over a 31-day window centred on
    each day of the year and pooled across sixteen years — the same two statistics RevIN
    computes, taken over climate instead of over one fortnight.
    """
    cov = pd.DataFrame({
        "airtemp_96": df["airtemp"].shift(-COVARIATE_SHIFT_HOURS),
        "pressure_96": df["pressure"].shift(-COVARIATE_SHIFT_HOURS),
    })
    doy = cov.index.dayofyear
    rows = []
    for day in range(1, 367):
        # Circular +-15 day window.
        offset = (doy - day + 183) % 366 - 183
        sel = np.abs(offset) <= 15
        block = cov[sel]
        loc = block.median()
        rows.append({"doy": day,
                     **{f"{c}_loc": loc[c] for c in cov.columns},
                     **{f"{c}_scale": float(np.sqrt(((block[c] - loc[c]) ** 2).mean()))
                        for c in cov.columns}})
    return pd.DataFrame(rows).set_index("doy")


@contextlib.contextmanager
def overridden_stats(model, location=None, scale=None, blend: float = 1.0):
    """Force RevIN's covariate statistics for the duration of the block.

    ``location`` and ``scale`` are per-slot dicts; a slot left out keeps the window's own.
    ``blend`` mixes the two: 0 leaves the window statistics alone, 1 replaces them. Between
    them the level reaches the model attenuated, which trades how much of it arrives
    against how far the input drifts from what the model was trained on.
    """
    revin = model.cluster.revin
    original = revin._get_statistics

    def patched(x):
        original(x)
        for slot in COVARIATE_SLOTS:
            if location is not None and slot in location:
                revin.location_stat[..., slot] = (
                    (1.0 - blend) * revin.location_stat[..., slot] + blend * float(location[slot])
                )
            if scale is not None and slot in scale:
                revin.scale_stat[..., slot] = (
                    (1.0 - blend) * revin.scale_stat[..., slot] + blend * float(scale[slot])
                )

    revin._get_statistics = patched
    try:
        yield
    finally:
        revin._get_statistics = original


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = bench.load_dataset()
    truth = df["eisbach"]
    clim = climatology(df)
    blocks = windows(df, n=60, every_hours=61)
    logger.info("%d windows", len(blocks))

    model, config = load_model()
    air_shifted = df["airtemp"].shift(-COVARIATE_SHIFT_HOURS)
    # A climatological hourly cycle, so "anomaly" removes the daily shape too.
    hourly_clim = air_shifted.groupby(
        [air_shifted.index.dayofyear, air_shifted.index.hour]).median()

    rows = []
    for block in blocks:
        anchor = block.index[-1]
        targets = pd.date_range(anchor + pd.Timedelta(hours=1), periods=96, freq="1h", tz="UTC")
        y = truth.reindex(targets).to_numpy(dtype=float)
        if np.isfinite(y).sum() < 48:
            continue

        day = anchor.dayofyear
        c = clim.loc[day]
        loc = {1: c["airtemp_96_loc"], 2: c["pressure_96_loc"]}
        scale = {1: c["airtemp_96_scale"], 2: c["pressure_96_scale"]}

        anomaly = block.copy()
        keys = pd.MultiIndex.from_arrays([block.index.dayofyear, block.index.hour])
        anomaly["airtemp_96"] = (block["airtemp_96"].to_numpy()
                                 - hourly_clim.reindex(keys).to_numpy())

        variants = [
            ("as shipped", block, {}),
            ("covariate -> climatological anomaly", anomaly, {}),
            ("RevIN location := climatological", block, {"location": loc}),
            ("RevIN location+scale := climatological", block, {"location": loc, "scale": scale}),
        ]
        for label, frame, override in variants:
            if frame.isna().any(axis=None):
                continue
            with overridden_stats(model, **override):
                fc = forecast(model, config, frame)[WATER_COLUMNS].to_numpy()
            run = bench.Run(label=label, reference_time=anchor, target_times=targets,
                            quantiles=bench.to_deciles(fc, QUANTILES), truth=y)
            rows.extend(bench.score(run, diurnal=bench.diurnal_baseline(truth, targets)))

    scores = pd.DataFrame(rows)
    scores.to_csv(bench.CACHE / "duet_deblind.csv", index=False)
    print(f"\n=== DUET, {scores.reference_time.nunique()} windows, same truth, decile grid ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "rmse", "crps", "bias",
                              "cov_80", "width_80"]].to_string(index=False))

    # How much of the level can the model take before the distribution shift costs more
    # than the level is worth?
    print("\n=== blending the window median towards the climatological one ===")
    print(f"{'blend':>7} {'MAE':>8} {'CRPS':>8} {'bias':>9}   "
          f"{'shift response at +3 °C':>24}")
    sweep = []
    for blend in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        rows_b = []
        for block in blocks:
            anchor = block.index[-1]
            targets = pd.date_range(anchor + pd.Timedelta(hours=1), periods=96,
                                    freq="1h", tz="UTC")
            y = truth.reindex(targets).to_numpy(dtype=float)
            if np.isfinite(y).sum() < 48 or block.isna().any(axis=None):
                continue
            c = clim.loc[anchor.dayofyear]
            loc = {1: c["airtemp_96_loc"], 2: c["pressure_96_loc"]}
            with overridden_stats(model, location=loc, blend=blend):
                fc = forecast(model, config, block)[WATER_COLUMNS].to_numpy()
            run = bench.Run(label=f"blend {blend}", reference_time=anchor,
                            target_times=targets,
                            quantiles=bench.to_deciles(fc, QUANTILES), truth=y)
            rows_b.extend(bench.score(run))
        pooled = bench.pool(pd.DataFrame(rows_b)).iloc[0]

        deltas = []
        for b in blocks[:20]:
            c = clim.loc[b.index[-1].dayofyear]
            loc = {1: c["airtemp_96_loc"], 2: c["pressure_96_loc"]}
            with overridden_stats(model, location=loc, blend=blend):
                base = forecast(model, config, b)["wassertemp_q0.5"].to_numpy()
                warm = b.copy()
                warm["airtemp_96"] = warm["airtemp_96"] + 3.0
                deltas.append(forecast(model, config, warm)["wassertemp_q0.5"].to_numpy() - base)
        response = float(np.abs(np.stack(deltas)).mean())

        print(f"{blend:>7.2f} {pooled.mae:>8.4f} {pooled.crps:>8.4f} {pooled.bias:>+9.4f}   "
              f"{response:>21.4f} °C")
        sweep.append({"blend": blend, "mae": pooled.mae, "crps": pooled.crps,
                      "bias": pooled.bias, "shift_response": response})
    pd.DataFrame(sweep).to_csv(bench.CACHE / "duet_deblind_sweep.csv", index=False)


if __name__ == "__main__":
    main()
