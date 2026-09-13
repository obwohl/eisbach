"""One cold snap, both models, same window — what the score tables actually mean."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench
import exp3_headtohead as e3  # noqa: E402
import viz  # noqa: E402
from exp4_replay import replay_airtemp  # noqa: E402
from exp5_regimes import swing  # noqa: E402

e3.CONTEXT = CONTEXT = 8760

df = bench.load_dataset()
series = df["eisbach"]
idx = df.index

s4 = pd.read_csv(bench.CACHE / "exp4_replay.csv", parse_dates=["reference_time"])
# Two runs anchored at or after 2026-09-09 08:00 UTC had the 154.4 C instrument fault
# in DUET's real input window but not in the cleaned series TimesFM is given. That is a
# confound, so neither is allowed to be the illustration.
FAULT = pd.Timestamp("2026-09-09 08:00", tz="UTC")
runs = [t for t in sorted(s4.reference_time.unique()) if pd.Timestamp(t) < FAULT]
swings = {ts: swing(series, pd.Timestamp(ts)) for ts in runs}
# The sharpest cooling window, and the sharpest warming one.
picks = [min(swings, key=lambda t: swings[t]), max(swings, key=lambda t: swings[t])]

duet_all = {r.reference_time: r for r in e3.duet_runs(series)}

from timesfm3 import TimesFM3Forecaster  # noqa: E402

fc = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=False)
for ax, ts in zip(axes, picks, strict=True):
    ts = pd.Timestamp(ts)
    d = duet_all[ts]
    targets = d.target_times
    air = replay_airtemp(ts, targets)

    pos = idx.get_loc(ts)
    ctx = series.iloc[pos - CONTEXT + 1: pos + 1].to_numpy(dtype=np.float32)
    hist_air = df["airtemp"].iloc[pos - CONTEXT + 1: pos + 1].to_numpy(dtype=np.float32)
    pf = np.concatenate([hist_air, air])[None, :].astype(np.float32)
    out = fc.predict(ctx, horizon=bench.HORIZON, past_future_covariates=pf, return_quantiles=True)
    tq = out.quantiles if out.quantiles.ndim == 2 else out.quantiles[0]
    dq = d.quantiles  # already on the decile grid

    past = series.loc[ts - pd.Timedelta(hours=48): ts]
    ax.plot(past.index, past.to_numpy(), color=viz.INK, lw=1.4)
    truth = series.reindex(targets)
    ax.plot(truth.index, truth.to_numpy(), color=viz.INK, lw=2.0, label="what happened")

    for q, colour, name in ((dq, viz.SERIES[1], "DUET (ours, live)"),
                            (tq, viz.SERIES[0], "TimesFM + DWD forecast")):
        ax.fill_between(targets, q[:, 0], q[:, -1], color=colour, alpha=0.16, lw=0)
        ax.plot(targets, q[:, 4], color=colour, lw=1.8, label=name)

    ax.axvline(ts, color=viz.INK_2, lw=0.8, ls=":")
    ax.set_title(f"{'cold snap' if swings[ts] < 0 else 'warm spell'}: "
                 f"{swings[ts]:+.1f} °C over 96 h   ·   anchor {ts:%d %b %H:%M} UTC", fontsize=9.5)
    ax.set_ylabel("water temperature (°C)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    mae_d = np.nanmean(np.abs(dq[:, 4] - truth.to_numpy()))
    mae_t = np.nanmean(np.abs(tq[:, 4] - truth.to_numpy()))
    ax.annotate(f"MAE  DUET {mae_d:.2f}  ·  TimesFM {mae_t:.2f}", (0.02, 0.03),
                xycoords="axes fraction", fontsize=8.5, color=viz.INK_2)
axes[0].legend(loc="upper right", fontsize=8.5)
print(viz.finish(fig, "case_windows.png",
                 title="The two sharpest regime changes in the replayable archive",
                 subtitle="Shaded: the nominal 80 % band each model claims. Both models saw the same "
                          "DWD forecast; neither saw the future river."))
