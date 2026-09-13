"""What the long dataset actually contains, and the one reading that should not be in it."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import viz  # noqa: E402

raw = bench.load_dataset(clean=False)
df = bench.load_dataset()

# --- 1. sixteen years of river temperature, gaps left as gaps ------------------
daily = df[["eisbach", "isar"]].resample("1D").mean()
fig, (ax, axc) = plt.subplots(2, 1, figsize=(11, 5.2), height_ratios=[4, 1], sharex=True)
for col, colour, label, dy in (("eisbach", viz.SERIES[0], "Eisbach", 9),
                               ("isar", viz.SERIES[1], "Isar", -11)):
    ax.plot(daily.index, daily[col], color=colour, lw=1.0, label=label)
    last = daily[col].dropna()
    ax.annotate(label, (last.index[-1], last.iloc[-1]), xytext=(8, dy),
                textcoords="offset points", color=colour, fontsize=9, weight="bold", va="center")
ax.margins(x=0.035)
ax.set_ylabel("daily mean water temperature (°C)")
ax.legend(loc="upper left", ncol=2)

# coverage strip: fraction of each month's hours the gauge actually measured
cov = df[["eisbach", "isar"]].notna().resample("MS").mean()
axc.fill_between(cov.index, 0, cov["eisbach"], color=viz.SERIES[0], alpha=0.75, step="post", lw=0)
axc.step(cov.index, cov["isar"], color=viz.SERIES[1], where="post", lw=1.4)
axc.set_ylim(0, 1.02)
axc.set_ylabel("hours\nmeasured")
axc.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
axc.xaxis.set_major_locator(mdates.YearLocator(2))
axc.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
p1 = viz.finish(fig, "data_overview.png",
                title="Sixteen years of hourly gauge data",
                subtitle="GKD Bayern, 15-minute samples averaged to hours. Breaks are real outages, "
                         "not interpolated. The Isar is missing 2020-01 to 2022-07.")

# --- 2. the instrument fault the pipeline has no gate against -----------------
win = slice("2026-09-08 00:00", "2026-09-10 12:00")
obs = bench.observations()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4), width_ratios=[1, 1])
a1.plot(raw["eisbach"][win].index, raw["eisbach"][win].to_numpy(), color=viz.SERIES[0],
        marker="o", ms=3, label="hourly mean of the 15-min samples")
a1.plot(obs[win].index, obs[win].to_numpy(), color=viz.CRITICAL, lw=1.2, ls="--",
        marker="s", ms=3, label="production observation store")
bad = obs[win].idxmax()
a1.annotate(f"{obs[bad]:.1f} °C", (bad, obs[bad]), xytext=(10, -6), textcoords="offset points",
            color=viz.CRITICAL, weight="bold", fontsize=10)
a1.set_ylabel("water temperature (°C)")
a1.legend(loc="center right")
a1.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))

sc = pd.concat([pd.read_csv(p) for p in sorted((bench.ARCHIVE / "verification").glob("*.csv"))])
sc["reference_time"] = pd.to_datetime(sc.reference_time, utc=True)
live = sc[sc.kind == "live"]
a2.scatter(live.reference_time, live.mae, s=9, color=viz.SERIES[0], alpha=0.5, lw=0)
poisoned = live[(live.reference_time <= pd.Timestamp("2026-09-09 08:00", tz="UTC")) &
                (live.reference_time >= pd.Timestamp("2026-09-09 08:00", tz="UTC") - pd.Timedelta(hours=96))]
a2.scatter(poisoned.reference_time, poisoned.mae, s=22, color=viz.CRITICAL, lw=0,
           label=f"{len(poisoned)} stored rows poisoned by that one hour")
a2.set_yscale("log")
a2.set_ylabel("MAE of a stored verification row (°C)")
a2.legend(loc="upper left")
a2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
p2 = viz.finish(fig, "bad_reading.png",
                title="One bad gauge reading, and what it did to the verification store",
                subtitle="2026-09-09 08:00 UTC. The store is append-only, so these rows stand "
                         "until their partition is deleted and rebuilt.")
print(p1)
print(p2)
