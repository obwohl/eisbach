"""What the live model does and does not see in its weather covariate."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "timesfm"))
import bench  # noqa: E402
import viz  # noqa: E402

LEAD_MID = {0: 12, 24: 36, 48: 60, 72: 84}

probe = pd.read_csv(bench.CACHE / "duet_covariate_probe.csv")
use = pd.read_csv(bench.CACHE / "duet_usefulness.csv", parse_dates=["reference_time"])

fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.8, 4.6), width_ratios=[1.15, 1])

# --- left: is the covariate used at all, and does it help? --------------------
pooled = bench.pool(use, by=["label", "lead_lo"])
colours = {"real DWD forecast": viz.SERIES[0],
           "air held at the window mean": viz.SERIES[3],
           "weather from a week earlier": viz.SERIES[1]}
for lab, colour in colours.items():
    g = pooled[pooled.label == lab].sort_values("lead_lo")
    x = [LEAD_MID[v] for v in g.lead_lo]
    ax.plot(x, g.mae, color=colour, marker="o", ms=5, label=lab)
    ax.annotate(lab.replace(" from ", "\nfrom ").replace(" at ", "\nat ").replace(" DWD", "\nDWD"),
                (x[-1], g.mae.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                color=colour, fontsize=8, weight="bold", va="center")
ax.set_xticks(list(LEAD_MID.values()))
ax.set_xlabel("forecast lead (hours ahead)")
ax.set_ylabel("MAE (°C)")
ax.set_ylim(0, None)
ax.margins(x=0.30)
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("Feeding it the wrong weather is worse than feeding it none", fontsize=9.5)

# --- right: what it is blind to ----------------------------------------------
probe = probe.sort_values("mean_abs_delta")
short = {
    "whole air channel +3 C": "air channel +3 °C",
    "whole air channel +10 C": "air channel +10 °C",
    "air daily swing doubled, mean kept": "air daily swing doubled",
    "whole pressure channel -> its own mean": "pressure → constant",
    "whole air channel -> its own mean": "air → constant",
    "future air +5 C": "forecast part +5 °C",
    "future air -> a real heatwave": "forecast part → a heatwave",
    "future air -> a real cold spell": "forecast part → a cold spell",
}
y = range(len(probe))
# Below this the response is float32 rounding, not signal: the three affine cases all
# top out at 1.907e-06, which is the number PRD R6 quotes.
NOISE = 1e-4
axb.barh(list(y), probe.mean_abs_delta, height=0.6,
         color=[viz.SERIES[1] if v < NOISE else viz.SERIES[0]
                for v in probe.mean_abs_delta], zorder=2)
for i, (_, r) in enumerate(probe.iterrows()):
    noise = r.mean_abs_delta < NOISE
    label = "2e-7 — float32 rounding" if noise else f"{r.mean_abs_delta:.2f} °C"
    axb.text(max(r.mean_abs_delta, 0.012) + 0.04, i, label, va="center", fontsize=8.5,
             color=viz.SERIES[1] if noise else viz.INK)
axb.set_yticks(list(y))
axb.set_yticklabels([short[v] for v in probe.variant], fontsize=8.5)
axb.set_xlabel("how far the water forecast moves (mean |Δ|, °C)")
axb.set_xlim(0, probe.mean_abs_delta.max() * 1.55)
axb.grid(axis="y", visible=False)
axb.set_title("Affine changes to a whole channel vanish; shape changes do not", fontsize=9.5)

print(viz.finish(fig, "duet_weather.png",
                 title="Does the live model use the weather forecast? Yes — but only its shape",
                 subtitle="DUET on real windows. RevIN normalises each covariate per window, so any "
                          "shift or rescale of a whole channel cancels exactly — and only exactly "
                          "that class of change is invisible."))
