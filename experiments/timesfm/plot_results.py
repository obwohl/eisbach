"""The four plots the experiments are worth looking at."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench  # noqa: E402
import viz  # noqa: E402

NAMES = {
    "duet_live": "DUET (ours, live)",
    "timesfm_air_replay": "TimesFM + DWD forecast",
    "timesfm_air_oracle": "TimesFM + observed air (oracle)",
    "timesfm_uni": "TimesFM, water only",
    "timesfm_air_past": "TimesFM + past air only",
    "timesfm_air_oracle_isar_past": "TimesFM + air (oracle) + Isar",
}
COLOUR = {
    "duet_live": viz.SERIES[1],
    "timesfm_air_replay": viz.SERIES[0],
    "timesfm_air_oracle": viz.SERIES[2],
    "timesfm_uni": viz.SERIES[3],
    "timesfm_air_past": viz.SERIES[4],
    "timesfm_air_oracle_isar_past": viz.SERIES[2],
}
LEAD_MID = {0: 12, 24: 36, 48: 60, 72: 84}

df = bench.load_dataset()
series = df["eisbach"]

# --- 1. how much history is worth having ------------------------------------
s1 = pd.read_csv(bench.CACHE / "exp1_context.csv", parse_dates=["reference_time"])
p1 = bench.pool(s1, by=["context_len", "lead_lo"])
fig, ax = plt.subplots(figsize=(7.4, 4.2))
for i, (lead, g) in enumerate(p1.groupby("lead_lo")):
    g = g.sort_values("context_len")
    ax.plot(g.context_len, g.mae, color=viz.SERIES[i], marker="o", ms=5, label=f"lead {lead}–{lead+24} h")
    ax.annotate(f"{lead}–{lead+24} h", (g.context_len.iloc[-1], g.mae.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", color=viz.SERIES[i], fontsize=8, weight="bold", va="center")
ax.set_xscale("log")
ax.set_xticks([384, 1024, 2048, 4380, 8760, 15360])
ax.set_xticklabels(["384\n(DUET)", "1k", "2k", "4380\n(½ yr)", "8760\n(1 yr)", "15360\n(max)"])
ax.set_xlabel("context length the model may look back over (hours)")
ax.set_ylabel("MAE (°C)")
ax.axvline(8760, color=viz.INK_2, lw=0.8, ls=":", zorder=0)
ax.legend(loc="lower left", ncol=2, fontsize=8)
ax.margins(x=0.12)
print(viz.finish(fig, "context_sweep.png",
                 title="A full year of history is the sweet spot",
                 subtitle="TimesFM 3.0 zero-shot, water only, 442 windows spread over 2023-2026. "
                          "Half a year is worse than a quarter; "
                          "the 1.75-year maximum adds nothing."))

# --- 2. the honest head to head ---------------------------------------------
s4 = pd.read_csv(bench.CACHE / "exp4_replay.csv", parse_dates=["reference_time"])
p4 = bench.pool(s4, by=["label", "lead_lo"])
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.2))
order = ["timesfm_air_replay", "duet_live", "timesfm_uni"]
for lab in order:
    g = p4[p4.label == lab].sort_values("lead_lo")
    x = [LEAD_MID[v] for v in g.lead_lo]
    axa.plot(x, g.mae, color=COLOUR[lab], marker="o", ms=5, label=NAMES[lab])
    axa.annotate(NAMES[lab].replace(" + ", "\n+ ").replace(", ", "\n"),
                 (x[-1], g.mae.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                 color=COLOUR[lab], fontsize=8, weight="bold", va="center")
axa.set_xlabel("forecast lead (hours ahead)")
axa.set_ylabel("MAE (°C)")
axa.set_xticks(list(LEAD_MID.values()))
axa.set_ylim(0, None)
axa.margins(x=0.28)
axa.legend(loc="upper left")

# calibration: nominal 80 % band, how wide and how often right
pooled = bench.pool(s4)
y = np.arange(len(pooled))
axb.barh(y, pooled.width_80, height=0.55, color=[COLOUR[lab] for lab in pooled.label], zorder=2)
for i, r in pooled.reset_index(drop=True).iterrows():
    axb.text(r.width_80 + 0.05, i, f"{r.width_80:.2f} °C wide · covers {r.cov_80:.0%}",
             va="center", fontsize=8, color=viz.INK)
axb.set_yticks(y)
axb.set_yticklabels([NAMES[lab] for lab in pooled.label], fontsize=8)
axb.invert_yaxis()
axb.set_xlabel("width of the nominal 80 % band (°C)  ·  target coverage 80 %")
axb.set_xlim(0, pooled.width_80.max() * 1.75)
axb.grid(axis="y", visible=False)
print(viz.finish(fig, "head_to_head.png",
                 title="TimesFM 3.0 zero-shot against our trained model, 76 identical windows",
                 subtitle="Both fed the DWD forecast as it was really issued; both scored on the same "
                          "deciles against the raw gauge. Narrower bands that still cover "
                          "are strictly better."))

# --- 3. the covariate pathway -----------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 3.8))
leads = np.array([24, 96])
tfm = np.array([0.5995, 1.0875])
ax.plot(leads, tfm, color=viz.SERIES[0], marker="o", ms=7, lw=2.2)
ax.annotate("TimesFM 3.0\n+1.09 °C at 96 h", (96, 1.0875), xytext=(-12, -34),
            textcoords="offset points", color=viz.SERIES[0], fontsize=9, weight="bold")
ax.axhline(1.9e-6, color=viz.SERIES[1], lw=2.2)
ax.annotate("DUET (ours): 1.9e-6 °C — the covariate never arrives", (26, 1.9e-6),
            xytext=(0, 9), textcoords="offset points", color=viz.SERIES[1], fontsize=9, weight="bold")
ax.set_yscale("log")
ax.set_ylim(1e-7, 5)
ax.set_xticks([24, 96])
ax.set_xlabel("forecast lead (hours)")
ax.set_ylabel("shift of the water forecast (°C)")
print(viz.finish(fig, "covariate_probe.png",
                 title="Add 3 °C to the air-temperature forecast — does the water move?",
                 subtitle="Same probe as PRD R6, 20 windows. A model that ignores the absolute level of "
                          "its covariate cannot use a heatwave forecast."))
