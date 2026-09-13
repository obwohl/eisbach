"""What restoring the covariate's absolute level buys, and what it costs."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "timesfm"))
import bench  # noqa: E402
import viz  # noqa: E402

s = pd.read_csv(bench.CACHE / "duet_deblind_sweep.csv")

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)

a.plot(s.blend, s.mae, color=viz.SERIES[0], marker="o", ms=5, label="MAE")
a.plot(s.blend, s.crps, color=viz.SERIES[2], marker="o", ms=5, label="CRPS")
for col, colour, name in (("mae", viz.SERIES[0], "MAE"), ("crps", viz.SERIES[2], "CRPS")):
    a.annotate(name, (s.blend.iloc[-1], s[col].iloc[-1]), xytext=(8, 0),
               textcoords="offset points", color=colour, fontsize=8.5, weight="bold", va="center")
a.set_ylabel("error (°C)")
a.set_title("What it costs", fontsize=9.5)
a.legend(loc="center left", fontsize=8.5)
a.margins(x=0.16)

b.plot(s.blend, s.shift_response, color=viz.SERIES[1], marker="o", ms=5,
       label="movement when the air forecast is +3 °C")
b.plot(s.blend, s.bias.abs(), color=viz.SERIES[3], marker="o", ms=5,
       label="|bias| of the median")
b.annotate("level sensitivity\nregained", (s.blend.iloc[-1], s.shift_response.iloc[-1]),
           xytext=(-4, -30), textcoords="offset points", color=viz.SERIES[1],
           fontsize=8.5, weight="bold", ha="right")
b.annotate("|bias|", (s.blend.iloc[-1], abs(s.bias.iloc[-1])), xytext=(8, 0),
           textcoords="offset points", color=viz.SERIES[3], fontsize=8.5,
           weight="bold", va="center")
b.set_ylabel("°C")
b.set_title("What it buys", fontsize=9.5)
b.legend(loc="upper left", fontsize=8.5)
b.margins(x=0.16)

for ax in (a, b):
    ax.set_xlabel("how far RevIN's window median is moved towards the climatological one")
    ax.set_xticks(list(s.blend))
    ax.set_xticklabels([f"{v:.0%}" for v in s.blend])

print(viz.finish(fig, "duet_deblind.png",
                 title="Giving the live model back the absolute level — without retraining it",
                 subtitle="The surgery works: at 100 % a +3 °C air forecast finally moves the water "
                          "forecast, and the cold bias goes to zero. It still makes the forecast worse, "
                          "because the model has never seen an off-centre covariate."))
