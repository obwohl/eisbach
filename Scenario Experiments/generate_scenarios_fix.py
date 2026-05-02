import pandas as pd
import torch
from chronos import Chronos2Pipeline
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import prepare_data

np.random.seed(42)

df_long, df_wetter = prepare_data()
df_wassertemp = df_long[df_long['cols'] == 'wassertemp'].dropna(subset=['data'])
df_wassertemp['date'] = pd.to_datetime(df_wassertemp['date'])
df_wassertemp = df_wassertemp.sort_values('date')

target_values = torch.tensor(df_wassertemp['data'].values, dtype=torch.float32)
last_timestamp = df_wassertemp['date'].max()

print("Loading Chronos-2 pipeline...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
    dtype=torch.float32,
)

print("Generating native 96-step forecast...")
native_pred = pipeline.predict(
    inputs=[{"target": target_values}],
    prediction_length=96
)[0][0]

quantiles = pipeline.quantiles
median_idx = quantiles.index(0.5)

num_scenarios = 20
horizon = 96

print(f"Generating {num_scenarios} autoregressive scenarios with controlled error accumulation...")
scenarios = []

for s in range(num_scenarios):
    sys.stdout.write(f"\rScenario {s+1}/{num_scenarios}")
    sys.stdout.flush()

    current_target = target_values.clone()
    scenario_path = []

    base_p = np.random.uniform(0.1, 0.9)

    for h in range(horizon):
        step_pred = pipeline.predict(
            inputs=[{"target": current_target}],
            prediction_length=1
        )[0][0]

        p = np.clip(np.random.normal(loc=base_p, scale=0.05), 0.01, 0.99)
        idx = np.abs(np.array(quantiles) - p).argmin()
        sampled_val = step_pred[idx, 0].item()

        scenario_path.append(sampled_val)
        current_target = torch.cat([current_target, torch.tensor([sampled_val], dtype=torch.float32)])

    scenarios.append(scenario_path)

print("\nDone generating scenarios.")

plt.figure(figsize=(12, 6))
time_axis = [last_timestamp + pd.Timedelta(hours=i) for i in range(1, horizon + 1)]

plt.fill_between(time_axis, native_pred[1].numpy(), native_pred[-2].numpy(), color='blue', alpha=0.1, label='Native 0.05-0.95 interval')
plt.plot(time_axis, native_pred[median_idx].numpy(), color='blue', linewidth=2, label='Native Median')

for i, sc in enumerate(scenarios):
    label = 'AR Scenarios' if i == 0 else None
    plt.plot(time_axis, sc, color='red', alpha=0.3, linewidth=1, label=label)

plt.title("Chronos-2: Native Multistep vs Autoregressive Scenarios (Fixed Quantile Bias)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Scenario Experiments/scenarios_comparison_fixed_bias.png')
print("Saved plot to Scenario Experiments/scenarios_comparison_fixed_bias.png")
