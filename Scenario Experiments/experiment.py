import pandas as pd
import torch
from chronos import Chronos2Pipeline
import numpy as np
import sys
import os

# Create data structures matching what Chronos2 expects
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import prepare_data

df_long, df_wetter = prepare_data()
df_wassertemp = df_long[df_long['cols'] == 'wassertemp'].dropna(subset=['data'])
df_wassertemp['date'] = pd.to_datetime(df_wassertemp['date'])
df_wassertemp = df_wassertemp.sort_values('date')

target_values = torch.tensor(df_wassertemp['data'].values, dtype=torch.float32)

print("Loading Chronos-2 pipeline...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
    dtype=torch.float32,
)

print("Running full 96-step forecast natively...")
full_prediction = pipeline.predict(
    inputs=[{"target": target_values}],
    prediction_length=96
)

print(f"Full forecast shape: {full_prediction[0].shape}")

print("Running step-by-step autoregressive forecast (10 steps to test determinism)...")
for s in range(3):
    print(f"\nScenario {s+1}:")
    current_target_s = target_values.clone()
    scenario_preds = []

    for i in range(10):
        step_pred = pipeline.predict(
            inputs=[{"target": current_target_s}],
            prediction_length=1
        )
        median_idx = pipeline.quantiles.index(0.5) if 0.5 in pipeline.quantiles else len(pipeline.quantiles)//2
        next_val = step_pred[0][0, median_idx, 0]
        scenario_preds.append(next_val.item())
        current_target_s = torch.cat([current_target_s, next_val.unsqueeze(0)])

    print(f"Predictions: {scenario_preds}")
