import pandas as pd
from datetime import datetime, timedelta

from src.data import prepare_data

df_long, df_wetter = prepare_data()

hist_wt = df_long[(df_long['cols'] == 'wassertemp')].dropna(subset=['data'])
max_date = hist_wt['date'].max()

print("Max valid historical date:", max_date)

bt_96 = max_date - pd.Timedelta(hours=96)
bt_192 = max_date - pd.Timedelta(hours=192)

print(f"96h backtest anchored at: {bt_96}")
print(f"192h backtest anchored at: {bt_192}")

# Let's read the saved inference data to verify ranges
df_bt_96 = pd.read_csv('data/inference_backtest_96_corrected.csv', parse_dates=[0], index_col=0)
df_bt_192 = pd.read_csv('data/inference_backtest_192_corrected.csv', parse_dates=[0], index_col=0)

print(f"96h forecast dates: {df_bt_96.index.min()} to {df_bt_96.index.max()}")
print(f"192h forecast dates: {df_bt_192.index.min()} to {df_bt_192.index.max()}")
