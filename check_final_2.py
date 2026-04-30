import pandas as pd

df_main = pd.read_csv('data/inference.csv', parse_dates=[0], index_col=0)
df_bt_96 = pd.read_csv('data/inference_backtest_96_corrected.csv', parse_dates=[0], index_col=0)

print(f"Main forecast dates: {df_main.index.min()} to {df_main.index.max()}")
print(f"96h backtest dates: {df_bt_96.index.min()} to {df_bt_96.index.max()}")
