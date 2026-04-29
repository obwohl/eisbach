import pandas as pd
from src.data import prepare_data

df_long, df_wetter = prepare_data()

hist_wt = df_long[df_long['cols'] == 'wassertemp']

print("Historical Wassertemp NaN count:", hist_wt['data'].isna().sum())

print("Last 10 dates of valid historical data:")
print(hist_wt.dropna().tail(10))

print("\nLast 10 dates of all historical data:")
print(hist_wt.tail(10))

print("\nDate range of all historical data:")
print(hist_wt['date'].min(), "to", hist_wt['date'].max())
