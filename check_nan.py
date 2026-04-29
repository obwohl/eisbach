import pandas as pd
from datetime import datetime, timedelta

from src.data import fetch_data_from_url

end_date = datetime.now()
start_date = end_date - timedelta(days=40)
url = f"https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/muenchen-himmelreichbruecke-16515005/messwerte/tabelle?beginn={start_date.strftime('%d.%m.%Y')}&ende={end_date.strftime('%d.%m.%Y')}"

df = fetch_data_from_url(url, "wassertemp")
print(f"Total rows: {len(df)}")
print(f"NaN rows: {df['wassertemp'].isna().sum()}")

nan_df = df[df['wassertemp'].isna()]
if len(nan_df) > 0:
    print(f"NaNs range from {nan_df['timestamp'].min()} to {nan_df['timestamp'].max()}")

valid_df = df.dropna(subset=['wassertemp'])
if len(valid_df) > 0:
    print(f"Valid data ranges from {valid_df['timestamp'].min()} to {valid_df['timestamp'].max()}")
