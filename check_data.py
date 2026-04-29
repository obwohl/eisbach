import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data import fetch_data_from_url

end_date = datetime.now()
start_date = end_date - timedelta(days=40)
url = f"https://www.gkd.bayern.de/de/fluesse/wassertemperatur/bayern/muenchen-himmelreichbruecke-16515005/messwerte/tabelle?beginn={start_date.strftime('%d.%m.%Y')}&ende={end_date.strftime('%d.%m.%Y')}"

print(f"Fetching from: {url}")
df = fetch_data_from_url(url, "wassertemp")
print(f"Rows fetched: {len(df)}")
if len(df) > 0:
    print(df.head())
    print(df.tail())

    # Check for gaps greater than 1 hour
    df = df.sort_values('timestamp')
    df['diff'] = df['timestamp'].diff()
    gaps = df[df['diff'] > pd.Timedelta(hours=2)]
    print("\nLarge gaps in data:")
    print(gaps[['timestamp', 'diff']])
