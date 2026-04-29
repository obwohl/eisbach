import pandas as pd
from datetime import datetime, timedelta

from src.data import fetch_data_from_url

# The issue is that the ffill/bfill inside src.data.prepare_data is forward filling the water temperature
# out 8 days into the future along with the weather covariates. This causes a "flat line" in the plot
# and feeds fake historical data to the last backtests.
