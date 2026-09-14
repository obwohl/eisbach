"""Confirm the missing hourly-mean arm: necessary to isolate accumulation at 8760 h."""
import json
import logging

import bench
import pandas as pd
from exp15_rain import BASE_FUTURE, BASE_PAST, load, sweep


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    manifest = json.loads((bench.CACHE / 'exp15_rain.json').read_text())
    anchors = [pd.Timestamp(t) for t in manifest['anchors']]
    df = load()
    variants = [('Mittel, stündlich', [*BASE_FUTURE, 'rain_catchment_hourly'], BASE_PAST)]
    sweep(bench.load_forecaster(), df, df.eisbach, anchors, variants,
          bench.CACHE / 'exp15_hourly_confirm.csv', context=8760)


if __name__ == '__main__':
    main()
