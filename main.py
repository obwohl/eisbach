"""Entrypoint: fetch, forecast, plot.

Run with ``python main.py``. Everything it produces — the two PNGs, ``Prediction.csv``
and the archive under ``data/archive/`` — is written relative to the working directory.
"""

import logging
import sys

import pandas as pd

from eisbach.data import prepare_data
from eisbach.inference import run_inference
from eisbach.plotting import plot_forecasts
from eisbach.validate import validate_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def main() -> int:
    try:
        issued_at = pd.Timestamp.now(tz="UTC")

        logger.info("Fetching water temperature and weather...")
        df_long, df_wetter, df_wt = prepare_data()

        logger.info("Running forecast and backtests...")
        df_inference, backtests = run_inference(df_long, df_wetter, df_wt)

        logger.info("Checking the result is plausible...")
        validate_run(df_inference, backtests, df_long)

        logger.info("Plotting...")
        plot_forecasts(df_long, df_wetter, df_inference, backtests, issued_at=issued_at)

        logger.info("Done.")
        return 0
    except Exception:
        # Log the traceback rather than just the message: when this fails it fails in
        # CI, where the traceback is the only thing anyone will have to go on.
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
