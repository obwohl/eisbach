import logging
import sys
import warnings
import os
import glob
from datetime import datetime
from src.data import prepare_data
from src.inference import run_inference
from src.plotting import plot_forecasts

# For automated execution, logging should be used.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Generate timestamp string
        timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        logging.info(f"Using timestamp for output files: {timestamp_str}")

        # Clean up old prediction files
        for f in glob.glob("Prediction_*_*.png") + glob.glob("Prediction_*_*.csv") + glob.glob("Prediction_*_*.html") + glob.glob("eisbach_new.png"):
            try:
                os.remove(f)
                logging.info(f"Removed old file: {f}")
            except Exception as e:
                logging.warning(f"Failed to remove old file {f}: {e}")

        logging.info("Starting data preparation...")
        df_long, df_wetter = prepare_data()

        logging.info("Data preparation complete. Starting inference...")
        df_inference, df_96, df_192, df_288 = run_inference(df_long, timestamp_str)

        logging.info("Inference complete. Starting plotting...")
        plot_forecasts(df_long, df_wetter, df_inference, df_96, df_192, df_288, timestamp_str)

        logging.info("Pipeline completed successfully.")
    except Exception as e:
        logging.exception(f"An error occurred during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
