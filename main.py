import logging
import sys
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

        # Outputs are written to fixed filenames and overwritten in place, so there is
        # nothing to clean up. The old glob-based cleanup matched timestamped names that
        # this pipeline stopped producing.

        logging.info("Starting data preparation...")
        df_long, df_wetter = prepare_data()

        logging.info("Data preparation complete. Starting inference...")
        df_inference, backtests = run_inference(df_long, timestamp_str)

        logging.info("Inference complete. Starting plotting...")
        plot_forecasts(df_long, df_wetter, df_inference, backtests, timestamp_str)

        logging.info("Pipeline completed successfully.")
    except Exception as e:
        logging.exception(f"An error occurred during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
