import logging
import sys
import warnings
from src.data import prepare_data
from src.inference import run_inference
from src.plotting import plot_forecasts

# For automated execution, logging should be used.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        logging.info("Starting data preparation...")
        df_long, df_wetter = prepare_data()

        logging.info("Data preparation complete. Starting inference...")
        df_inference, df_96, df_192, df_288 = run_inference(df_long)

        logging.info("Inference complete. Starting plotting...")
        plot_forecasts(df_long, df_wetter, df_inference, df_96, df_192, df_288)

        logging.info("Pipeline completed successfully.")
    except Exception as e:
        logging.exception(f"An error occurred during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
