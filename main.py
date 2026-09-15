"""Entrypoint: fetch, forecast, plot.

Run with ``python main.py``. Everything it produces — the PNGs, the CSVs and the archive
under ``data/archive/`` — is written relative to the working directory.

Two models run here, and only one of them is load-bearing. DUET is the production
forecast; TimesFM is a candidate published beside it so it can be judged on real runs
rather than on sweeps over weather that had already happened. The candidate is therefore
run *after* everything DUET needs is on disk, and its failures are logged and swallowed:
a candidate that breaks costs the page one of its two graphs, never the forecast.
"""

import logging
import sys

import pandas as pd

from eisbach.covariates import prepare_live
from eisbach.inference import run_inference
from eisbach.plotting import plot_forecasts, plot_timesfm
from eisbach.validate import validate_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def main() -> int:
    try:
        issued_at = pd.Timestamp.now(tz="UTC")

        logger.info("Fetching water temperature and weather...")
        df_long, df_weather, df_wt = prepare_live()

        logger.info("Running forecast and backtests...")
        df_inference, backtests = run_inference(df_long, df_weather, df_wt)

        logger.info("Checking the result is plausible...")
        validate_run(df_inference, backtests, df_long)

        logger.info("Plotting...")
        plot_forecasts(df_long, df_weather, df_inference, backtests, issued_at=issued_at)

        run_candidate(issued_at)

        logger.info("Done.")
        return 0
    except Exception:
        # Log the traceback rather than just the message: when this fails it fails in
        # CI, where the traceback is the only thing anyone will have to go on.
        logger.exception("Pipeline failed")
        return 1


def run_candidate(issued_at) -> None:
    """Run TimesFM beside the production forecast, and never let it take the run down.

    Deliberately broad: this catches a missing checkpoint, a station Bright Sky stopped
    forecasting, a context too thin to run on, and whatever the next surprise is. The
    published DUET forecast has already been written by the time this is called, so the
    worst outcome is a page showing one model instead of two, with the reason in the log.

    It is not silent about it. The traceback is logged at error level, which is what CI
    keeps, because a candidate that quietly stops running would look exactly like a
    candidate that is doing fine.
    """
    try:
        # Imported here, not at module scope: this pulls in the TimesFM checkpoint loader
        # and a 1.3 GB download, and nothing else in the run should wait on that.
        from eisbach.timesfm import backtests as resolve_backtests
        from eisbach.timesfm import run as run_timesfm
        from eisbach.timesfm import write_csv

        logger.info("Running the TimesFM candidate...")
        quantiles, context, future = run_timesfm(issued_at=issued_at)
        write_csv(quantiles)
        # After the forecast and its archive write, never before: a backtest that fails
        # must not cost the forecast people actually read.
        backtests = resolve_backtests(context.index[-1], issued_at=issued_at)
        written = plot_timesfm(context, future, quantiles, issued_at=issued_at,
                               backtests=backtests)
        logger.info("Candidate wrote %s", ", ".join(written))
    except Exception:
        logger.exception("TimesFM candidate failed; the published DUET forecast stands")


if __name__ == "__main__":
    sys.exit(main())
