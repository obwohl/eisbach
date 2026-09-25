"""Entrypoint: fetch, forecast, plot.

Run with ``python main.py``. Everything it produces — the PNGs, the CSVs and the archive
under ``data/archive/`` — is written relative to the working directory.

Two models run here, and only one of them is load-bearing. DUET is the production
forecast; TimesFM is a candidate published beside it so it can be judged on real runs
rather than on sweeps over weather that had already happened. The candidate is therefore
run *after* everything DUET needs is on disk, and its failures are logged and swallowed:
a candidate that breaks costs the page one of its two models, never the forecast.

Both models are plotted together, at the end, because the page switches between them on
one set of axes: the limits are the union of the two, so they can only be chosen once
both forecasts exist.
"""

import logging
import sys

import pandas as pd

from eisbach.covariates import prepare_live
from eisbach.inference import run_inference
from eisbach.plotting import duet_view, plot_models, remove
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

        duet = duet_view(df_long, df_weather, df_inference, backtests, issued_at=issued_at)
        candidate = run_candidate(issued_at)

        logger.info("Plotting...")
        plot(duet, candidate)

        logger.info("Done.")
        return 0
    except Exception:
        # Log the traceback rather than just the message: when this fails it fails in
        # CI, where the traceback is the only thing anyone will have to go on.
        logger.exception("Pipeline failed")
        return 1


def plot(duet, candidate) -> None:
    """Both models on shared axes, or DUET alone if drawing the candidate fails.

    A candidate whose frames break the renderer must not cost the production images, and
    it must not leave one half of its own pair behind either: the page offers the switch
    whenever the candidate's forecast image exists.
    """
    if candidate is not None:
        try:
            plot_models([duet, candidate])
            return
        except Exception:
            logger.exception("Plotting the TimesFM candidate failed; plotting DUET alone")
            remove(candidate.paths)
    plot_models([duet])


def run_candidate(issued_at):
    """Run TimesFM beside the production forecast, and never let it take the run down.

    Deliberately broad: this catches a missing checkpoint, a station Bright Sky stopped
    forecasting, a context too thin to run on, and whatever the next surprise is. The
    published DUET forecast has already been written by the time this is called, so the
    worst outcome is a page showing one model instead of two, with the reason in the log.
    Returns the candidate's picture, ready to draw, or ``None``.

    It is not silent about it. The traceback is logged at error level, which is what CI
    keeps, because a candidate that quietly stops running would look exactly like a
    candidate that is doing fine.
    """
    try:
        # Imported here, not at module scope: this pulls in the TimesFM checkpoint loader
        # and a 1.3 GB download, and nothing else in the run should wait on that.
        from eisbach.plotting import timesfm_view
        from eisbach.timesfm import backtests as resolve_backtests
        from eisbach.timesfm import run as run_timesfm
        from eisbach.timesfm import write_csv

        logger.info("Running the TimesFM candidate...")
        quantiles, context, future = run_timesfm(issued_at=issued_at)
        write_csv(quantiles)
        # After the forecast and its archive write, never before: a backtest that fails
        # must not cost the forecast people actually read.
        backtests, missing = resolve_backtests(context.index[-1], issued_at=issued_at)
        return timesfm_view(context, future, quantiles, issued_at=issued_at,
                            backtests=backtests, missing=missing)
    except Exception:
        logger.exception("TimesFM candidate failed; the published DUET forecast stands")
        return None


if __name__ == "__main__":
    sys.exit(main())
