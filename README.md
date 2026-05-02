# Eisbach Forecast

This project uses modern time-series forecasting (Chronos-2 via AutoGluon) to predict the water temperature of the Eisbach river in Munich, driven by historical water temperature data and future weather covariates.

## Workflow & Output Tracking
The workflow is triggered via a daily cronjob (or manually via GitHub dispatch). Upon execution, the following happens:

1. **Clean-Up**: Previous prediction plots and CSV files in the root folder are deleted to prevent unnecessary repo bloat.
2. **Inference**: A forecast is calculated.
3. **Data Outputs**:
   - `Prediction_[YYYY-MM-DD_HH-MM].csv`: A readable, raw prediction file including quantile ranges. Only the newest file is tracked.
4. **Plot Outputs**:
   - `Prediction_[YYYY-MM-DD_HH-MM].png`: A clean visualization showing just the main prediction along with historical context and air temperature.
   - `Prediction_Backtest_[YYYY-MM-DD_HH-MM].png`: An extended plot that includes the main forecast as well as historical backtests (-96h, -192h, -288h) for accuracy verification.
   - Both plots dynamically embed the execution timestamp in the file name and the plot title.
5. **Archiving (`src/archive_forecast.py`)**:
   - A pure, 5-day non-overlapping weather forecast archive is maintained at `data/forecast_archive/forecast_5d_archive.csv`.
   - To avoid redundant overlapping data, the script explicitly verifies the timestamp of the last archive entry. A new 5-day window is appended *only* if the previous entry is at least ~5 days old (4 days, 23 hours to account for runner variations).
   - This ensures we keep clean, true 5-day forward predictions for future quality validation without data duplication.

## Directory Structure
- `data/`: Used for intermediate execution data (ignored by git, except the `forecast_archive` sub-folder).
- `src/`: Core Python modules for data fetching, inference, plotting, and archiving.
- `.github/workflows/`: Contains the CI/CD pipeline definition (`daily_cron.yml`), which handles auto-committing the newest prediction outputs and archive data back to the repository.

## Installation & Local Execution
Ensure `ts_proba_cuda` submodule is initialized:
```bash
git submodule update --init --recursive
```

Install requirements:
```bash
pip install -r requirements.txt
```

Run the pipeline:
```bash
PYTHONPATH=. python main.py
```
