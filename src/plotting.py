from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, LinearAxis, Range1d
from bokeh.layouts import column
from bokeh.io import save
import pandas as pd
from datetime import timedelta

def plot_forecasts(data: pd.DataFrame, future_pred: pd.DataFrame, backtest_preds: list, output_filename="eisbach_plot.html"):
    """
    Generates an interactive Bokeh plot.
    data: Historical data (dataframe with 'timestamp', 'wassertemp', 'lufttemperatur_c', 'niederschlag_mm')
    future_pred: Main forecast dataframe
    backtest_preds: List of backtest forecast dataframes
    """

    # Prepare data for plotting
    # Convert TSDataFrame to pandas if needed
    if hasattr(future_pred, "to_data_frame"): # Just in case
        future_pred = future_pred # It is already a DF usually

    # Combine backtests for easier handling?
    # Or plot them one by one.

    source = ColumnDataSource(data)

    p_main = figure(width=1600, height=600, x_axis_type="datetime", title="Eisbach: Wasser-/Lufttemperatur & Niederschlag")
    p_main.yaxis.axis_label = 'Temperatur [°C]'

    # Second axis for precipitation
    has_precip_data = 'niederschlag_mm' in data.columns and not data['niederschlag_mm'].isnull().all()
    if has_precip_data:
        precip_range_name = 'precip_range'
        max_precip = data['niederschlag_mm'].max(skipna=True)
        p_main.extra_y_ranges = {precip_range_name: Range1d(start=0, end=max(max_precip * 1.5, 1.0))}
        p_main.add_layout(LinearAxis(y_range_name=precip_range_name, axis_label="Niederschlag [mm]"), "right")

    # Plot Historical
    p_main.line(x='timestamp', y='wassertemp', source=source, line_width=2, color="black", legend_label="Wassertemperatur (Messwert)")
    if 'lufttemperatur_c' in data.columns:
        p_main.line(x='timestamp', y='lufttemperatur_c', source=source, line_width=1.5, color="orange", line_dash='dashed', legend_label="Lufttemperatur")

    # Plot Precipitation
    if has_precip_data:
        p_main.vbar(x='timestamp', top='niederschlag_mm', source=source,
                    width=timedelta(hours=0.8),
                    color="navy", alpha=0.6, legend_label="Niederschlag (1h)",
                    y_range_name=precip_range_name)

    # Plot Forecasts
    # Colors for backtests
    colors = ['#cce0ff', '#99c2ff', '#66a3ff', '#3385ff']

    # Helper to plot one forecast
    def plot_single_forecast(pred_df, color, label_suffix=""):
        # Reset index if multi-index (item_id, timestamp)
        if 'timestamp' not in pred_df.columns:
            pred_df = pred_df.reset_index()

        src = ColumnDataSource(pred_df)

        # Mean
        p_main.line(x='timestamp', y='mean', source=src, line_width=2.5, color="blue", legend_label=f"Prognose {label_suffix}")

        # Intervals (AutoGluon produces 0.1, 0.2, ... 0.9 columns)
        # We can plot 0.1-0.9 area
        if '0.1' in pred_df.columns and '0.9' in pred_df.columns:
             p_main.varea(x='timestamp', y1='0.1', y2='0.9', source=src, fill_color=color, alpha=0.4, legend_label=f"80% KI {label_suffix}")

    # Plot Future Forecast
    plot_single_forecast(future_pred, colors[-1], "(Zukunft)")

    # Plot Backtests
    for i, bp in enumerate(backtest_preds):
        # We fade the color for older backtests? or just use same
        col = colors[i % len(colors)]
        plot_single_forecast(bp, col, f"(Backtest -{i+1})")

    # Tools
    hover_main = HoverTool(
        tooltips=[
            ("Zeitpunkt", "@timestamp{%F %T}"),
            ("Messwert Wasser", "@{wassertemp}{0.0} °C"),
            ("Prognose", "@mean{0.0} °C"),
        ],
        formatters={'@timestamp': 'datetime'},
        mode='vline'
    )
    p_main.add_tools(hover_main)

    p_main.legend.location = "top_left"
    p_main.legend.click_policy = "hide"

    save(p_main, filename=output_filename, title="Eisbach Wassertemperatur Prognose")
    print(f"Plot saved to {output_filename}")
