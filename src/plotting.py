from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, LinearAxis, Range1d
from bokeh.layouts import column
from bokeh.io import save
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from cycler import cycler

def plot_forecasts(data: pd.DataFrame, future_pred: pd.DataFrame, backtest_preds: list, output_filename="eisbach_plot.html"):
    """
    Generates an interactive Bokeh plot.
    data: Historical data (dataframe with 'timestamp', 'wassertemp', 'lufttemperatur_c', 'niederschlag_mm')
    future_pred: Main forecast dataframe
    backtest_preds: List of backtest forecast dataframes
    """

    # Prepare data for plotting
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
    colors = ['#3385ff', '#6f42c1', '#d9534f', '#5cb85c'] # Future, Backtest-1, -2, -3

    # Helper to plot one forecast
    def plot_single_forecast(pred_df, color, label_suffix=""):
        # Reset index if multi-index (item_id, timestamp)
        if 'timestamp' not in pred_df.columns:
            pred_df = pred_df.reset_index()

        src = ColumnDataSource(pred_df)

        # Mean
        p_main.line(x='timestamp', y='mean', source=src, line_width=2.5, color=color, legend_label=f"Prognose {label_suffix}")

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

def save_static_plot(data: pd.DataFrame, future_pred: pd.DataFrame, backtest_preds: list, output_filename="eisbach_new.png"):
    """
    Generates a static Matplotlib plot (PNG).
    """

    # Style
    primer = {
      "theme_color": "#231F20",
      "style": {
        "lines.linewidth": 1.0, "lines.linestyle": "-", "font.family": "sans-serif",
        "font.size": 10, "text.color": "#231F20", "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#231F20", "axes.linewidth": 0.8, "axes.grid": True,
        "axes.labelsize": 10, "axes.labelweight": "normal", "axes.labelcolor": "#231F20",
        "axes.prop_cycle": cycler(color=["#1771F1", "#F85C50", "#35D073", "#FFC11E", "#8E44AD"]),
        "xtick.major.size": 2, "xtick.minor.size": 1, "xtick.major.width": 0.8,
        "xtick.minor.width": 0.6, "xtick.major.top": True, "xtick.major.bottom": True,
        "xtick.minor.top": True, "xtick.minor.bottom": True, "xtick.color": "#231F20", "xtick.labelsize": 8,
        "ytick.major.size": 2, "ytick.minor.size": 1, "ytick.major.width": 0.8,
        "ytick.minor.width": 0.6, "ytick.color": "#231F20", "ytick.major.left": True,
        "ytick.major.right": True, "ytick.minor.left": True, "ytick.minor.right": True,
        "grid.color": "#231F20", "grid.linestyle": ":", "grid.linewidth": 0.4,
        "grid.alpha": 1.0, "legend.frameon": False, "legend.edgecolor": "#231F20",
        "figure.figsize": [15, 9], "figure.dpi": 96, "figure.facecolor": "#FFFFFF",
        "figure.edgecolor": "#FFFFFF"
      }
    }
    plt.rcParams.update(primer['style'])

    fig, ax = plt.subplots()
    colors = primer['style']['axes.prop_cycle'].by_key()['color']

    # Plot Historical
    ax.plot(data['timestamp'], data['wassertemp'], label='Historical Wassertemp', color='black', linestyle='--')

    if 'lufttemperatur_c' in data.columns:
        ax.plot(data['timestamp'], data['lufttemperatur_c'], label='Air Temp (DWD)', color='purple', linestyle=':', linewidth=1.5, alpha=0.6)

    def plot_one(df, label, color):
        if isinstance(df, pd.DataFrame):
            # Ensure timestamp is available as column or index
            if 'timestamp' in df.columns:
                x = df['timestamp']
            else:
                x = df.index.get_level_values('timestamp')

            y = df['mean']
            ax.plot(x, y, label=label, color=color)

            # Fill between
            # We assume 0.1 and 0.9 exist
            if '0.1' in df.columns and '0.9' in df.columns:
                ax.fill_between(x, df['0.1'], df['0.9'], alpha=0.2, color=color, lw=0)

    # Plot Future
    plot_one(future_pred, 'Forecast Wassertemp', colors[0])

    # Plot Backtests
    for i, bp in enumerate(backtest_preds):
        col = colors[(i + 1) % len(colors)]
        plot_one(bp, f'Backtest -{i+1}', col)

    ax.set_title('Eisbach Wassertemperatur Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.legend(loc='upper left')

    # Adjust limits to show relevant area
    if not backtest_preds:
        start_plot = data['timestamp'].max() - timedelta(days=7)
    else:
        # Find earliest backtest start
        earliest_list = [bp['start_timestamp'].min() for bp in backtest_preds if 'start_timestamp' in bp and not bp.empty]
        if earliest_list:
            earliest = min(earliest_list)
            if isinstance(earliest, pd.Series): earliest = earliest.min()
            start_plot = earliest - timedelta(days=2) # a bit of context
        else:
            start_plot = data['timestamp'].max() - timedelta(days=7)

    # Ensure start_plot is timestamp
    if isinstance(start_plot, pd.Series): start_plot = start_plot.min()

    # Set xlim if start_plot is valid
    if start_plot is not None:
        ax.set_xlim(left=start_plot)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Static plot saved to {output_filename}")
