import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cycler import cycler

def plot_forecasts(df_long, df_wetter, df_inference, df_inference_backtest_96_corr, df_inference_backtest_192_corr, df_inference_backtest_288_corr, timestamp_str=""):
    # --- Final Plotting ---

    # Define the style primer
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
        "figure.figsize": [12, 10], "figure.dpi": 96, "figure.facecolor": "#FFFFFF",
        "figure.edgecolor": "#FFFFFF"
      }
    }
    plt.rcParams.update(primer['style'])

    # Define plotting parameters
    channel = 'wassertemp'
    fig, ax = plt.subplots()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

    colors = primer['style']['axes.prop_cycle'].by_key()['color']
    quantile_pairs = [(0.01, 0.99), (0.05, 0.95), (0.25, 0.75)]
    alphas = [0.1, 0.15, 0.2]
    quantile_labels = ['q0.01-q0.99', 'q0.05-q0.95', 'q0.25-q0.75']

    # Plot Historical Data for wassertemp on the main axis
    historical_data = df_long[df_long['cols'] == channel]
    ax.plot(historical_data['date'], historical_data['data'], label='Historical Wassertemp', color='black', linestyle='--')

    # Helper function to plot a forecast
    def plot_forecast(df_forecast, label, color):
        median_col = f"{channel}_q0.5"
        ax.plot(df_forecast.index, df_forecast[median_col], label=label, color=color)
        for j, (q_low, q_high) in enumerate(quantile_pairs):
            col_low = f"{channel}_q{q_low}"
            col_high = f"{channel}_q{q_high}"
            if col_low in df_forecast.columns and col_high in df_forecast.columns:
                ax.fill_between(df_forecast.index, df_forecast[col_low], df_forecast[col_high],
                                alpha=alphas[j], color=color, lw=0)

    # Add invisible artists for the legend (added once)
    for j, label in enumerate(quantile_labels):
        ax.fill_between([], [], [], color='gray', alpha=alphas[j], label=label)

    # ----------------------------------------------------
    # Plot 1: Prediction ONLY
    # ----------------------------------------------------
    plot_forecast(df_inference, f'Forecast Wassertemp ({timestamp_str})', colors[0])

    # Plot the unshifted air temperature forecast on the same axis
    air_temp_line = ax.plot(df_wetter.index, df_wetter['lufttemperatur_c'], label='Air Temp (DWD)', color='purple', linestyle=':', linewidth=1.5, alpha=0.6)

    # Calculate view window for Prediction ONLY plot
    pred_only_start_date = df_inference.index.min() - pd.Timedelta(days=1) # show 1 day of history
    pred_only_end_date = df_inference.index.max()
    ax.set_xlim(left=pred_only_start_date, right=pred_only_end_date)

    visible_wetter = df_wetter.loc[(df_wetter.index >= pred_only_start_date) & (df_wetter.index <= pred_only_end_date), 'lufttemperatur_c']
    visible_history = historical_data.loc[(historical_data['date'] >= pred_only_start_date) & (historical_data['date'] <= pred_only_end_date), 'data']

    inference_min = df_inference[f"{channel}_q0.01"].min()
    inference_max = df_inference[f"{channel}_q0.99"].max()

    y_view_min = inference_min
    y_view_max = inference_max
    if not visible_wetter.empty:
        y_view_min = min(y_view_min, visible_wetter.min())
        y_view_max = max(y_view_max, visible_wetter.max())
    if not visible_history.empty:
        y_view_min = min(y_view_min, visible_history.min())
        y_view_max = max(y_view_max, visible_history.max())

    ax.set_ylim(y_view_min - 0.5, y_view_max + 0.5)

    # Formatting and titles
    ax.set_title(f'Eisbach - forecast ({timestamp_str})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperatur (°C)')

    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left')

    # Annotate daily maximum values (Max Median)
    # Convert index to timezone aware (Europe/Berlin) for grouping and text labels
    df_local = df_inference.copy()
    if df_local.index.tzinfo is None:
        df_local.index = df_local.index.tz_localize('UTC')
    df_local.index = df_local.index.tz_convert('Europe/Berlin')

    median_col = f"{channel}_q0.5"
    for date, group in df_local.groupby(df_local.index.date):
        if median_col in group.columns:
            # Find row with maximum median temperature
            max_row = group.loc[group[median_col].idxmax()]
            max_val = max_row[median_col]
            # Convert timestamp back to UTC for x-coordinate in plot (since main plot is UTC)
            max_time_utc = max_row.name.tz_convert('UTC')

            # Extract local time string
            local_time_str = max_row.name.strftime('%H:%M')

            # Annotate with an arrow
            ax.annotate(f"Max: {max_val:.1f}°C\n{local_time_str}",
                        xy=(max_time_utc, max_val),
                        xytext=(0, 20), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.0, alpha=0.7))

    # Save Prediction ONLY Plot
    file_path_pred = f'Prediction_{timestamp_str}.png' if timestamp_str else 'Prediction.png'
    plt.savefig(file_path_pred, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Plot saved to: {file_path_pred}")

    # ----------------------------------------------------
    # Plot 2: Prediction AND Backtests
    # ----------------------------------------------------
    # Now plot backtests over the existing plot
    plot_forecast(df_inference_backtest_96_corr, f'Backtest -96h ({timestamp_str})', colors[1])
    plot_forecast(df_inference_backtest_192_corr, f'Backtest -192h ({timestamp_str})', colors[2])
    plot_forecast(df_inference_backtest_288_corr, f'Backtest -288h ({timestamp_str})', colors[3])

    # Final plot adjustments for backtest
    plot_start_date = df_inference_backtest_288_corr.index.min()
    plot_end_date = df_inference.index.max() # Set the end date to the last point of the main forecast
    ax.set_xlim(left=plot_start_date, right=plot_end_date) # Set both start and end limits

    visible_wetter_bt = df_wetter.loc[(df_wetter.index >= plot_start_date) & (df_wetter.index <= plot_end_date), 'lufttemperatur_c']

    y_view_min_bt = inference_min
    y_view_max_bt = inference_max

    for df_bt in [df_inference_backtest_96_corr, df_inference_backtest_192_corr, df_inference_backtest_288_corr]:
        bt_min = df_bt[f"{channel}_q0.01"].min()
        bt_max = df_bt[f"{channel}_q0.99"].max()
        y_view_min_bt = min(y_view_min_bt, bt_min)
        y_view_max_bt = max(y_view_max_bt, bt_max)

    if not visible_wetter_bt.empty:
        y_view_min_bt = min(visible_wetter_bt.min(), y_view_min_bt)
        y_view_max_bt = max(visible_wetter_bt.max(), y_view_max_bt)

    ax.set_ylim(y_view_min_bt - 0.5, y_view_max_bt + 0.5)

    # Main Y-axis
    ax.set_title(f'Eisbach - backtesting and forecast ({timestamp_str})')

    # Refresh legends to include backtests
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left')

    # Save Backtest Plot
    file_path_backtest = f'Prediction_Backtest_{timestamp_str}.png' if timestamp_str else 'Prediction_Backtest.png'
    plt.savefig(file_path_backtest, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Plot saved to: {file_path_backtest}")

    # Close figure
    plt.close(fig)
