import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cycler import cycler
from scipy.signal import find_peaks
import plotly.graph_objects as go

def generate_html_plot(df_long_plot, df_wetter_plot, df_inference_plot, timestamp_str, peaks, median_col, channel):
    """
    Generates an interactive HTML plot using Plotly, optimized for mobile viewing.
    """
    fig = go.Figure()

    # Historical Data
    historical_data = df_long_plot[df_long_plot['cols'] == channel]
    fig.add_trace(go.Scatter(
        x=historical_data['date'], y=historical_data['data'],
        mode='lines', name=f'Historical {channel.capitalize()}',
        line=dict(color='black', dash='dash')
    ))

    # 99% Quantile Band
    if f'{channel}_q0.01' in df_inference_plot.columns and f'{channel}_q0.99' in df_inference_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_inference_plot.index.tolist() + df_inference_plot.index.tolist()[::-1],
            y=df_inference_plot[f'{channel}_q0.99'].tolist() + df_inference_plot[f'{channel}_q0.01'].tolist()[::-1],
            fill='toself', fillcolor='rgba(23, 113, 241, 0.1)', line=dict(color='rgba(255,255,255,0)'),
            name='1%-99% Quantile'
        ))

    # 75% Quantile Band
    if f'{channel}_q0.25' in df_inference_plot.columns and f'{channel}_q0.75' in df_inference_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_inference_plot.index.tolist() + df_inference_plot.index.tolist()[::-1],
            y=df_inference_plot[f'{channel}_q0.75'].tolist() + df_inference_plot[f'{channel}_q0.25'].tolist()[::-1],
            fill='toself', fillcolor='rgba(23, 113, 241, 0.2)', line=dict(color='rgba(255,255,255,0)'),
            name='25%-75% Quantile'
        ))

    # Main Forecast Median
    fig.add_trace(go.Scatter(
        x=df_inference_plot.index, y=df_inference_plot[median_col],
        mode='lines', name='Forecast Median',
        line=dict(color='#1771F1', width=2)
    ))

    # Air Temperature
    fig.add_trace(go.Scatter(
        x=df_wetter_plot.index, y=df_wetter_plot['lufttemperatur_c'],
        mode='lines', name='Air Temp (DWD)',
        line=dict(color='purple', dash='dot', width=1.5)
    ))

    # Annotate Peaks
    for peak_idx in peaks:
        max_row = df_inference_plot.iloc[peak_idx]
        max_val = max_row[median_col]
        max_time_local = max_row.name
        local_time_str = max_time_local.strftime('%H:%M')

        fig.add_annotation(
            x=max_time_local, y=max_val,
            text=f"Max: {max_val:.1f}°C<br>{local_time_str}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            ax=0, ay=-40, bgcolor="white", bordercolor="gray", opacity=0.8
        )

    # Calculate limits
    pred_only_start_date = df_inference_plot.index.min() - pd.Timedelta(days=1)
    pred_only_end_date = df_inference_plot.index.max()

    fig.update_xaxes(range=[pred_only_start_date, pred_only_end_date], title_text="Datum (Ortszeit / Europe/Berlin)")
    fig.update_yaxes(title_text="Temperatur (°C)")

    fig.update_layout(
        title=f"Eisbach Forecast ({timestamp_str})<br><sup>All times in Ortszeit / Europe/Berlin</sup>",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=80, b=20)
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Save to HTML
    html_file = f'Prediction_{timestamp_str}.html' if timestamp_str else 'Prediction.html'
    fig.write_html(html_file, include_plotlyjs='cdn')
    print(f"Interactive HTML plot saved to: {html_file}")

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

    # --- Timezone Conversion for Plotting ---
    # Convert all inputs to local time (Europe/Berlin) and then to naive datetime
    # to avoid matplotlib timezone issues while ensuring correct local hours on the x-axis.
    def to_local_naive(series_or_index):
        if hasattr(series_or_index, 'dt'):
            dt_obj = series_or_index.dt
            is_series = True
        else:
            dt_obj = series_or_index
            is_series = False

        if dt_obj.tz is None:
            if is_series:
                aware = series_or_index.dt.tz_localize('UTC')
            else:
                aware = series_or_index.tz_localize('UTC')
        else:
            aware = series_or_index

        if is_series:
            return aware.dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
        else:
            return aware.tz_convert('Europe/Berlin').tz_localize(None)

    df_long_plot = df_long.copy()
    df_long_plot['date'] = to_local_naive(df_long_plot['date'])

    df_wetter_plot = df_wetter.copy()
    df_wetter_plot.index = to_local_naive(df_wetter_plot.index)

    df_inference_plot = df_inference.copy()
    df_inference_plot.index = to_local_naive(df_inference_plot.index)

    df_inference_backtest_96_corr_plot = df_inference_backtest_96_corr.copy()
    df_inference_backtest_96_corr_plot.index = to_local_naive(df_inference_backtest_96_corr_plot.index)

    df_inference_backtest_192_corr_plot = df_inference_backtest_192_corr.copy()
    df_inference_backtest_192_corr_plot.index = to_local_naive(df_inference_backtest_192_corr_plot.index)

    df_inference_backtest_288_corr_plot = df_inference_backtest_288_corr.copy()
    df_inference_backtest_288_corr_plot.index = to_local_naive(df_inference_backtest_288_corr_plot.index)


    # Plot Historical Data for wassertemp on the main axis
    historical_data = df_long_plot[df_long_plot['cols'] == channel]
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
    plot_forecast(df_inference_plot, f'Forecast Wassertemp ({timestamp_str})', colors[0])

    # Plot the unshifted air temperature forecast on the same axis
    air_temp_line = ax.plot(df_wetter_plot.index, df_wetter_plot['lufttemperatur_c'], label='Air Temp (DWD)', color='purple', linestyle=':', linewidth=1.5, alpha=0.6)

    # Calculate view window for Prediction ONLY plot
    pred_only_start_date = df_inference_plot.index.min() - pd.Timedelta(days=1) # show 1 day of history
    pred_only_end_date = df_inference_plot.index.max()
    ax.set_xlim(left=pred_only_start_date, right=pred_only_end_date)

    visible_wetter = df_wetter_plot.loc[(df_wetter_plot.index >= pred_only_start_date) & (df_wetter_plot.index <= pred_only_end_date), 'lufttemperatur_c']
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
    ax.set_title(f'Eisbach - forecast ({timestamp_str})\n(All times in Ortszeit / Europe/Berlin)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperatur (°C)')

    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left')

    # Annotate maximum values (Max Median) using SciPy find_peaks
    median_col = f"{channel}_q0.5"
    annotation_artists = []
    peaks = []

    if median_col in df_inference_plot.columns:
        # Distance of at least 18 hours between peaks to avoid double-counting the same day
        # Prominence ensures we only get significant peaks, not tiny ripples
        peaks, _ = find_peaks(df_inference_plot[median_col], distance=18, prominence=0.2)

        for peak_idx in peaks:
            max_row = df_inference_plot.iloc[peak_idx]
            max_val = max_row[median_col]
            # Time is already localized naive from df_inference_plot
            max_time_local = max_row.name

            # Extract local time string
            local_time_str = max_time_local.strftime('%H:%M')

            # Annotate with an arrow
            annotation = ax.annotate(f"Max: {max_val:.1f}°C\n{local_time_str}",
                                     xy=(max_time_local, max_val),
                                     xytext=(0, 20), textcoords="offset points",
                                     ha='center', va='bottom', fontsize=9,
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                                     arrowprops=dict(arrowstyle="->", color="black", lw=1.0, alpha=0.7))
            annotation_artists.append(annotation)

    # Save Prediction ONLY Plot
    file_path_pred = f'Prediction_{timestamp_str}.png' if timestamp_str else 'Prediction.png'
    plt.savefig(file_path_pred, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Plot saved to: {file_path_pred}")

    # Generate HTML plot for better user interaction (mobile-friendly)
    if median_col in df_inference_plot.columns:
        try:
            generate_html_plot(df_long_plot, df_wetter_plot, df_inference_plot, timestamp_str, peaks, median_col, channel)
        except Exception as e:
            print(f"Warning: Failed to generate HTML plot: {e}")

    # Remove annotations from the plot before saving the backtest version
    for annotation in annotation_artists:
        annotation.remove()

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
    ax.set_title(f'Eisbach - backtesting and forecast ({timestamp_str})\n(All times in Ortszeit / Europe/Berlin)')

    # Refresh legends to include backtests
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left')

    # Save Backtest Plot
    file_path_backtest = f'Prediction_Backtest_{timestamp_str}.png' if timestamp_str else 'Prediction_Backtest.png'
    plt.savefig(file_path_backtest, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Plot saved to: {file_path_backtest}")

    # Close figure
    plt.close(fig)
