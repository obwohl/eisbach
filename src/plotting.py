import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from datetime import timedelta

def save_static_plot(data: pd.DataFrame,
                     future_pred_cov: pd.DataFrame, backtest_preds_cov: list,
                     future_pred_naive: pd.DataFrame = None, backtest_preds_naive: list = None,
                     output_filename="eisbach_new.png"):
    """
    Generates a static Matplotlib plot (PNG) using the Primer style and surgical Y-axis scaling.
    Plots both Covariate (Main) and Naive models if provided.

    Covariate Model: Solid Line, Filled Intervals.
    Naive Model: Dashed Line, Dotted Interval Lines (No Fill).
    """

    # 1. Define Style
    primer = {
        "theme_color": "#231F20",
        "style": {
            "lines.linewidth": 1.5, "lines.linestyle": "-", "font.family": "sans-serif",
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

    # 2. Prepare Data
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp')

    # 3. Plot Historical Wassertemp
    ax.plot(data['timestamp'], data['wassertemp'], label='Historical Wassertemp', color='black', linestyle='-', linewidth=1)

    # 4. Helper to plot forecast
    def plot_forecast_df(df, label, color, is_naive=False):
        if df is None or df.empty:
            return

        # Normalize columns
        if 'timestamp' in df.columns:
            x = df['timestamp']
        else:
            x = df.index.get_level_values('timestamp')

        y = df['mean']

        if is_naive:
            # Naive: Dashed Line, Dotted Interval Lines (No Fill)
            ax.plot(x, y, label=label, color=color, linestyle='--')
            if '0.1' in df.columns and '0.9' in df.columns:
                ax.plot(x, df['0.1'], color=color, linestyle=':', linewidth=1, alpha=0.8)
                ax.plot(x, df['0.9'], color=color, linestyle=':', linewidth=1, alpha=0.8)
        else:
            # Covariate: Solid Line, Filled Interval
            ax.plot(x, y, label=label, color=color, linestyle='-')
            if '0.1' in df.columns and '0.9' in df.columns:
                ax.fill_between(x, df['0.1'], df['0.9'], color=color, alpha=0.2, lw=0, label=f"{label} 80% CI")

    # 5. Plot Forecasts
    backtest_labels = ['Backtest -96h', 'Backtest -192h', 'Backtest -288h']

    # Future
    plot_forecast_df(future_pred_cov, 'Forecast (Covariates)', colors[0], is_naive=False)
    if future_pred_naive is not None:
        plot_forecast_df(future_pred_naive, 'Forecast (Naive)', colors[0], is_naive=True)

    # Backtests
    for i, pred_df in enumerate(backtest_preds_cov):
        col_idx = (i + 1) % len(colors)
        col = colors[col_idx]
        lbl = backtest_labels[i] if i < len(backtest_labels) else f"Backtest {i+1}"

        plot_forecast_df(pred_df, f"{lbl} (Cov)", col, is_naive=False)

        if backtest_preds_naive and i < len(backtest_preds_naive):
             pred_naive = backtest_preds_naive[i]
             plot_forecast_df(pred_naive, f"{lbl} (Naive)", col, is_naive=True)

    # 6. Plot Air Temp (Covariate)
    if 'lufttemperatur_c' in data.columns:
        ax.plot(data['timestamp'], data['lufttemperatur_c'], label='Air Temp (DWD)', color='purple', linestyle=':', linewidth=1.5, alpha=0.6)

    # 7. Secondary Axis for Precipitation
    if 'niederschlag_mm' in data.columns:
        ax2 = ax.twinx()
        ax2.set_ylabel('Niederschlag [mm]', color='navy')
        ax2.bar(data['timestamp'], data['niederschlag_mm'], color='navy', alpha=0.3, width=0.04, label='Niederschlag')
        ax2.tick_params(axis='y', labelcolor='navy')
        max_precip = data['niederschlag_mm'].max()
        if pd.notna(max_precip) and max_precip > 0:
            ax2.set_ylim(0, max_precip * 3)
        ax.plot([], [], color='navy', alpha=0.3, linewidth=5, label='Niederschlag')

    # 8. Set Limits
    plot_start_date = None
    if backtest_preds_cov:
        last_bt = backtest_preds_cov[-1]
        if 'start_timestamp' in last_bt.columns:
             plot_start_date = last_bt['start_timestamp'].iloc[0]

    if plot_start_date is None:
         plot_start_date = data['timestamp'].max() - timedelta(hours=288)

    if not isinstance(plot_start_date, pd.Timestamp):
        plot_start_date = pd.to_datetime(plot_start_date)

    plot_end_date = future_pred_cov['timestamp'].max() if (future_pred_cov is not None) else data['timestamp'].max()
    ax.set_xlim(left=plot_start_date, right=plot_end_date)

    # Surgical Y Zoom
    mask_data = (data['timestamp'] >= plot_start_date) & (data['timestamp'] <= plot_end_date)
    visible_weather = data.loc[mask_data, 'lufttemperatur_c'] if 'lufttemperatur_c' in data.columns else pd.Series(dtype=float)
    visible_water = data.loc[mask_data, 'wassertemp']

    min_candidates = []
    max_candidates = []

    if not visible_weather.empty:
        min_candidates.append(visible_weather.min())
        max_candidates.append(visible_weather.max())
    if not visible_water.empty:
         min_candidates.append(visible_water.min())
         max_candidates.append(visible_water.max())

    all_preds = [future_pred_cov] + backtest_preds_cov
    if future_pred_naive is not None: all_preds.append(future_pred_naive)
    if backtest_preds_naive: all_preds.extend(backtest_preds_naive)

    for df in all_preds:
        if df is None or df.empty: continue
        # Only consider mean for zoom to avoid outliers in wide CIs blowing up the scale?
        # Or consider 0.1/0.9 bounds. User wants "nothing cut off".
        if '0.1' in df.columns: min_candidates.append(df['0.1'].min())
        if '0.9' in df.columns: max_candidates.append(df['0.9'].max())
        if 'mean' in df.columns:
            min_candidates.append(df['mean'].min())
            max_candidates.append(df['mean'].max())

    if min_candidates and max_candidates:
        y_view_min = min([x for x in min_candidates if pd.notna(x)])
        y_view_max = max([x for x in max_candidates if pd.notna(x)])
        ax.set_ylim(y_view_min - 1.0, y_view_max + 1.0)

    # 9. Legend
    ax.set_title(f'Eisbach Forecast: Covariates vs Naive')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperatur (°C)')

    lines, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(lines, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc='upper left')

    # 10. Save
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Static plot saved to {output_filename}")
