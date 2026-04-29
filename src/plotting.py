import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

def plot_forecasts(df_long, df_wetter, df_inference, df_inference_backtest_96_corr, df_inference_backtest_192_corr, df_inference_backtest_288_corr):
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
        "figure.figsize": [15, 9], "figure.dpi": 96, "figure.facecolor": "#FFFFFF",
        "figure.edgecolor": "#FFFFFF"
      }
    }
    plt.rcParams.update(primer['style'])

    # Define plotting parameters
    channel = 'wassertemp'
    fig, ax = plt.subplots()

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

    # Plot all four forecasts for wassertemp
    plot_forecast(df_inference, 'Forecast Wassertemp', colors[0])
    plot_forecast(df_inference_backtest_96_corr, 'Backtest -96h', colors[1])
    plot_forecast(df_inference_backtest_192_corr, 'Backtest -192h', colors[2])
    plot_forecast(df_inference_backtest_288_corr, 'Backtest -288h', colors[3])

    # Plot the unshifted air temperature forecast on the same axis
    ax.plot(df_wetter.index, df_wetter['lufttemperatur_c'], label='Air Temp (DWD)', color='purple', linestyle=':', linewidth=1.5, alpha=0.6)

    # Add invisible artists for the legend
    for j, label in enumerate(quantile_labels):
        ax.fill_between([], [], [], color='gray', alpha=alphas[j], label=label)

    # Final plot adjustments
    plot_start_date = df_inference_backtest_288_corr.index.min()
    plot_end_date = df_inference.index.max() # Set the end date to the last point of the main forecast
    ax.set_xlim(left=plot_start_date, right=plot_end_date) # Set both start and end limits

    # --- CHIRURGISCHER EINGRIFF -BLOCK START ---
    # Wir holen uns die Wetter-Daten, die WIRKLICH im aktuellen Zoom-Fenster liegen
    visible_wetter = df_wetter.loc[(df_wetter.index >= plot_start_date) & (df_wetter.index <= plot_end_date), 'lufttemperatur_c']

    # Wir berechnen das Min/Max aus sichtbarem Wetter UND der Vorhersage (damit nichts abgeschnitten wird)
    # Hinweis: Wir nehmen q0.01 und q0.99 der Vorhersage für die Sicherheit
    y_view_min = min(visible_wetter.min(), df_inference[f"{channel}_q0.01"].min())
    y_view_max = max(visible_wetter.max(), df_inference[f"{channel}_q0.99"].max())

    # Wir setzen die Y-Achse neu mit einem kleinen Puffer (z.B. 1 Grad oben/unten)
    ax.set_ylim(y_view_min - 0.5, y_view_max + 0.5)
    # --- CHIRURGISCHER EINFÜGE-BLOCK ENDE ---

    # Main Y-axis
    ax.set_title(f'Eisbach - backtesting and forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperatur (°C)')

    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left')

    # --- Save the figure ---
    # Save as a high-resolution PNG file
    file_path = 'eisbach_new.png'
    plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Plot saved to: {file_path}")

    # No plt.show() needed in the automated script
