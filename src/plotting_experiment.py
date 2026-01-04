import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os

def plot_experiment_window(history_df, window_df,
                           pred_cov, pred_naive,
                           loss_cov, loss_naive,
                           window_info, output_filename):
    """
    Plots a specific experiment window.
    Covariate Model: Solid Line, Filled Intervals.
    Naive Model: Dashed Line, Dotted Interval Lines (No Fill).
    """

    # 1. Style
    primer = {
        "theme_color": "#231F20",
        "style": {
            "lines.linewidth": 1.5, "lines.linestyle": "-", "font.family": "sans-serif",
            "font.size": 10, "text.color": "#231F20", "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#231F20", "axes.linewidth": 0.8, "axes.grid": True,
            "axes.labelsize": 10, "axes.labelweight": "normal", "axes.labelcolor": "#231F20",
            "axes.prop_cycle": cycler(color=["#1771F1", "#F85C50", "#35D073", "#FFC11E", "#8E44AD"]),
            "figure.figsize": [12, 6], "figure.dpi": 100
        }
    }
    plt.rcParams.update(primer['style'])

    fig, ax = plt.subplots()
    colors = primer['style']['axes.prop_cycle'].by_key()['color']

    # 2. Plot Context
    ax.plot(window_df['timestamp'], window_df['wassertemp'], label='Truth', color='black', linewidth=2)
    ax.plot(history_df['timestamp'], history_df['wassertemp'], color='black', linestyle=':', label='History')

    # 3. Plot Predictions

    # Covariate (Primary)
    if pred_cov is not None:
        ax.plot(pred_cov['timestamp'], pred_cov['mean'], label=f'Covariate (Loss: {loss_cov:.2f})', color=colors[0], linestyle='-')
        if '0.1' in pred_cov.columns and '0.9' in pred_cov.columns:
            ax.fill_between(pred_cov['timestamp'], pred_cov['0.1'], pred_cov['0.9'], color=colors[0], alpha=0.2, label="Covariate 80% CI")

    # Naive (Secondary)
    if pred_naive is not None:
        ax.plot(pred_naive['timestamp'], pred_naive['mean'], label=f'Naive (Loss: {loss_naive:.2f})', color=colors[1], linestyle='--')
        if '0.1' in pred_naive.columns and '0.9' in pred_naive.columns:
            # Dotted lines for CI bounds, NO fill
            ax.plot(pred_naive['timestamp'], pred_naive['0.1'], color=colors[1], linestyle=':', linewidth=1.5, label="Naive 80% CI")
            ax.plot(pred_naive['timestamp'], pred_naive['0.9'], color=colors[1], linestyle=':', linewidth=1.5)

    # 4. Title & Labels
    start_str = window_info['start_time'].strftime('%Y-%m-%d')
    ax.set_title(f"High Variance Window: {start_str} (Var: {window_info['variance']:.2f})")
    ax.set_ylabel("Wassertemperatur [°C]")
    ax.legend(loc='best')

    # 5. Secondary Axis for Air Temp
    if 'lufttemperatur_c' in window_df.columns:
        ax2 = ax.twinx()
        ax2.plot(window_df['timestamp'], window_df['lufttemperatur_c'], color='purple', alpha=0.3, linestyle='-.', label='Air Temp')
        ax2.set_ylabel("Air Temp [°C]", color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved plot to {output_filename}")
