"""Visualization utilities for StockPredictionAI Pro.

All functions save plots to *outputs/* and optionally show them.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt


_OUT = "outputs"


def _ensure_dir():
    os.makedirs(_OUT, exist_ok=True)


# ------------------------------------------------------------------
# 1. Predictions vs Real with quantile bands
# ------------------------------------------------------------------

def plot_predictions(y_true, y_pred, y_q=None, quantiles=(0.1, 0.5, 0.9),
                     title="Predicted vs Real Price Changes",
                     filename="pred_vs_real.png"):
    """Plot predicted vs actual values with optional quantile uncertainty bands.

    Args:
        y_true: 1-D array of true values.
        y_pred: 1-D array of predicted values.
        y_q: 2-D array [N, Q] of quantile predictions (optional).
        quantiles: tuple of quantile levels.
        title: plot title.
        filename: output filename inside outputs/.
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(14, 5), dpi=100)
    n = len(y_true)
    x = np.arange(n)

    ax.plot(x, y_true, label="Actual", linewidth=1.2, alpha=0.9)
    ax.plot(x, y_pred, label="Predicted", linewidth=1.0, alpha=0.8)

    if y_q is not None and y_q.shape[1] >= 2:
        q_low = y_q[:, 0]
        q_high = y_q[:, -1]
        q_labels = [f"q{int(q * 100)}" for q in quantiles]
        ax.fill_between(x, q_low, q_high, alpha=0.2, color="orange",
                        label=f"{q_labels[0]}-{q_labels[-1]} band")

    ax.set_xlabel("Sample")
    ax.set_ylabel("Delta Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, filename))
    plt.close(fig)


# ------------------------------------------------------------------
# 2. Training curves
# ------------------------------------------------------------------

def plot_training_curves(g_losses, d_losses, val_maes=None,
                         filename="training_curves.png"):
    """Plot generator / discriminator loss curves over epochs.

    Args:
        g_losses: list of generator losses per epoch.
        d_losses: list of discriminator losses per epoch.
        val_maes: optional list of validation MAE per epoch.
        filename: output filename inside outputs/.
    """
    _ensure_dir()
    epochs = np.arange(1, len(g_losses) + 1)
    n_axes = 2 if val_maes is None else 3
    fig, axes = plt.subplots(1, n_axes, figsize=(6 * n_axes, 4), dpi=100)

    axes[0].plot(epochs, g_losses, label="G loss")
    axes[0].set_title("Generator Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, d_losses, label="D loss", color="tab:orange")
    axes[1].set_title("Discriminator Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    if val_maes is not None and len(val_maes) == len(g_losses):
        axes[2].plot(epochs, val_maes, label="Val MAE", color="tab:green")
        axes[2].set_title("Validation MAE")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, filename))
    plt.close(fig)


# ------------------------------------------------------------------
# 3. Technical indicators dashboard
# ------------------------------------------------------------------

def plot_technical_indicators(panel, ticker, last_days=400,
                              filename="technical_indicators.png"):
    """Plot price with SMA/EMA/Bollinger and MACD subplot.

    Args:
        panel: DataFrame with columns from build_panel (sma7, sma21, bb_upper, etc.).
        ticker: target ticker column name.
        last_days: number of trailing days to display.
        filename: output filename inside outputs/.
    """
    _ensure_dir()
    df = panel.iloc[-last_days:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), dpi=100,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df.index, df[ticker], label="Close", linewidth=1.2)
    if "sma7" in df.columns:
        ax1.plot(df.index, df["sma7"], label="SMA 7", linestyle="--", alpha=0.7)
    if "sma21" in df.columns:
        ax1.plot(df.index, df["sma21"], label="SMA 21", linestyle="--", alpha=0.7)
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        ax1.fill_between(df.index, df["bb_lower"], df["bb_upper"],
                         alpha=0.15, color="gray", label="Bollinger Bands")
    ax1.set_title(f"Technical Indicators -- {ticker} (last {last_days} days)")
    ax1.set_ylabel("USD")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    if "macd" in df.columns and "macd_signal" in df.columns:
        ax2.plot(df.index, df["macd"], label="MACD", linewidth=1.0)
        ax2.plot(df.index, df["macd_signal"], label="Signal", linewidth=1.0)
        if "macd_hist" in df.columns:
            ax2.bar(df.index, df["macd_hist"], label="Histogram",
                    alpha=0.4, width=1, color="gray")
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_title("MACD")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, filename))
    plt.close(fig)


# ------------------------------------------------------------------
# 4. Fourier decomposition
# ------------------------------------------------------------------

def plot_fourier_components(series, components=(3, 6, 9),
                            filename="fourier_components.png"):
    """Overlay multiple Fourier approximations on the original series.

    Args:
        series: pd.Series of the target price.
        components: tuple of k values for Fourier reconstruction.
        filename: output filename inside outputs/.
    """
    from ..features.fourier import fourier_approx

    _ensure_dir()
    fig, ax = plt.subplots(figsize=(14, 5), dpi=100)
    ax.plot(series.index, series.values, label="Original", linewidth=1.2, alpha=0.8)
    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, k in enumerate(components):
        approx = fourier_approx(series, k)
        ax.plot(series.index, approx.values,
                label=f"FFT k={k}", linewidth=1.0,
                linestyle="--", color=colors[i % len(colors)])
    ax.set_title("Fourier Transform Components")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, filename))
    plt.close(fig)
