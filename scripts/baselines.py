"""Naive baseline benchmarks for comparison with GAN predictions.

Computes several simple baselines on the same test data and prints a
comparison table. This is essential for validating that the GAN model
actually outperforms trivial strategies.

Usage:
    python scripts/baselines.py --ticker GS --start 2010-01-01 --end 2018-12-31
    python scripts/baselines.py --predictions outputs/test_predictions.csv
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import mae, mape, direction_accuracy, pinball_loss, smape


# ---------------------------------------------------------------------------
# Baseline strategies (all operate on delta-price targets)
# ---------------------------------------------------------------------------

def naive_persistence(y_true: np.ndarray) -> np.ndarray:
    """Predict delta = 0 (tomorrow's price = today's price)."""
    return np.zeros_like(y_true)


def random_walk(y_true: np.ndarray, seed: int = 42) -> np.ndarray:
    """Predict random direction with same std as true deltas."""
    rng = np.random.RandomState(seed)
    return rng.randn(len(y_true)) * np.std(y_true)


def mean_reversion(y_true: np.ndarray) -> np.ndarray:
    """Predict negative of previous delta (contrarian)."""
    pred = np.zeros_like(y_true)
    pred[1:] = -y_true[:-1]
    return pred


def momentum_baseline(y_true: np.ndarray) -> np.ndarray:
    """Predict same direction as previous delta (trend following)."""
    pred = np.zeros_like(y_true)
    pred[1:] = y_true[:-1]
    return pred


def sma_baseline(y_true: np.ndarray, window: int = 5) -> np.ndarray:
    """Predict SMA of recent deltas."""
    pred = np.zeros_like(y_true)
    for i in range(window, len(y_true)):
        pred[i] = np.mean(y_true[i - window:i])
    return pred


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard metrics for a baseline."""
    # Create dummy quantile predictions (repeat point prediction for 3 quantiles)
    y_q = np.column_stack([y_pred, y_pred, y_pred])
    return {
        "Model": name,
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "DirAcc": direction_accuracy(y_true, y_pred),
        "PinballLoss": pinball_loss(y_true, y_q, (0.1, 0.5, 0.9)),
    }


def run_all_baselines(y_true: np.ndarray, gan_pred: np.ndarray = None,
                      gan_q: np.ndarray = None) -> pd.DataFrame:
    """Run all baseline strategies and optionally include GAN results.

    Returns a DataFrame comparison table.
    """
    results = []

    baselines = {
        "Naive (delta=0)": naive_persistence(y_true),
        "Random Walk": random_walk(y_true),
        "Mean Reversion": mean_reversion(y_true),
        "Momentum (prev delta)": momentum_baseline(y_true),
        "SMA(5) delta": sma_baseline(y_true, 5),
        "SMA(10) delta": sma_baseline(y_true, 10),
    }

    for name, pred in baselines.items():
        results.append(evaluate_baseline(name, y_true, pred))

    if gan_pred is not None:
        gan_result = {
            "Model": "GAN",
            "MAE": mae(y_true, gan_pred),
            "MAPE": mape(y_true, gan_pred),
            "sMAPE": smape(y_true, gan_pred),
            "DirAcc": direction_accuracy(y_true, gan_pred),
        }
        if gan_q is not None:
            gan_result["PinballLoss"] = pinball_loss(y_true, gan_q, (0.1, 0.5, 0.9))
        else:
            y_q_dummy = np.column_stack([gan_pred, gan_pred, gan_pred])
            gan_result["PinballLoss"] = pinball_loss(y_true, y_q_dummy, (0.1, 0.5, 0.9))
        results.append(gan_result)

    df = pd.DataFrame(results)
    # Sort by MAE ascending
    df = df.sort_values("MAE").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run baseline benchmarks")
    parser.add_argument("--predictions", default="outputs/test_predictions.csv",
                        help="Path to GAN predictions CSV (must have y_true, y_pred columns)")
    parser.add_argument("--output", default="outputs/baselines_comparison.csv",
                        help="Path to save comparison table")
    args = parser.parse_args()

    if not os.path.exists(args.predictions):
        print(f"ERROR: Predictions file not found: {args.predictions}")
        print("Run the GAN training first:  python -m src.train --ticker GS")
        sys.exit(1)

    df_pred = pd.read_csv(args.predictions)
    y_true = df_pred["y_true"].values
    y_pred = df_pred["y_pred"].values

    # Load quantile predictions if available
    q_cols = [c for c in df_pred.columns if c.startswith("q")]
    gan_q = df_pred[q_cols].values if len(q_cols) >= 2 else None

    print(f"Loaded {len(y_true)} test samples from {args.predictions}\n")

    table = run_all_baselines(y_true, gan_pred=y_pred, gan_q=gan_q)

    # Pretty print
    print("=" * 80)
    print("BASELINE COMPARISON TABLE")
    print("=" * 80)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 80)

    # Highlight GAN vs best naive
    gan_row = table[table["Model"] == "GAN"]
    naive_row = table[table["Model"] == "Naive (delta=0)"]
    if not gan_row.empty and not naive_row.empty:
        gan_mae = gan_row["MAE"].iloc[0]
        naive_mae = naive_row["MAE"].iloc[0]
        improvement = (naive_mae - gan_mae) / naive_mae * 100
        print(f"\nGAN vs Naive Persistence:  MAE improvement = {improvement:+.1f}%")
        gan_dir = gan_row["DirAcc"].iloc[0]
        print(f"GAN DirAcc = {gan_dir:.3f}  (Naive = 0.500)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
