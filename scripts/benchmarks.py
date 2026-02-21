"""Stage 10: Compare GAN with naive baselines on MOEX SBER M5.

Benchmarks:
  1. Naive Persistence: predict delta=0 (tomorrow = today)
  2. Random Walk: predict random direction with same std
  3. SMA Baseline: predict delta = SMA(21) change
  4. Mean Reversion: predict delta = -last_delta

Usage:
    python scripts/benchmarks.py --ticker SBER --data_source local --data_path G:\\data2 --timeframe M5
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import build_panel_auto
from src.dataset import train_test_split_by_years
from src.train import (
    build_features_safe, fit_transforms, run_one_split,
    set_global_seed,
)
from src.utils.metrics import mae, mape, direction_accuracy, smape

SEED = 42


def compute_naive_benchmarks(test_deltas):
    """Compute metrics for all naive baselines given test target deltas."""
    y = np.array(test_deltas, dtype=np.float32)
    n = len(y)
    results = []

    naive_pred = np.zeros(n)
    results.append({
        "model": "Naive (delta=0)",
        "MAE": mae(y, naive_pred),
        "MAPE": mape(y, naive_pred),
        "sMAPE": smape(y, naive_pred),
        "DirAcc": direction_accuracy(y, naive_pred),
    })

    np.random.seed(SEED)
    rw_pred = np.random.randn(n) * y.std()
    results.append({
        "model": "Random Walk",
        "MAE": mae(y, rw_pred),
        "MAPE": mape(y, rw_pred),
        "sMAPE": smape(y, rw_pred),
        "DirAcc": direction_accuracy(y, rw_pred),
    })

    sma_window = 21
    sma_pred = np.zeros(n)
    for i in range(sma_window, n):
        sma_pred[i] = np.mean(y[i - sma_window:i])
    results.append({
        "model": f"SMA({sma_window}) delta",
        "MAE": mae(y[sma_window:], sma_pred[sma_window:]),
        "MAPE": mape(y[sma_window:], sma_pred[sma_window:]),
        "sMAPE": smape(y[sma_window:], sma_pred[sma_window:]),
        "DirAcc": direction_accuracy(y[sma_window:], sma_pred[sma_window:]),
    })

    mr_pred = np.zeros(n)
    mr_pred[1:] = -y[:-1]
    results.append({
        "model": "Mean Reversion",
        "MAE": mae(y[1:], mr_pred[1:]),
        "MAPE": mape(y[1:], mr_pred[1:]),
        "sMAPE": smape(y[1:], mr_pred[1:]),
        "DirAcc": direction_accuracy(y[1:], mr_pred[1:]),
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 10: Benchmark comparison")
    parser.add_argument('--ticker', default='SBER')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2021-12-31')
    parser.add_argument('--test_years', type=int, default=1)
    parser.add_argument('--data_source', default='local', choices=['yfinance', 'local'])
    parser.add_argument('--timeframe', default='M5')
    parser.add_argument('--data_path', default='G:\\data2')
    parser.add_argument('--raw_source', default='m1', choices=['m1', 'ticks'])
    parser.add_argument('--generator', default='tst', choices=['lstm', 'tst'])
    parser.add_argument('--lr_g', type=float, default=None)
    parser.add_argument('--lr_d', type=float, default=None)
    parser.add_argument('--q_weight', type=float, default=None)
    parser.add_argument('--seq_len', type=int, default=None)
    args = parser.parse_args()

    set_global_seed(SEED)

    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_years=args.test_years, data_source=args.data_source,
        timeframe=args.timeframe, data_path=args.data_path,
        local_raw_source=args.raw_source, generator=args.generator,
    )
    if args.lr_g is not None:
        cfg.lr_g = args.lr_g
    if args.lr_d is not None:
        cfg.lr_d = args.lr_d
    if args.q_weight is not None:
        cfg.q_weight = args.q_weight
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if cfg.data_source == "local" or cfg.timeframe != "D1":
        cfg.apply_timeframe_defaults()

    print(f"Loading panel: {args.ticker} {args.timeframe} [{args.start} .. {args.end}]")
    panel = build_panel_auto(cfg)
    print(f"Panel shape: {panel.shape}")

    train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)
    print(f"Train: {len(train_panel)}, Test: {len(test_panel)}")

    close_col = cfg.ticker
    if close_col not in test_panel.columns:
        candidates = [c for c in test_panel.columns
                      if "Close" in c or c == cfg.ticker]
        if not candidates:
            candidates = [test_panel.columns[0]]
        close_col = candidates[0]
    print(f"Using close column: {close_col}")

    test_close = test_panel[close_col].values
    test_deltas = np.diff(test_close) / (test_close[:-1] + 1e-9)

    print("\n" + "=" * 60)
    print("NAIVE BENCHMARKS")
    print("=" * 60)
    naive_results = compute_naive_benchmarks(test_deltas)
    for r in naive_results:
        print(f"  {r['model']:25s}  MAE={r['MAE']:.6f}  MAPE={r['MAPE']:.4f}  "
              f"DirAcc={r['DirAcc']:.4f}  sMAPE={r['sMAPE']:.4f}")

    print("\n" + "=" * 60)
    print(f"GAN ({args.generator.upper()}) -- running full pipeline")
    print("=" * 60)
    t0 = time.time()
    tr_feat, te_feat = build_features_safe(train_panel, test_panel, cfg)
    tr_all, te_all = fit_transforms(tr_feat, te_feat, cfg)
    met, y_true, y_pred, _, _, _ = run_one_split(tr_all, te_all, cfg)
    elapsed = time.time() - t0
    gan_result = {
        "model": f"GAN ({args.generator.upper()})",
        "MAE": met["MAE"],
        "MAPE": met["MAPE"],
        "sMAPE": smape(y_true, y_pred),
        "DirAcc": met["DirAcc"],
    }
    print(f"  {gan_result['model']:25s}  MAE={gan_result['MAE']:.6f}  "
          f"MAPE={gan_result['MAPE']:.4f}  DirAcc={gan_result['DirAcc']:.4f}  "
          f"sMAPE={gan_result['sMAPE']:.4f}  ({elapsed:.0f}s)")

    all_results = naive_results + [gan_result]

    os.makedirs("outputs/experiments", exist_ok=True)
    out_path = f"outputs/experiments/benchmarks_{args.timeframe}_{args.ticker}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "MAE", "MAPE", "sMAPE", "DirAcc"])
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved: {out_path}")

    best_naive_mae = min(r["MAE"] for r in naive_results)
    gan_mae = gan_result["MAE"]
    improvement = (best_naive_mae - gan_mae) / best_naive_mae * 100
    print(f"\n{'=' * 60}")
    print(f"GAN MAE: {gan_mae:.6f} vs Best Naive MAE: {best_naive_mae:.6f}")
    print(f"Improvement: {improvement:+.1f}%")
    if gan_mae < best_naive_mae:
        print("GAN BEATS naive baselines!")
    else:
        print("GAN does NOT beat naive baselines.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
