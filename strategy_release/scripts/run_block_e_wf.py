"""Block E Walk-Forward Validation.

Runs the momentum-trend strategy through 5 rolling train/test windows
to verify robustness across different market regimes.

Windows (6yr train / 2yr test, sliding by 2yr):
    1: TRAIN 2007-2012 | TEST 2013-2014
    2: TRAIN 2009-2014 | TEST 2015-2016
    3: TRAIN 2011-2016 | TEST 2017-2018
    4: TRAIN 2013-2018 | TEST 2019-2020
    5: TRAIN 2015-2020 | TEST 2021-2022

Usage:
    python scripts/run_block_e_wf.py --n_trials 20000
    python scripts/run_block_e_wf.py --n_trials 20000 --min_trades 150
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_local import load_m1_bars
from src.strategy.momentum_trend import (
    StrategyParams,
    prepare_strategy_data,
    generate_signals,
)
from src.strategy.backtester import (
    simulate_trades,
    compute_metrics,
    trades_to_dataframe,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "experiments",
)

WF_WINDOWS = [
    {"train_start": "2007-01-01", "train_end": "2012-12-31",
     "test_start": "2013-01-01", "test_end": "2014-12-31"},
    {"train_start": "2009-01-01", "train_end": "2014-12-31",
     "test_start": "2015-01-01", "test_end": "2016-12-31"},
    {"train_start": "2011-01-01", "train_end": "2016-12-31",
     "test_start": "2017-01-01", "test_end": "2018-12-31"},
    {"train_start": "2013-01-01", "train_end": "2018-12-31",
     "test_start": "2019-01-01", "test_end": "2020-12-31"},
    {"train_start": "2015-01-01", "train_end": "2020-12-31",
     "test_start": "2021-01-01", "test_end": "2022-12-31"},
]


def _compute_1c_metrics(trades) -> dict:
    if not trades:
        return {"avg_trade_1c": 0.0, "total_pts_1c": 0.0,
                "pf_1c": 0.0, "sharpe_1c": 0.0, "mdd_1c": 0.0}
    pnl = np.array([(t.exit_price - t.entry_price) for t in trades])
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    cum = np.cumsum(pnl)
    rmax = np.maximum.accumulate(cum)
    mdd = float((rmax - cum).max()) if len(pnl) else 0.0
    n = len(pnl)
    sharpe = 0.0
    if n > 1 and pnl.std() > 0:
        sharpe = float((pnl.mean() / pnl.std()) * np.sqrt(min(250, n)))
    return {
        "avg_trade_1c": float(pnl.mean()),
        "total_pts_1c": float(pnl.sum()),
        "pf_1c": gp / gl if gl > 0 else float("inf"),
        "sharpe_1c": sharpe,
        "mdd_1c": mdd,
    }


def _run_backtest(df_m1, params):
    df = prepare_strategy_data(df_m1, params)
    df = generate_signals(df, params)
    trades = simulate_trades(df, params)
    metrics = compute_metrics(trades)
    metrics.update(_compute_1c_metrics(trades))
    return trades, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Block E Walk-Forward Validation")
    parser.add_argument("--ticker", default="RTSF")
    parser.add_argument("--data_path", default=r"G:\data2")
    parser.add_argument("--n_trials", type=int, default=20000)
    parser.add_argument("--min_trades", type=int, default=150,
                        help="Min trades for 6yr window (scaled from 300/12yr)")
    parser.add_argument("--capital", type=float, default=2_300_000)
    parser.add_argument("--point_mult", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("  BLOCK E: Walk-Forward Validation")
    print(f"  {len(WF_WINDOWS)} folds, {args.n_trials} trials/fold")
    print(f"{'='*60}")

    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {dev_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[1] Loading all M1 bars ...")
    t0 = time.time()
    df_m1_all = load_m1_bars(args.data_path, args.ticker,
                             "2007-01-01", "2022-12-31")
    print(f"  {len(df_m1_all):,} bars in {time.time()-t0:.1f}s")

    from src.strategy.gpu_batch import (
        run_gpu_optimization, gpu_best_to_params,
        TF1_GRID, TF2_GRID,
    )

    base_params = StrategyParams(
        capital=args.capital,
        point_value_mult=args.point_mult,
    )

    fold_results = []
    total_t0 = time.time()

    for fold_idx, w in enumerate(WF_WINDOWS, 1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx}/{len(WF_WINDOWS)}: "
              f"TRAIN {w['train_start'][:4]}-{w['train_end'][:4]} | "
              f"TEST {w['test_start'][:4]}-{w['test_end'][:4]}")
        print(f"{'='*60}")

        tr_start = pd.Timestamp(w["train_start"])
        tr_end = pd.Timestamp(w["train_end"])
        te_start = pd.Timestamp(w["test_start"])
        te_end = pd.Timestamp(w["test_end"])

        df_train = df_m1_all[(df_m1_all.index >= tr_start)
                             & (df_m1_all.index <= tr_end)].copy()
        df_test = df_m1_all[(df_m1_all.index >= te_start)
                            & (df_m1_all.index <= te_end)].copy()
        print(f"  Train: {len(df_train):,} M1 bars")
        print(f"  Test:  {len(df_test):,} M1 bars")

        fold_t0 = time.time()
        top_results, _ = run_gpu_optimization(
            df_train,
            n_samples=args.n_trials,
            base_params=base_params,
            min_trades=args.min_trades,
            seed=args.seed + fold_idx,
            device=device,
            tf1_grid=TF1_GRID,
            tf2_grid=TF2_GRID,
        )
        opt_sec = time.time() - fold_t0

        if not top_results:
            print(f"  WARNING: Fold {fold_idx} -- no valid trials!")
            fold_results.append({
                "fold": fold_idx,
                "train_period": f"{w['train_start'][:4]}-{w['train_end'][:4]}",
                "test_period": f"{w['test_start'][:4]}-{w['test_end'][:4]}",
                "train_avg_1c": 0.0, "test_avg_1c": 0.0,
                "train_trades": 0, "test_trades": 0,
                "train_pf_1c": 0.0, "test_pf_1c": 0.0,
                "train_sharpe_1c": 0.0, "test_sharpe_1c": 0.0,
                "opt_sec": opt_sec,
            })
            continue

        best = gpu_best_to_params(top_results[0], base_params)

        _, tr_met = _run_backtest(df_train, best)
        _, te_met = _run_backtest(df_test, best)

        print(f"  TRAIN: avg_1c={tr_met.get('avg_trade_1c',0):>8.1f}  "
              f"trades={tr_met['num_trades']}  "
              f"PF={tr_met.get('pf_1c',0):.2f}  "
              f"Sharpe={tr_met.get('sharpe_1c',0):.2f}")
        print(f"  TEST:  avg_1c={te_met.get('avg_trade_1c',0):>8.1f}  "
              f"trades={te_met['num_trades']}  "
              f"PF={te_met.get('pf_1c',0):.2f}  "
              f"Sharpe={te_met.get('sharpe_1c',0):.2f}")
        print(f"  Best TF: ({best.tf1_minutes}, {best.tf2_minutes})  "
              f"lookback={best.lookback}  length={best.length}")
        print(f"  Optimised in {opt_sec:.0f}s")

        fold_results.append({
            "fold": fold_idx,
            "train_period": f"{w['train_start'][:4]}-{w['train_end'][:4]}",
            "test_period": f"{w['test_start'][:4]}-{w['test_end'][:4]}",
            "train_avg_1c": tr_met.get("avg_trade_1c", 0.0),
            "test_avg_1c": te_met.get("avg_trade_1c", 0.0),
            "train_trades": tr_met["num_trades"],
            "test_trades": te_met["num_trades"],
            "train_pf_1c": tr_met.get("pf_1c", 0.0),
            "test_pf_1c": te_met.get("pf_1c", 0.0),
            "train_sharpe_1c": tr_met.get("sharpe_1c", 0.0),
            "test_sharpe_1c": te_met.get("sharpe_1c", 0.0),
            "tf1": best.tf1_minutes,
            "tf2": best.tf2_minutes,
            "lookback": best.lookback,
            "length": best.length,
            "opt_sec": round(opt_sec, 1),
        })

    total_elapsed = time.time() - total_t0

    # --- Save CSV ---
    csv_path = os.path.join(OUT_DIR, "block_e_wf_summary.csv")
    if fold_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fold_results[0].keys())
            w.writeheader()
            w.writerows(fold_results)
        print(f"\n  Saved: {csv_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  WALK-FORWARD SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Fold':6s} {'Train':12s} {'Test':12s} "
          f"{'TR avg':>8s} {'TE avg':>8s} {'TR PF':>6s} {'TE PF':>6s} "
          f"{'TR Sh':>6s} {'TE Sh':>6s} {'TE tr':>5s}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} "
          f"{'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")

    for r in fold_results:
        print(f"  {r['fold']:<6d} {r['train_period']:12s} {r['test_period']:12s} "
              f"{r['train_avg_1c']:>8.1f} {r['test_avg_1c']:>8.1f} "
              f"{r['train_pf_1c']:>6.2f} {r['test_pf_1c']:>6.2f} "
              f"{r['train_sharpe_1c']:>6.2f} {r['test_sharpe_1c']:>6.2f} "
              f"{r['test_trades']:>5d}")

    oos_avgs = [r["test_avg_1c"] for r in fold_results]
    oos_pfs = [r["test_pf_1c"] for r in fold_results if r["test_pf_1c"] > 0]
    oos_sharpes = [r["test_sharpe_1c"] for r in fold_results]
    n_positive = sum(1 for a in oos_avgs if a > 0)

    print(f"\n  --- Aggregated OOS metrics ---")
    print(f"  Mean OOS avg_trade : {np.mean(oos_avgs):>8.1f} pts")
    print(f"  Mean OOS PF        : {np.mean(oos_pfs):>8.2f}" if oos_pfs
          else "  Mean OOS PF        :      N/A")
    print(f"  Mean OOS Sharpe    : {np.mean(oos_sharpes):>8.2f}")
    print(f"  OOS consistency    : {n_positive}/{len(fold_results)} "
          f"folds profitable")

    robust = n_positive >= 4 and (np.mean(oos_pfs) > 1.3 if oos_pfs else False)
    verdict = "ROBUST" if robust else "NOT ROBUST"
    print(f"\n  VERDICT: {verdict}")
    print(f"  Total time: {total_elapsed:.0f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
