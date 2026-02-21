"""Block E: Momentum-trend strategy for RTSF (RTS index futures).

Ports the MultiCharts EasyLanguage strategy to Python.  Loads M1 bars,
aggregates to two analysis timeframes (default 210-min and 90-min),
runs parameter optimisation (Optuna CPU or GPU batch) to maximise
average trade, then validates on the out-of-sample (OOS) test period.

Usage:
    python scripts/run_block_e.py                        # Optuna (CPU+Numba)
    python scripts/run_block_e.py --gpu                  # GPU batch search
    python scripts/run_block_e.py --gpu --n_trials 10000 # more GPU trials
    python scripts/run_block_e.py --use_defaults         # MC defaults only
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
    pre_aggregate,
    generate_signals,
)
from src.strategy.backtester import (
    simulate_trades,
    compute_metrics,
    trades_to_dataframe,
)
from src.strategy.optimizer import (
    run_optimization,
    study_to_dataframe,
    best_params_to_strategy,
)

SEED = 42
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "experiments",
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _print_metrics(label: str, metrics: dict, params: StrategyParams):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Num trades      : {metrics['num_trades']:>12d}")
    print(f"  Win rate        : {metrics['win_rate']:>12.2%}")
    print(f"  {'':18s}  {'1-contract':>12s}  {'sized':>14s}")
    print(f"  {'Avg trade':18s}  {metrics.get('avg_trade_1c',0):>10.1f} pt"
          f"  {metrics['avg_trade']:>14.0f}")
    print(f"  {'Total PnL':18s}  {metrics.get('total_pts_1c',0):>10.0f} pt"
          f"  {metrics['total_pnl']:>14.0f}")
    print(f"  {'Profit factor':18s}  {metrics.get('pf_1c',0):>12.2f}"
          f"  {metrics['profit_factor']:>14.2f}")
    print(f"  {'Sharpe':18s}  {metrics.get('sharpe_1c',0):>12.2f}"
          f"  {metrics['sharpe']:>14.2f}")
    print(f"  {'Max drawdown':18s}  {metrics.get('mdd_1c',0):>10.0f} pt"
          f"  {metrics['max_drawdown']:>14.0f}")
    print(f"  --- Key params ---")
    print(f"  tf1={params.tf1_minutes}  tf2={params.tf2_minutes}")
    print(f"  lookback={params.lookback}  length={params.length}")
    print(f"  lookback2={params.lookback2}  length2={params.length2}")
    print(f"  min_s={params.min_s}  max_s={params.max_s}")
    print(f"  koeff1={params.koeff1:.4f}  koeff2={params.koeff2:.4f}")
    print(f"  mmcoff={params.mmcoff}  exitday={params.exitday}"
          f"  sdel_day={params.sdel_day}  exit_week={params.exit_week}")
    sl = getattr(params, "sl_type", "none")
    if sl != "none":
        sl_detail = f"  sl_type={sl}"
        if sl == "fixed":
            sl_detail += f"  sl_pts={params.sl_pts}"
        elif sl == "atr":
            sl_detail += f"  sl_atr_mult={params.sl_atr_mult}"
        elif sl == "trail_atr":
            sl_detail += f"  trail_atr={params.trail_atr}"
        elif sl == "trail_pts":
            sl_detail += f"  trail_pts={params.trail_pts}"
        elif sl == "breakeven":
            sl_detail += f"  be_target={params.be_target}  be_offset={params.be_offset}"
        elif sl == "time":
            sl_detail += f"  time_bars={params.time_bars}  time_target={params.time_target}"
        print(sl_detail)
    print(f"{'='*60}\n")


def _compute_1c_metrics(trades, direction="long") -> dict:
    """Per-1-contract metrics (no position sizing bias)."""
    if not trades:
        return {"avg_trade_1c": 0.0, "total_pts_1c": 0.0,
                "pf_1c": 0.0, "sharpe_1c": 0.0, "mdd_1c": 0.0}
    sign = -1.0 if direction == "short" else 1.0
    pnl = np.array([sign * (t.exit_price - t.entry_price) for t in trades])
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


def _run_backtest(df_m1, params, label):
    t0 = time.time()
    df = prepare_strategy_data(df_m1, params)
    df = generate_signals(df, params)
    trades = simulate_trades(df, params)
    metrics = compute_metrics(trades)
    metrics["time_sec"] = round(time.time() - t0, 2)
    metrics.update(_compute_1c_metrics(trades, getattr(params, "direction", "long")))
    _print_metrics(label, metrics, params)
    return trades, metrics


def _save_csv(rows, fieldnames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def _print_device_info():
    """Print GPU / compute device information."""
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {dev}  ({mem:.1f} GB)")
    else:
        print("  GPU: not available (will use CPU)")
    try:
        from numba import njit
        print("  Numba JIT: available (fast CPU backtest)")
    except ImportError:
        print("  Numba JIT: not installed (pure Python fallback)")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Block E: RTSF momentum-trend strategy")
    parser.add_argument("--ticker", default="RTSF")
    parser.add_argument("--data_path", default=r"G:\data2")
    parser.add_argument("--train_start", default="2007-01-01")
    parser.add_argument("--train_end", default="2018-12-31")
    parser.add_argument("--test_start", default="2019-01-01")
    parser.add_argument("--test_end", default="2022-12-31")
    parser.add_argument("--tf1", type=int, default=210,
                        help="Trend timeframe in minutes")
    parser.add_argument("--tf2", type=int, default=90,
                        help="Trading timeframe in minutes")
    parser.add_argument("--n_trials", type=int, default=2000)
    parser.add_argument("--min_trades", type=int, default=30)
    parser.add_argument("--exit_week", type=int, default=0, choices=[0, 1],
                        help="Force exit at end of week (0=no, 1=yes)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU-accelerated batch evaluation "
                             "(requires CUDA)")
    parser.add_argument("--refine", action="store_true",
                        help="Directed search: narrow grid around centre point")
    parser.add_argument("--centre_json", default="",
                        help="Centre point JSON for --refine "
                             "(default: block_e_best_params.json)")
    parser.add_argument("--skip_optuna", action="store_true",
                        help="Skip optimisation, use defaults or --params_json")
    parser.add_argument("--use_defaults", action="store_true",
                        help="Run with original MC default params (no optuna)")
    parser.add_argument("--params_json", default="",
                        help="Path to JSON with custom params to evaluate")
    parser.add_argument("--direction", default="long", choices=["long", "short"],
                        help="Trade direction: long or short")
    parser.add_argument("--sl_types", default="none",
                        help="Comma-separated stop-loss types to test. "
                             "Options: none,fixed,atr,trail_atr,trail_pts,"
                             "breakeven,time  (default: none)")
    parser.add_argument("--capital", type=float, default=5_000_000)
    parser.add_argument("--point_mult", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.sl_types_tuple = tuple(s.strip() for s in args.sl_types.split(","))

    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(args.seed)
    dir_tag = "_short" if args.direction == "short" else ""
    if args.sl_types_tuple != ("none",):
        dir_tag += "_sl"

    sl_label = ""
    if args.sl_types_tuple != ("none",):
        sl_label = f" + SL({','.join(args.sl_types_tuple)})"
    print(f"\n{'='*60}")
    print(f"  BLOCK E: Momentum-Trend Strategy ({args.direction.upper()}{sl_label})")
    print(f"{'='*60}")
    _print_device_info()

    # --- 1. Load M1 data ------------------------------------------------
    print(f"\n[1/5] Loading M1 bars for {args.ticker} ...")
    t0 = time.time()
    df_m1_full = load_m1_bars(args.data_path, args.ticker,
                              args.train_start, args.test_end)
    print(f"  Loaded {len(df_m1_full):,} M1 bars "
          f"({df_m1_full.index.min()} -> {df_m1_full.index.max()}) "
          f"in {time.time()-t0:.1f}s")

    # --- 2. Split train / test ------------------------------------------
    print(f"\n[2/5] Splitting train ({args.train_start} .. {args.train_end})"
          f" / test ({args.test_start} .. {args.test_end}) ...")
    train_end_dt = pd.Timestamp(args.train_end)
    test_start_dt = pd.Timestamp(args.test_start)
    df_m1_train = df_m1_full[df_m1_full.index <= train_end_dt].copy()
    df_m1_test = df_m1_full[df_m1_full.index >= test_start_dt].copy()
    print(f"  Train: {len(df_m1_train):,} M1 bars")
    print(f"  Test:  {len(df_m1_test):,} M1 bars")

    base_params = StrategyParams(
        tf1_minutes=args.tf1,
        tf2_minutes=args.tf2,
        capital=args.capital,
        point_value_mult=args.point_mult,
        direction=args.direction,
    )

    # --- 3. Determine parameters ----------------------------------------
    best_params: StrategyParams

    if args.params_json:
        print(f"\n[3/5] Loading params from {args.params_json} ...")
        with open(args.params_json, "r", encoding="utf-8") as f:
            custom = json.load(f)
        remap = {"tf1": "tf1_minutes", "tf2": "tf2_minutes"}
        custom = {remap.get(k, k): v for k, v in custom.items()}
        best_params = StrategyParams(**{**asdict(base_params), **custom})
        print("  Loaded custom params.")

    elif args.skip_optuna or args.use_defaults:
        print("\n[3/5] Using default (MC original) parameters ...")
        best_params = StrategyParams(
            tf1_minutes=args.tf1,
            tf2_minutes=args.tf2,
            capital=args.capital,
            point_value_mult=args.point_mult,
        )
        print(f"  lookback={best_params.lookback}, length={best_params.length}, "
              f"lookback2={best_params.lookback2}, length2={best_params.length2}")

    elif args.gpu:
        # ---- GPU batch optimisation ------------------------------------
        ew_label = " + exit_week" if args.exit_week else ""
        mode_label = "REFINE" if args.refine else "BROAD"
        print(f"\n[3/5] GPU {mode_label} optimisation "
              f"({args.n_trials} samples) ...")
        print(f"  Objective: per-1-contract avg_trade{ew_label}")
        print(f"  min_trades={args.min_trades}")
        if not torch.cuda.is_available():
            print("  WARNING: CUDA not available, falling back to CPU tensors")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from src.strategy.gpu_batch import (
            run_gpu_optimization, gpu_best_to_params,
            TF1_GRID, TF2_GRID,
            REFINE_TF1_GRID, REFINE_TF2_GRID,
        )

        refine_centre = None
        if args.refine:
            centre_path = (args.centre_json
                           or os.path.join(OUT_DIR, "block_e_best_params.json"))
            print(f"  Loading centre from {centre_path}")
            with open(centre_path, "r", encoding="utf-8") as f:
                refine_centre = json.load(f)
            print(f"  Centre: {refine_centre}")

        use_tf1 = REFINE_TF1_GRID if args.refine else TF1_GRID
        use_tf2 = REFINE_TF2_GRID if args.refine else TF2_GRID

        t0 = time.time()
        top_results, all_df = run_gpu_optimization(
            df_m1_train,
            n_samples=args.n_trials,
            base_params=base_params,
            min_trades=args.min_trades,
            seed=args.seed,
            device=device,
            tf1_grid=use_tf1,
            tf2_grid=use_tf2,
            force_exit_week=bool(args.exit_week),
            refine_centre=refine_centre,
            direction=args.direction,
            sl_types=args.sl_types_tuple,
        )
        elapsed = time.time() - t0
        print(f"  GPU optimisation done in {elapsed:.1f}s")

        if not top_results:
            print("  ERROR: No valid trials found. Try increasing --n_trials "
                  "or lowering --min_trades.")
            sys.exit(1)

        suffix = "_refined" if args.refine else ""
        gpu_path = os.path.join(OUT_DIR, f"block_e{dir_tag}_gpu{suffix}_trials.csv")
        all_df.to_csv(gpu_path, index=False)
        print(f"  Saved {len(all_df)} trials -> {gpu_path}")

        best_json_path = os.path.join(OUT_DIR, f"block_e{dir_tag}_best_params.json")
        param_keys = ("tf1", "tf2", "lookback", "length", "lookback2",
                      "length2", "min_s", "max_s", "koeff1", "koeff2",
                      "mmcoff", "exitday", "sdel_day",
                      "sl_type", "sl_pts", "sl_atr_mult",
                      "trail_atr", "trail_pts", "be_target", "be_offset",
                      "time_bars", "time_target")
        best_dict = {k: top_results[0][k] for k in param_keys
                     if k in top_results[0]}
        best_dict["exit_week"] = args.exit_week
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump(best_dict, f, indent=2)
        print(f"  Saved best params -> {best_json_path}")

        best_params = gpu_best_to_params(top_results[0], base_params,
                                         exit_week=args.exit_week,
                                         direction=args.direction)

    else:
        # ---- Optuna optimisation (CPU + Numba JIT) ---------------------
        print(f"\n[3/5] Optuna optimisation ({args.n_trials} trials, "
              f"pre-aggregated + Numba JIT) ...")
        t0 = time.time()
        study = run_optimization(
            df_m1_train,
            n_trials=args.n_trials,
            base_params=base_params,
            min_trades=args.min_trades,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        print(f"  Optimisation done in {elapsed:.1f}s")
        print(f"  Best avg_trade = {study.best_value:.2f}")
        print(f"  Best params: {study.best_params}")

        trials_df = study_to_dataframe(study)
        optuna_path = os.path.join(OUT_DIR, "block_e_optuna.csv")
        trials_df.to_csv(optuna_path, index=False)
        print(f"  Saved {len(trials_df)} trials -> {optuna_path}")

        best_json_path = os.path.join(OUT_DIR, "block_e_best_params.json")
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, indent=2)
        print(f"  Saved best params -> {best_json_path}")

        best_params = best_params_to_strategy(study, base_params)

    # --- 4. Evaluate on TRAIN -------------------------------------------
    print(f"\n[4/5] Evaluating best params on TRAIN set ...")
    train_trades, train_metrics = _run_backtest(
        df_m1_train, best_params, "TRAIN results")

    # --- 5. Evaluate on TEST (OOS) --------------------------------------
    print(f"[5/5] Evaluating best params on TEST (OOS) set ...")
    test_trades, test_metrics = _run_backtest(
        df_m1_test, best_params, "TEST (OOS) results")

    # --- Save results ----------------------------------------------------
    result_rows = []
    for label, metrics in [("train", train_metrics), ("test", test_metrics)]:
        row = {"split": label}
        row.update(metrics)
        row.update(asdict(best_params))
        result_rows.append(row)

    fieldnames = list(result_rows[0].keys())
    results_path = os.path.join(OUT_DIR, f"block_e{dir_tag}_results.csv")
    _save_csv(result_rows, fieldnames, results_path)

    for label, trades in [("train", train_trades), ("test", test_trades)]:
        tdf = trades_to_dataframe(trades)
        tpath = os.path.join(OUT_DIR, f"block_e{dir_tag}_trades_{label}.csv")
        tdf.to_csv(tpath, index=False)
        print(f"  Saved {len(tdf)} trades -> {tpath}")

    # --- Summary ---------------------------------------------------------
    print(f"\n{'='*60}")
    print("  BLOCK E SUMMARY  (per 1 contract)")
    print(f"{'='*60}")
    tr1c = train_metrics.get("avg_trade_1c", 0)
    te1c = test_metrics.get("avg_trade_1c", 0)
    print(f"  {'':20s}  {'TRAIN':>12s}  {'TEST':>12s}")
    print(f"  {'avg trade (pts)':20s}  {tr1c:>12.1f}  {te1c:>12.1f}")
    print(f"  {'trades':20s}  {train_metrics['num_trades']:>12d}"
          f"  {test_metrics['num_trades']:>12d}")
    print(f"  {'win rate':20s}  {train_metrics['win_rate']:>12.2%}"
          f"  {test_metrics['win_rate']:>12.2%}")
    print(f"  {'profit factor':20s}  {train_metrics.get('pf_1c',0):>12.2f}"
          f"  {test_metrics.get('pf_1c',0):>12.2f}")
    print(f"  {'sharpe':20s}  {train_metrics.get('sharpe_1c',0):>12.2f}"
          f"  {test_metrics.get('sharpe_1c',0):>12.2f}")
    print(f"  {'max DD (pts)':20s}  {train_metrics.get('mdd_1c',0):>12.0f}"
          f"  {test_metrics.get('mdd_1c',0):>12.0f}")
    print(f"  --- Timeframes: tf1={best_params.tf1_minutes}  "
          f"tf2={best_params.tf2_minutes}")
    decay = 0.0
    if tr1c != 0:
        decay = (1 - te1c / tr1c) * 100
    print(f"  OOS decay (1c): {decay:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
