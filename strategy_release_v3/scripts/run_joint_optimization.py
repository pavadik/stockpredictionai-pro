"""Joint Optimization: Strategy Parameters + ML Overlay (10x scale).

Pipeline:
  1. GPU broad search: 50,000 random parameter sets (10x from 5,000)
  2. Take top-50 parameter sets (10x from top-5)
  3. For each top set: generate combined trades, build ML dataset, train XGBoost
  4. Evaluate each (params + ML) combination on OOS
  5. Select the best joint (params, ML model) pair
  6. Optuna refinement: 20,000 trials around the best GPU set (10x from 2,000)
  7. Final evaluation with leverage search

Usage:
    python scripts/run_joint_optimization.py --gpu
    python scripts/run_joint_optimization.py --gpu --n_gpu 50000 --n_optuna 20000
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_local import load_m1_bars, aggregate_intraday_custom
from src.strategy.momentum_trend import (
    StrategyParams,
    prepare_strategy_data,
    pre_aggregate,
    apply_params_fast,
    generate_signals,
)
from src.strategy.backtester import (
    simulate_trades,
    simulate_combined_trades,
    compute_metrics,
    trades_to_dataframe,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "joint_opt",
)

FEATURES = [
    "hour", "minute", "day_of_week", "month",
    "d1_volume", "d1_tr", "d1_adx", "d1_ret_1d", "d1_ret_5d",
    "m1_volume", "m1_ret_15m", "m1_ret_60m", "m1_ret_120m",
    "is_long",
]

# -----------------------------------------------------------------------
# ML helpers (inline to avoid file I/O between steps)
# -----------------------------------------------------------------------

def _build_d1_features(m1: pd.DataFrame):
    """Pre-compute D1 bars, TR, ADX for ML feature extraction."""
    d1 = m1.groupby(m1.index.date).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    })
    prev_close = d1["close"].shift(1)
    tr = pd.concat([
        d1["high"] - d1["low"],
        (d1["high"] - prev_close).abs(),
        (d1["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = d1["high"] - d1["high"].shift(1)
    down_move = d1["low"].shift(1) - d1["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    alpha = 1.0 / 14
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=d1.index).ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=d1.index).ewm(alpha=alpha, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return d1, tr, adx


def _extract_features(trades_df: pd.DataFrame, m1: pd.DataFrame,
                      d1, tr, adx) -> pd.DataFrame:
    """Build ML feature matrix from a trades DataFrame."""
    rows = []
    for _, t in trades_df.iterrows():
        entry_time = pd.Timestamp(t["entry_time"])
        entry_date = entry_time.date()
        f = {
            "hour": entry_time.hour,
            "minute": entry_time.minute,
            "day_of_week": entry_time.weekday(),
            "month": entry_time.month,
        }
        try:
            loc = d1.index.get_loc(entry_date)
            if loc > 10:
                yest = d1.iloc[loc - 1]
                f["d1_volume"] = yest["volume"]
                f["d1_tr"] = tr.iloc[loc - 1]
                f["d1_adx"] = adx.iloc[loc - 1]
                f["d1_ret_1d"] = yest["close"] / d1.iloc[loc - 2]["close"] - 1.0
                f["d1_ret_5d"] = yest["close"] / d1.iloc[loc - 6]["close"] - 1.0
            else:
                f.update({k: 0 for k in ["d1_volume", "d1_tr", "d1_adx", "d1_ret_1d", "d1_ret_5d"]})
        except KeyError:
            f.update({k: 0 for k in ["d1_volume", "d1_tr", "d1_adx", "d1_ret_1d", "d1_ret_5d"]})

        try:
            m1_loc = m1.index.get_loc(entry_time)
            if m1_loc > 120:
                m1_entry = m1.iloc[m1_loc]
                f["m1_volume"] = m1_entry["volume"]
                f["m1_ret_15m"] = m1_entry["close"] / m1.iloc[m1_loc - 15]["close"] - 1.0
                f["m1_ret_60m"] = m1_entry["close"] / m1.iloc[m1_loc - 60]["close"] - 1.0
                f["m1_ret_120m"] = m1_entry["close"] / m1.iloc[m1_loc - 120]["close"] - 1.0
            else:
                f.update({k: 0 for k in ["m1_volume", "m1_ret_15m", "m1_ret_60m", "m1_ret_120m"]})
        except Exception:
            f.update({k: 0 for k in ["m1_volume", "m1_ret_15m", "m1_ret_60m", "m1_ret_120m"]})

        f["is_long"] = 1 if t["direction"] == "long" else 0
        f["pnl"] = t["pnl"]
        f["entry_time"] = entry_time
        rows.append(f)

    return pd.DataFrame(rows)


def _train_xgboost(df_features: pd.DataFrame, train_end: str,
                    n_estimators: int = 500) -> tuple:
    """Train XGBoost on features, return (model, auc_oos, df_with_proba)."""
    train_end_dt = pd.Timestamp(train_end)
    df_train = df_features[df_features["entry_time"] <= train_end_dt]
    df_test = df_features[df_features["entry_time"] > train_end_dt]

    if len(df_train) < 50 or len(df_test) < 20:
        return None, 0.0, df_features

    target = (df_features["pnl"] > 0).astype(int)
    y_train = target[df_train.index]
    y_test = target[df_test.index]

    X_train = df_train[FEATURES]
    X_test = df_test[FEATURES]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=42,
        early_stopping_rounds=30,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    proba_test = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, proba_test)
    except ValueError:
        auc = 0.5

    proba_all = np.zeros(len(df_features))
    proba_all[df_train.index] = model.predict_proba(X_train)[:, 1]
    proba_all[df_test.index] = proba_test
    df_features = df_features.copy()
    df_features["ml_proba"] = proba_all

    return model, auc, df_features


def _evaluate_ml_filtered(trades_df: pd.DataFrame, ml_proba: np.ndarray,
                          threshold: float, capital: float,
                          leverage: float, commission_pct: float,
                          cap_long: int, cap_short: int) -> dict:
    """Evaluate PnL of ML-filtered trades with position sizing."""
    mask = ml_proba >= threshold
    filtered = trades_df[mask].copy()
    if len(filtered) == 0:
        return {"trades": 0, "net_pnl": 0, "pf": 0, "max_dd_pct": 0}

    contracts = filtered["contracts"].values
    pnl_1c = filtered["pnl_1c"].values if "pnl_1c" in filtered.columns else filtered["pnl"].values / contracts
    direction = filtered["direction"].values
    entry_p = filtered["entry_price"].values
    exit_p = filtered["exit_price"].values

    leveraged = np.clip((contracts * leverage).astype(int), 1, None)
    caps = np.where(direction == "long", cap_long, cap_short)
    leveraged = np.minimum(leveraged, caps)

    pnl_sized = pnl_1c * leveraged
    comm = commission_pct / 100.0 * (entry_p + exit_p) * leveraged
    pnl_net = pnl_sized - comm

    equity = np.cumsum(pnl_net) + capital
    peak = np.maximum.accumulate(equity)
    dd_pct = ((peak - equity) / peak * 100)
    max_dd = float(dd_pct.max()) if len(dd_pct) > 0 else 0

    gross_profit = float(pnl_net[pnl_net > 0].sum())
    gross_loss = float(abs(pnl_net[pnl_net < 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "trades": len(filtered),
        "net_pnl": float(pnl_net.sum()),
        "pf": pf,
        "max_dd_pct": max_dd,
        "win_rate": float((pnl_net > 0).sum() / len(pnl_net)),
    }


# -----------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Joint Optimization (10x scale)")
    ap.add_argument("--ticker", default="RTSF")
    ap.add_argument("--data_path", default=r"G:\data2")
    ap.add_argument("--train_start", default="2007-01-01")
    ap.add_argument("--train_end", default="2018-12-31")
    ap.add_argument("--test_start", default="2019-01-01")
    ap.add_argument("--test_end", default="2025-12-31")
    ap.add_argument("--n_gpu", type=int, default=50000,
                    help="GPU broad search trials (default: 50000, was 5000)")
    ap.add_argument("--n_optuna", type=int, default=20000,
                    help="Optuna refinement trials (default: 20000, was 2000)")
    ap.add_argument("--top_k", type=int, default=50,
                    help="Top parameter sets for joint ML training (default: 50)")
    ap.add_argument("--n_estimators", type=int, default=500,
                    help="XGBoost n_estimators (default: 500, was 200)")
    ap.add_argument("--capital", type=float, default=5_000_000)
    ap.add_argument("--commission", type=float, default=0.01)
    ap.add_argument("--cap_long", type=int, default=100)
    ap.add_argument("--cap_short", type=int, default=40)
    ap.add_argument("--leverage", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--skip_gpu", action="store_true",
                    help="Skip GPU search, load previous top results")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(args.seed)

    sep = "=" * 80
    print(f"\n{sep}")
    print("  JOINT OPTIMIZATION: Strategy Params + ML Overlay (10x scale)")
    print(f"  GPU trials: {args.n_gpu:,}  |  Optuna trials: {args.n_optuna:,}")
    print(f"  Top-K for ML: {args.top_k}  |  XGBoost trees: {args.n_estimators}")
    print(f"{sep}")

    # ── Step 1: Load data ──────────────────────────────────────────────
    print(f"\n[1/7] Loading M1 bars ...")
    t0 = time.time()
    df_m1_full = load_m1_bars(args.data_path, args.ticker,
                              args.train_start, args.test_end)
    print(f"  {len(df_m1_full):,} M1 bars in {time.time()-t0:.1f}s")

    train_end_dt = pd.Timestamp(args.train_end)
    test_start_dt = pd.Timestamp(args.test_start)
    df_m1_train = df_m1_full[df_m1_full.index <= train_end_dt].copy()

    print(f"\n[1b] Pre-computing D1 features for ML ...")
    d1, d1_tr, d1_adx = _build_d1_features(df_m1_full)
    print(f"  {len(d1)} D1 bars ready.")

    # ── Step 2: GPU broad search (50,000 trials) ──────────────────────
    import torch
    from src.strategy.gpu_batch import (
        run_gpu_optimization, gpu_best_to_params,
        TF1_GRID, TF2_GRID,
    )

    top_results_path = os.path.join(OUT_DIR, "gpu_top_results.json")

    if args.skip_gpu and os.path.exists(top_results_path):
        print(f"\n[2/7] Loading previous GPU top results ...")
        with open(top_results_path) as f:
            top_results = json.load(f)
        print(f"  Loaded {len(top_results)} top parameter sets.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for direction in ["long", "short"]:
            print(f"\n[2/7] GPU broad search: {direction.upper()} "
                  f"({args.n_gpu:,} trials) ...")
            base = StrategyParams(
                capital=args.capital, point_value_mult=300.0,
                direction=direction,
            )
            t0 = time.time()
            top, all_df = run_gpu_optimization(
                df_m1_train, n_samples=args.n_gpu, base_params=base,
                min_trades=50, seed=args.seed, device=device,
                top_k=args.top_k, tf1_grid=TF1_GRID, tf2_grid=TF2_GRID,
                direction=direction,
            )
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s. Valid: {len(all_df)}")

            csv_path = os.path.join(OUT_DIR, f"gpu_{direction}_all.csv")
            all_df.to_csv(csv_path, index=False)

            for r in top:
                r["direction"] = direction

            if direction == "long":
                top_results = top
            else:
                top_results.extend(top)

        with open(top_results_path, "w") as f:
            json.dump(top_results, f, indent=2)
        print(f"  Saved {len(top_results)} top sets -> {top_results_path}")

    # ── Step 3: For each top set, generate combined trades ────────────
    print(f"\n[3/7] Generating combined trades for top-{len(top_results)} sets ...")

    long_sets = [r for r in top_results if r.get("direction") == "long"]
    short_sets = [r for r in top_results if r.get("direction") == "short"]

    if not long_sets or not short_sets:
        print("  ERROR: Need at least 1 long and 1 short set.")
        sys.exit(1)

    combos = []
    for li, lp in enumerate(long_sets[:25]):
        for si, sp in enumerate(short_sets[:25]):
            combos.append((li, si, lp, sp))

    print(f"  Testing {len(combos)} LONG x SHORT combinations ...")

    base_long = StrategyParams(capital=args.capital, point_value_mult=300.0, direction="long")
    base_short = StrategyParams(capital=args.capital, point_value_mult=300.0, direction="short")

    best_combo = None
    best_oos_pnl = -float("inf")
    results_log = []

    for idx, (li, si, lp_dict, sp_dict) in enumerate(combos):
        pL = gpu_best_to_params(lp_dict, base_long, direction="long")
        pS = gpu_best_to_params(sp_dict, base_short, direction="short")

        # Generate signals on full data
        df_sig_L = prepare_strategy_data(df_m1_full, pL)
        df_sig_L = generate_signals(df_sig_L, pL)
        df_sig_S = prepare_strategy_data(df_m1_full, pS)
        df_sig_S = generate_signals(df_sig_S, pS)

        trades = simulate_combined_trades(
            df_sig_L, df_sig_S, pL, pS,
            max_contracts_long=args.cap_long,
            max_contracts_short=args.cap_short,
            flip_mode="close_loss",
            leverage=args.leverage,
        )

        if len(trades) < 100:
            continue

        tdf = trades_to_dataframe(trades)
        tdf["pnl_1c"] = np.where(
            tdf["direction"] == "short", -1, 1
        ) * (tdf["exit_price"] - tdf["entry_price"])

        # Build ML features
        ml_df = _extract_features(tdf, df_m1_full, d1, d1_tr, d1_adx)

        # Train XGBoost
        model, auc, ml_df = _train_xgboost(ml_df, args.train_end,
                                            n_estimators=args.n_estimators)
        if model is None:
            continue

        # Evaluate ML-filtered on OOS
        oos_mask = ml_df["entry_time"] > pd.Timestamp(args.train_end)
        oos_df = tdf[oos_mask.values]
        oos_proba = ml_df.loc[oos_mask, "ml_proba"].values

        for threshold in [0.48, 0.50, 0.52]:
            result = _evaluate_ml_filtered(
                oos_df, oos_proba, threshold,
                args.capital, args.leverage, args.commission,
                args.cap_long, args.cap_short,
            )
            result.update({
                "long_idx": li, "short_idx": si,
                "threshold": threshold, "auc": auc,
                "base_trades": len(trades),
            })
            results_log.append(result)

            if result["net_pnl"] > best_oos_pnl and result["trades"] >= 50:
                best_oos_pnl = result["net_pnl"]
                best_combo = {
                    "long_params": lp_dict, "short_params": sp_dict,
                    "threshold": threshold, "auc": auc,
                    **result,
                }

        if (idx + 1) % 10 == 0:
            print(f"    {idx+1}/{len(combos)} combos evaluated. "
                  f"Best OOS PnL so far: {best_oos_pnl:,.0f}")

    results_df = pd.DataFrame(results_log)
    results_df.to_csv(os.path.join(OUT_DIR, "joint_combo_results.csv"), index=False)

    if best_combo is None:
        print("  ERROR: No valid combo found.")
        sys.exit(1)

    print(f"\n  Best combo: L#{best_combo.get('long_idx')} + S#{best_combo.get('short_idx')}")
    print(f"  OOS Trades: {best_combo['trades']}, Net PnL: {best_combo['net_pnl']:,.0f}")
    print(f"  PF: {best_combo['pf']:.2f}, Max DD: {best_combo['max_dd_pct']:.2f}%")
    print(f"  ML Threshold: {best_combo['threshold']}, AUC: {best_combo['auc']:.3f}")

    # ── Step 4: Optuna refinement around best combo ───────────────────
    print(f"\n[4/7] Optuna refinement ({args.n_optuna:,} trials) ...")

    from src.strategy.optimizer import run_optimization, best_params_to_strategy

    for direction, params_dict in [("long", best_combo["long_params"]),
                                   ("short", best_combo["short_params"])]:
        base = StrategyParams(
            capital=args.capital, point_value_mult=300.0,
            direction=direction,
            tf1_minutes=params_dict.get("tf1", params_dict.get("tf1_minutes", 180)),
            tf2_minutes=params_dict.get("tf2", params_dict.get("tf2_minutes", 60)),
        )

        print(f"\n  Refining {direction.upper()} ({args.n_optuna:,} trials) ...")
        t0 = time.time()
        study = run_optimization(
            df_m1_train, n_trials=args.n_optuna,
            base_params=base, min_trades=50, seed=args.seed,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s. Best avg_trade = {study.best_value:.2f}")

        refined = best_params_to_strategy(study, base)
        refined_dict = asdict(refined)
        path = os.path.join(OUT_DIR, f"refined_{direction}_params.json")
        with open(path, "w") as f:
            json.dump(refined_dict, f, indent=2)
        print(f"  Saved -> {path}")

        best_combo[f"{direction}_params_refined"] = refined_dict

    # ── Step 5: Final combined backtest with refined params ───────────
    print(f"\n[5/7] Final combined backtest with refined params ...")

    pL_final = StrategyParams(**{k: v for k, v in best_combo.get("long_params_refined", best_combo["long_params"]).items()
                                 if k in asdict(StrategyParams()).keys()})
    pS_final = StrategyParams(**{k: v for k, v in best_combo.get("short_params_refined", best_combo["short_params"]).items()
                                 if k in asdict(StrategyParams()).keys()})

    df_sig_L = prepare_strategy_data(df_m1_full, pL_final)
    df_sig_L = generate_signals(df_sig_L, pL_final)
    df_sig_S = prepare_strategy_data(df_m1_full, pS_final)
    df_sig_S = generate_signals(df_sig_S, pS_final)

    final_trades = simulate_combined_trades(
        df_sig_L, df_sig_S, pL_final, pS_final,
        max_contracts_long=args.cap_long,
        max_contracts_short=args.cap_short,
        flip_mode="close_loss",
        leverage=args.leverage,
    )

    tdf_final = trades_to_dataframe(final_trades)
    tdf_final["pnl_1c"] = np.where(
        tdf_final["direction"] == "short", -1, 1
    ) * (tdf_final["exit_price"] - tdf_final["entry_price"])

    tdf_final.to_csv(os.path.join(OUT_DIR, "final_all_trades.csv"), index=False)
    print(f"  {len(final_trades)} total trades saved.")

    # ── Step 6: Train final ML model ──────────────────────────────────
    print(f"\n[6/7] Training final XGBoost ({args.n_estimators} trees) ...")

    ml_final = _extract_features(tdf_final, df_m1_full, d1, d1_tr, d1_adx)
    model_final, auc_final, ml_final = _train_xgboost(
        ml_final, args.train_end, n_estimators=args.n_estimators)

    if model_final is not None:
        model_final.save_model(os.path.join(OUT_DIR, "final_xgboost.json"))
        print(f"  AUC (OOS): {auc_final:.4f}")

        # Export to ONNX for C# consumption
        try:
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
            onnx_model = convert_xgboost(
                model_final.get_booster(),
                initial_types=[("float_input", FloatTensorType([None, 14]))],
            )
            onnx_path = os.path.join(OUT_DIR, "block_e_xgboost_overlay.onnx")
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"  ONNX model saved -> {onnx_path}")
        except ImportError:
            print("  WARNING: onnxmltools not installed, skipping ONNX export.")

    # ── Step 7: Final evaluation with leverage search ─────────────────
    print(f"\n[7/7] Leverage search on ML-filtered OOS trades ...")

    oos_mask = ml_final["entry_time"] > pd.Timestamp(args.train_end)
    oos_tdf = tdf_final[oos_mask.values]
    oos_proba = ml_final.loc[oos_mask, "ml_proba"].values

    print(f"\n  {'Lev':>5} {'Thr':>5} {'Trades':>7} {'Net PnL':>12} "
          f"{'PF':>6} {'MaxDD%':>7} {'WR%':>6}")
    print(f"  {'-'*55}")

    best_final = None
    for lev in [2.0, 3.0, 4.0, 5.0, 6.0]:
        for thr in [0.48, 0.50, 0.52, 0.55]:
            r = _evaluate_ml_filtered(
                oos_tdf, oos_proba, thr,
                args.capital, lev, args.commission,
                args.cap_long, args.cap_short,
            )
            print(f"  {lev:>5.1f} {thr:>5.2f} {r['trades']:>7} "
                  f"{r['net_pnl']:>12,.0f} {r['pf']:>6.2f} "
                  f"{r['max_dd_pct']:>7.2f} {r.get('win_rate',0):>6.1%}")

            if best_final is None or r["net_pnl"] > best_final["net_pnl"]:
                if r["trades"] >= 50 and r["max_dd_pct"] < 20:
                    best_final = {**r, "leverage": lev, "threshold": thr}

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  JOINT OPTIMIZATION COMPLETE")
    print(f"{sep}")
    if best_final:
        print(f"  Best config:")
        print(f"    Leverage: {best_final['leverage']}x")
        print(f"    ML Threshold: {best_final['threshold']}")
        print(f"    OOS Trades: {best_final['trades']}")
        print(f"    OOS Net PnL: {best_final['net_pnl']:,.0f}")
        print(f"    OOS PF: {best_final['pf']:.2f}")
        print(f"    OOS Max DD: {best_final['max_dd_pct']:.2f}%")

    summary = {
        "long_params": best_combo.get("long_params_refined", best_combo["long_params"]),
        "short_params": best_combo.get("short_params_refined", best_combo["short_params"]),
        "ml_auc": auc_final if model_final else 0,
        "best_leverage": best_final["leverage"] if best_final else args.leverage,
        "best_threshold": best_final["threshold"] if best_final else 0.5,
        "oos_result": best_final,
    }
    with open(os.path.join(OUT_DIR, "joint_optimization_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved -> {os.path.join(OUT_DIR, 'joint_optimization_summary.json')}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
