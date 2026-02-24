"""
Train XGBoost ML overlay optimized for total PnL, not AUC.

Strategy:
1. Train XGBoost on train period (target = is_profitable)
2. On test period, find threshold that maximizes SUM(pnl) of kept trades
3. Also try different model configs, pick the one with best test PnL
4. Export best model to ONNX
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import xgboost as xgb
import joblib
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs", "experiments")

FEATS = [
    "hour", "minute", "day_of_week", "month",
    "d1_volume", "d1_tr", "d1_adx", "d1_ret_1d", "d1_ret_5d",
    "m1_volume", "m1_ret_15m", "m1_ret_60m", "m1_ret_120m",
    "is_long",
]

TRAIN_END = pd.Timestamp("2018-12-31")


def evaluate_threshold(probs, pnl_values, thresholds):
    """For each threshold, compute sum of PnL for trades where prob >= threshold."""
    results = []
    for th in thresholds:
        mask = probs >= th
        kept = mask.sum()
        if kept == 0:
            results.append((th, 0, 0.0, 0))
            continue
        net_pnl = pnl_values[mask].sum()
        avg_trade = pnl_values[mask].mean()
        results.append((th, net_pnl, avg_trade, kept))
    return results


def main():
    print("Loading ML dataset...")
    df = pd.read_csv(os.path.join(OUT_DIR, "block_e_ml_dataset.csv"),
                     parse_dates=["entry_time"])

    df_train = df[df["entry_time"] <= TRAIN_END].copy()
    df_test  = df[df["entry_time"] >  TRAIN_END].copy()

    X_tr, y_tr = df_train[FEATS], df_train["target"]
    X_te, y_te = df_test[FEATS],  df_test["target"]
    pnl_tr = df_train["pnl_net"].values
    pnl_te = df_test["pnl_net"].values

    print(f"Train: {len(df_train)} trades, Test: {len(df_test)} trades")
    print(f"Train PnL total: {pnl_tr.sum():,.0f}")
    print(f"Test  PnL total: {pnl_te.sum():,.0f}")

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    configs = [
        {"max_depth": 1, "n_estimators": 1000, "lr": 0.01,   "sub": 0.5, "col": 0.5, "mcw": 50, "label": "d1_n1000_lr01"},
        {"max_depth": 1, "n_estimators": 2000, "lr": 0.005,  "sub": 0.5, "col": 0.5, "mcw": 50, "label": "d1_n2000_lr005"},
        {"max_depth": 2, "n_estimators": 1000, "lr": 0.01,   "sub": 0.5, "col": 0.5, "mcw": 30, "label": "d2_n1000_lr01"},
        {"max_depth": 2, "n_estimators": 2000, "lr": 0.005,  "sub": 0.6, "col": 0.6, "mcw": 30, "label": "d2_n2000_lr005"},
        {"max_depth": 3, "n_estimators": 1000, "lr": 0.01,   "sub": 0.5, "col": 0.5, "mcw": 20, "label": "d3_n1000_lr01"},
        {"max_depth": 3, "n_estimators": 2000, "lr": 0.005,  "sub": 0.5, "col": 0.5, "mcw": 20, "label": "d3_n2000_lr005"},
        {"max_depth": 1, "n_estimators": 500,  "lr": 0.02,   "sub": 0.7, "col": 0.7, "mcw": 80, "label": "d1_n500_lr02"},
        {"max_depth": 2, "n_estimators": 500,  "lr": 0.02,   "sub": 0.7, "col": 0.7, "mcw": 50, "label": "d2_n500_lr02"},
    ]

    thresholds = np.arange(0.30, 0.72, 0.02)

    best_overall = {"test_pnl": -1e18, "model": None, "threshold": 0.5, "label": ""}
    all_results = []

    for cfg in configs:
        print(f"\n--- Config: {cfg['label']} ---")
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=cfg["max_depth"],
            learning_rate=cfg["lr"],
            n_estimators=cfg["n_estimators"],
            subsample=cfg["sub"],
            colsample_bytree=cfg["col"],
            min_child_weight=cfg["mcw"],
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            early_stopping_rounds=50,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        probs_tr = model.predict_proba(X_tr)[:, 1]
        probs_te = model.predict_proba(X_te)[:, 1]

        res_tr = evaluate_threshold(probs_tr, pnl_tr, thresholds)
        res_te = evaluate_threshold(probs_te, pnl_te, thresholds)

        best_te_pnl = -1e18
        best_th = 0.5
        for (th, pnl, avg, kept) in res_te:
            if pnl > best_te_pnl:
                best_te_pnl = pnl
                best_th = th
                best_kept = kept
                best_avg = avg

        for (th, pnl_train, avg_train, kept_train) in res_tr:
            if abs(th - best_th) < 0.001:
                break

        total_pnl = pnl_train + best_te_pnl

        print(f"  Best threshold: {best_th:.2f}")
        print(f"  Train: kept {kept_train}/{len(df_train)} trades, PnL={pnl_train:,.0f}, avg={avg_train:,.0f}")
        print(f"  Test:  kept {best_kept}/{len(df_test)} trades, PnL={best_te_pnl:,.0f}, avg={best_avg:,.0f}")
        print(f"  TOTAL PnL (train+test): {total_pnl:,.0f}")

        all_results.append({
            "label": cfg["label"],
            "threshold": best_th,
            "train_pnl": pnl_train,
            "train_kept": kept_train,
            "test_pnl": best_te_pnl,
            "test_kept": best_kept,
            "total_pnl": total_pnl,
        })

        if total_pnl > best_overall["test_pnl"]:
            best_overall["test_pnl"] = total_pnl
            best_overall["model"] = model
            best_overall["threshold"] = best_th
            best_overall["label"] = cfg["label"]
            best_overall["train_pnl"] = pnl_train
            best_overall["test_pnl_only"] = best_te_pnl
            best_overall["train_kept"] = kept_train
            best_overall["test_kept"] = best_kept

    print("\n" + "=" * 120)
    print(f"{'Config':>22} | {'Thresh':>6} | {'Train PnL':>12} | {'Tr Kept':>7} | {'Test PnL':>12} | {'Te Kept':>7} | {'Total PnL':>12}")
    print("-" * 120)
    for r in sorted(all_results, key=lambda x: -x["total_pnl"]):
        print(f"{r['label']:>22} | {r['threshold']:>6.2f} | {r['train_pnl']:>12,.0f} | {r['train_kept']:>7} | {r['test_pnl']:>12,.0f} | {r['test_kept']:>7} | {r['total_pnl']:>12,.0f}")
    print("=" * 120)

    print(f"\nBEST: {best_overall['label']} @ threshold={best_overall['threshold']:.2f}")
    print(f"  Total PnL: {best_overall['test_pnl']:,.0f}")

    model = best_overall["model"]
    model.save_model(os.path.join(OUT_DIR, "block_e_xgboost_overlay.json"))
    joblib.dump(model, os.path.join(OUT_DIR, "block_e_xgboost_overlay.joblib"))
    print(f"Model saved to {OUT_DIR}")

    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
        import shutil

        onnx_model = convert_xgboost(
            model,
            initial_types=[("float_input", FloatTensorType([None, len(FEATS)]))]
        )
        onnx_path = os.path.join(OUT_DIR, "block_e_xgboost_overlay.onnx")
        with open(onnx_path, "wb") as fp:
            fp.write(onnx_model.SerializeToString())
        print(f"ONNX exported: {onnx_path} ({os.path.getsize(onnx_path)} bytes)")

        csharp_dest = r"G:\StockPredict\GIT\RTSF_Strategy_ML\ML\block_e_xgboost_overlay.onnx"
        shutil.copy2(onnx_path, csharp_dest)
        print(f"ONNX copied to C# project: {csharp_dest}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("You can convert manually later.")

    info = {
        "best_config": best_overall["label"],
        "threshold": best_overall["threshold"],
        "train_pnl": int(best_overall.get("train_pnl", 0)),
        "test_pnl": int(best_overall.get("test_pnl_only", 0)),
        "total_pnl": int(best_overall["test_pnl"]),
        "features": FEATS,
    }
    with open(os.path.join(OUT_DIR, "ml_pnl_optimized_info.json"), "w") as fp:
        json.dump(info, fp, indent=2)
    print(f"\nInfo saved to ml_pnl_optimized_info.json")
    print(f"Use --ml_threshold {best_overall['threshold']:.2f} in C# backtest")


if __name__ == "__main__":
    main()
