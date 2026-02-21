"""Run all Block C architectural experiments.

Executes C1a-C1d, C2a-C2d, C3a-C3c, C4a-C4c sequentially,
loading the panel once for efficiency.

Usage:
    python scripts/run_block_c.py
    python scripts/run_block_c.py --experiments C1a C1b
    python scripts/run_block_c.py --experiments C2a C2b C2c C2d
"""
import argparse
import csv
import os
import sys
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import build_panel_auto
from src.dataset import train_test_split_by_years
from src.train import (
    build_features_safe, fit_transforms, run_one_split,
    set_global_seed, _log_device_info,
)

SEED = 42

# All experiment definitions
EXPERIMENTS = {
    # C1: Supervised (drop GAN wrapper)
    "C1a": {"model_type": "supervised", "loss_fn": "huber", "generator": "lstm",
            "l1_weight": 0.4, "cls_weight": 0.2, "q_weight": 0.3,
            "desc": "Supervised LSTM + Huber + cls + q"},
    "C1b": {"model_type": "supervised", "loss_fn": "mse", "generator": "lstm",
            "l1_weight": 0.4, "cls_weight": 0.2, "q_weight": 0.3,
            "desc": "Supervised LSTM + MSE + cls + q"},
    "C1c": {"model_type": "supervised", "loss_fn": "huber", "generator": "lstm",
            "l1_weight": 1.0, "cls_weight": 0.0, "q_weight": 0.0,
            "desc": "Supervised LSTM + Huber only (minimal)"},
    "C1d": {"model_type": "supervised", "loss_fn": "huber", "generator": "tst",
            "l1_weight": 0.4, "cls_weight": 0.2, "q_weight": 0.3,
            "desc": "Supervised TST + Huber + cls + q"},

    # C2: Classification (predict direction)
    "C2a": {"model_type": "classifier", "loss_fn": "ce", "generator": "lstm",
            "n_classes": 2, "cls_threshold": 0.0,
            "desc": "Classifier 2-class (up/down) + CrossEntropy"},
    "C2b": {"model_type": "classifier", "loss_fn": "ce", "generator": "lstm",
            "n_classes": 3, "cls_threshold": "auto",
            "desc": "Classifier 3-class (up/flat/down) + CrossEntropy"},
    "C2c": {"model_type": "classifier", "loss_fn": "focal", "generator": "lstm",
            "n_classes": 2, "cls_threshold": 0.0,
            "desc": "Classifier 2-class + FocalLoss(gamma=2)"},
    "C2d": {"model_type": "classifier", "loss_fn": "ce", "generator": "lstm",
            "n_classes": 2, "cls_threshold": 0.0,
            "desc": "Classifier 2-class + CE (confidence filtering at eval)"},

    # C3: Cross-Attention
    "C3a": {"model_type": "cross_attn", "loss_fn": "huber", "generator": "tst",
            "d_model": 64, "nhead": 4, "num_layers_tst": 2,
            "desc": "CrossAttn d=64, h=4, L=2"},
    "C3b": {"model_type": "cross_attn", "loss_fn": "huber", "generator": "tst",
            "d_model": 32, "nhead": 2, "num_layers_tst": 1,
            "desc": "CrossAttn d=32, h=2, L=1 (small)"},
    "C3c": {"model_type": "cross_attn", "loss_fn": "huber", "generator": "tst",
            "d_model": 64, "nhead": 4, "num_layers_tst": 2,
            "l1_weight": 0.4, "cls_weight": 0.2, "q_weight": 0.3,
            "desc": "CrossAttn d=64 + supervised loss"},

    # C4: Simplified TFT
    "C4a": {"model_type": "tft", "loss_fn": "huber", "generator": "lstm",
            "d_model": 32, "nhead": 2, "num_layers": 1,
            "desc": "TFT d=32, h=2, 1 LSTM layer (minimal)"},
    "C4b": {"model_type": "tft", "loss_fn": "huber", "generator": "lstm",
            "d_model": 64, "nhead": 4, "num_layers": 2,
            "desc": "TFT d=64, h=4, 2 LSTM layers (full)"},
    "C4c": {"model_type": "tft", "loss_fn": "huber", "generator": "lstm",
            "d_model": 32, "nhead": 2, "num_layers": 1,
            "desc": "TFT d=32 (print VSN weights)"},
}


def _make_cfg(args, exp_params):
    """Create Config with experiment-specific parameters."""
    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_years=args.test_years, generator=exp_params.get("generator", "lstm"),
        data_source=args.data_source, timeframe=args.timeframe,
        data_path=args.data_path, seq_len=args.seq_len,
    )
    cfg.model_type = exp_params["model_type"]
    cfg.loss_fn = exp_params.get("loss_fn", "huber")
    cfg.lr_g = args.lr_g
    cfg.lr_d = args.lr_d

    if "l1_weight" in exp_params:
        cfg.l1_weight = exp_params["l1_weight"]
    if "cls_weight" in exp_params:
        cfg.cls_weight = exp_params["cls_weight"]
    if "q_weight" in exp_params:
        cfg.q_weight = exp_params["q_weight"]
    if "n_classes" in exp_params:
        cfg.n_classes = exp_params["n_classes"]
    if "cls_threshold" in exp_params and exp_params["cls_threshold"] != "auto":
        cfg.cls_threshold = exp_params["cls_threshold"]
    if "d_model" in exp_params:
        cfg.d_model = exp_params["d_model"]
    if "nhead" in exp_params:
        cfg.nhead = exp_params["nhead"]
    if "num_layers_tst" in exp_params:
        cfg.num_layers_tst = exp_params["num_layers_tst"]
    if "num_layers" in exp_params:
        cfg.num_layers = exp_params["num_layers"]

    cfg.apply_timeframe_defaults()
    return cfg


def run_experiment(panel, cfg, exp_name, desc, col_names=None):
    """Run one experiment with pre-loaded panel."""
    set_global_seed(SEED, deterministic=cfg.deterministic)

    t0 = time.time()
    train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)

    # Auto-compute classification threshold from train deltas if needed
    if cfg.model_type == "classifier" and cfg.n_classes == 3 and cfg.cls_threshold == 0.0:
        deltas = train_panel[cfg.ticker].diff().dropna()
        cfg.cls_threshold = float(np.percentile(np.abs(deltas), 50))
        print(f"  Auto cls_threshold = {cfg.cls_threshold:.6f} (median |delta|)")

    print(f"\n{'='*65}")
    print(f"  {exp_name}: {desc}")
    print(f"  model={cfg.model_type}, gen={cfg.generator}, loss={cfg.loss_fn}")
    print(f"  l1={cfg.l1_weight}, cls={cfg.cls_weight}, q={cfg.q_weight}")
    print(f"  Train: {train_panel.shape}, Test: {test_panel.shape}")
    print(f"{'='*65}")

    feat_train, feat_test = build_features_safe(train_panel, test_panel, cfg)
    feat_train, feat_test = fit_transforms(feat_train, feat_test, cfg)

    met, yte, y_pred, y_logit, y_q, model = run_one_split(
        feat_train, feat_test, cfg, verbose=True)

    elapsed = time.time() - t0
    met["time_sec"] = round(elapsed, 1)
    met["n_features"] = feat_train.shape[1]
    met["train_samples"] = len(feat_train)
    met["test_samples"] = len(feat_test)

    # Compute direction accuracy on high-confidence predictions (for C2d)
    if cfg.model_type == "classifier":
        if hasattr(model, 'net'):
            import torch
            from torch.utils.data import TensorDataset, DataLoader
            from src.dataset import make_sequences
            Xte, yte_seq = make_sequences(feat_test, target_col=cfg.ticker, seq_len=cfg.seq_len)
            te_ds = TensorDataset(torch.tensor(Xte))
            te_dl = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False)
            all_logits = []
            model.net.eval()
            with torch.no_grad():
                for (xb,) in te_dl:
                    xb = xb.to(model.device)
                    logits = model.net(xb)
                    all_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())
            probs = np.concatenate(all_logits)
            max_probs = probs.max(axis=1)
            high_conf_mask = max_probs > 0.6
            if high_conf_mask.sum() > 10:
                if cfg.n_classes == 2:
                    preds_dir = (probs[:, 1] > 0.5).astype(float) * 2 - 1
                else:
                    preds_dir = np.sign(probs[:, 2] - probs[:, 0])
                true_dir = np.sign(yte_seq)
                hc_acc = float(np.mean(preds_dir[high_conf_mask] == true_dir[high_conf_mask]))
                met["HighConfDirAcc"] = round(hc_acc, 4)
                met["HighConfCount"] = int(high_conf_mask.sum())
                met["HighConfPct"] = round(100 * high_conf_mask.sum() / len(high_conf_mask), 1)

    # Print VSN weights for TFT experiments
    if cfg.model_type == "tft" and hasattr(model, 'get_var_weights'):
        vsn_w = model.get_var_weights()
        if vsn_w is not None:
            col_list = list(feat_train.columns)
            print(f"\n  VSN Variable Importance Weights:")
            sorted_idx = np.argsort(-vsn_w)
            for idx in sorted_idx:
                name = col_list[idx] if idx < len(col_list) else f"var_{idx}"
                print(f"    {name:20s}: {vsn_w[idx]:.4f}")

    print(f"\n  => MAE={met['MAE']:.4f}  MAPE={met['MAPE']:.4f}  "
          f"DirAcc={met['DirAcc']:.3f}  Pinball={met['PinballLoss']:.4f}  "
          f"Time={elapsed:.1f}s")

    return met


def main():
    parser = argparse.ArgumentParser(description="Block C: Architectural Experiments")
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2021-12-31")
    parser.add_argument("--test_years", type=int, default=1)
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--data_source", default="local")
    parser.add_argument("--data_path", default=r"G:\data2")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--lr_g", type=float, default=2e-3)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Specific experiments to run (e.g. C1a C1b). Default: all")
    parser.add_argument("--output", default="outputs/experiments/block_c_results.csv")
    args = parser.parse_args()

    exp_list = args.experiments or list(EXPERIMENTS.keys())
    print(f"Block C: Running {len(exp_list)} experiments on {args.ticker} {args.timeframe}")
    print(f"Experiments: {', '.join(exp_list)}")

    # Load panel once
    first_exp = EXPERIMENTS[exp_list[0]]
    cfg_load = _make_cfg(args, first_exp)
    print(f"\nLoading panel for {args.ticker} ({args.data_source}, {args.timeframe})...")
    t_load = time.time()
    panel = build_panel_auto(cfg_load)
    print(f"Panel loaded: {panel.shape} in {time.time()-t_load:.1f}s")

    _log_device_info()

    results = []
    for exp_name in exp_list:
        if exp_name not in EXPERIMENTS:
            print(f"  WARNING: Unknown experiment '{exp_name}', skipping")
            continue

        exp_params = EXPERIMENTS[exp_name]
        cfg = _make_cfg(args, exp_params)

        try:
            met = run_experiment(panel, cfg, exp_name, exp_params["desc"])
        except Exception as e:
            print(f"  !! FAILED: {e}")
            traceback.print_exc()
            met = {
                "MAE": float("nan"), "MAPE": float("nan"),
                "DirAcc": float("nan"), "PinballLoss": float("nan"),
                "time_sec": 0, "n_features": 0,
                "train_samples": 0, "test_samples": 0,
            }

        met["experiment"] = exp_name
        met["model_type"] = cfg.model_type
        met["loss_fn"] = cfg.loss_fn
        met["generator"] = cfg.generator
        met["description"] = exp_params["desc"]
        results.append(met)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    keys = ["experiment", "model_type", "loss_fn", "generator", "description",
            "MAE", "MAPE", "DirAcc", "PinballLoss",
            "HighConfDirAcc", "HighConfCount", "HighConfPct",
            "time_sec", "n_features", "train_samples", "test_samples"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # Summary table
    print(f"\n{'='*80}")
    print(f"BLOCK C RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Exp':<6} {'Model':<12} {'Loss':<8} {'Gen':<5} "
          f"{'MAE':>8} {'DirAcc':>8} {'Pinball':>8} {'Time':>8}")
    print("-" * 80)
    for r in results:
        m = r.get('MAE', float('nan'))
        d = r.get('DirAcc', float('nan'))
        p = r.get('PinballLoss', float('nan'))
        ms = f"{m:.4f}" if m == m else "NaN"
        ds = f"{d:.3f}" if d == d else "NaN"
        ps = f"{p:.4f}" if p == p else "NaN"
        print(f"{r['experiment']:<6} {r['model_type']:<12} {r['loss_fn']:<8} "
              f"{r['generator']:<5} {ms:>8} {ds:>8} {ps:>8} "
              f"{r.get('time_sec', 0):>7.1f}s")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
