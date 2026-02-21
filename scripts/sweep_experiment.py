"""Run multiple experiments varying one parameter, loading data only once.

Usage:
    python scripts/sweep_experiment.py --sweep seq_len --values 12 24 48 78 156
    python scripts/sweep_experiment.py --sweep lr --values "5e-4,5e-5" "1e-3,1e-4"
    python scripts/sweep_experiment.py --sweep loss_weights --values "1.0,0.4,0.2,0.3" "1.0,1.0,0.2,0.3"
    python scripts/sweep_experiment.py --sweep ablation --values full no_fourier no_ae no_pca no_corr no_indicators
"""
import argparse
import csv
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import build_panel_auto
from src.dataset import train_test_split_by_years
from src.train import (
    build_features_safe, fit_transforms, run_one_split,
    set_global_seed, _log_device_info,
)

SEED = 42


def run_experiment(panel, cfg, exp_name, ablation_mode=None):
    """Run one experiment with pre-loaded panel. Returns metrics dict."""
    set_global_seed(SEED, deterministic=cfg.deterministic)

    t0 = time.time()

    train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"  seq_len={cfg.seq_len}, lr_g={cfg.lr_g}, lr_d={cfg.lr_d}, gen={cfg.generator}")
    print(f"  adv={cfg.adv_weight}, l1={cfg.l1_weight}, cls={cfg.cls_weight}, q={cfg.q_weight}")
    if ablation_mode:
        print(f"  ablation: {ablation_mode}")
    print(f"  Train panel: {train_panel.shape}, Test panel: {test_panel.shape}")
    print(f"{'='*60}")

    # Ablation: drop columns before feature engineering
    if ablation_mode == "no_corr":
        keep = [cfg.ticker]
        train_panel = train_panel[[c for c in train_panel.columns if c == cfg.ticker
                                   or not any(train_panel[c].dtype == float for _ in [1])
                                   or c.startswith(("SMA", "EMA", "MACD", "BB", "RSI",
                                                    "Momentum", "Volume"))]]
        # Simpler: keep only columns that start with ticker name or indicators
        ticker_cols = [c for c in train_panel.columns
                       if c == cfg.ticker or c.startswith(("SMA", "EMA", "MACD", "BB",
                                                           "RSI", "Momentum", "Volume",
                                                           "OBV", "VWAP"))]
        train_panel = train_panel[ticker_cols]
        test_panel = test_panel[ticker_cols]
    elif ablation_mode == "no_indicators":
        ind_prefixes = ("SMA", "EMA", "MACD", "BB_", "RSI", "Momentum")
        keep = [c for c in train_panel.columns if not any(c.startswith(p) for p in ind_prefixes)]
        train_panel = train_panel[keep]
        test_panel = test_panel[keep]

    feat_train, feat_test = build_features_safe(train_panel, test_panel, cfg)

    # Ablation: drop feature-engineered columns
    if ablation_mode == "no_fourier":
        fft_cols = [c for c in feat_train.columns if c.startswith("fft_")]
        feat_train = feat_train.drop(columns=fft_cols, errors="ignore")
        feat_test = feat_test.drop(columns=fft_cols, errors="ignore")

    feat_train, feat_test = fit_transforms(feat_train, feat_test, cfg)

    # Ablation: drop AE/PCA columns after transforms
    if ablation_mode == "no_ae":
        ae_cols = [c for c in feat_train.columns if c.startswith("ae_")]
        feat_train = feat_train.drop(columns=ae_cols, errors="ignore")
        feat_test = feat_test.drop(columns=ae_cols, errors="ignore")
    elif ablation_mode == "no_pca":
        pca_cols = [c for c in feat_train.columns if c.startswith("PC")]
        feat_train = feat_train.drop(columns=pca_cols, errors="ignore")
        feat_test = feat_test.drop(columns=pca_cols, errors="ignore")

    met, yte_arr, y_pred, y_logit, y_q, _ = run_one_split(
        feat_train, feat_test, cfg
    )

    elapsed = time.time() - t0
    met["time_sec"] = round(elapsed, 1)
    met["n_features"] = feat_train.shape[1]
    met["train_samples"] = len(feat_train)
    met["test_samples"] = len(feat_test)

    print(f"  -> MAE={met['MAE']:.4f}  MAPE={met['MAPE']:.4f}  "
          f"DirAcc={met['DirAcc']:.3f}  PinballLoss={met['PinballLoss']:.4f}  "
          f"Features={feat_train.shape[1]}  Time={elapsed:.1f}s")

    return met


def _make_cfg(args):
    """Create a fresh Config with Phase 1 best defaults applied."""
    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_years=args.test_years, generator=args.generator,
        data_source=args.data_source, timeframe=args.timeframe,
        data_path=args.data_path, seq_len=args.seq_len,
    )
    cfg.model_type = args.model_type
    cfg.loss_fn = args.loss_fn
    cfg.n_classes = args.n_classes
    cfg.cls_threshold = args.cls_threshold
    if args.lr_g is not None:
        cfg.lr_g = args.lr_g
    if args.lr_d is not None:
        cfg.lr_d = args.lr_d
    cfg.apply_timeframe_defaults()
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2021-12-31")
    parser.add_argument("--test_years", type=int, default=1)
    parser.add_argument("--generator", default="tst")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--data_source", default="local")
    parser.add_argument("--data_path", default=r"G:\data2")
    parser.add_argument("--model_type", default="gan",
                        choices=["gan", "supervised", "classifier", "cross_attn", "tft"],
                        help="Model type to use")
    parser.add_argument("--loss_fn", default="huber",
                        choices=["huber", "mse", "focal", "ce"],
                        help="Loss function for supervised/classifier models")
    parser.add_argument("--n_classes", type=int, default=2,
                        help="Number of classes for classifier (2 or 3)")
    parser.add_argument("--cls_threshold", type=float, default=0.0,
                        help="Classification threshold for flat class")
    parser.add_argument("--sweep", required=True,
                        choices=["seq_len", "lr", "loss_weights", "ablation"],
                        help="Parameter to sweep")
    parser.add_argument("--values", nargs="+", required=True,
                        help="Values to try.")
    parser.add_argument("--seq_len", type=int, default=12,
                        help="Base seq_len")
    parser.add_argument("--lr_g", type=float, default=None,
                        help="Override lr_g (generator learning rate)")
    parser.add_argument("--lr_d", type=float, default=None,
                        help="Override lr_d (discriminator learning rate)")
    parser.add_argument("--output", default="outputs/experiments/sweep_results.csv")
    args = parser.parse_args()

    cfg_base = _make_cfg(args)

    print(f"Loading panel for {args.ticker} ({args.data_source}, {args.timeframe})...")
    t_load = time.time()
    panel = build_panel_auto(cfg_base)
    print(f"Panel loaded: {panel.shape} in {time.time()-t_load:.1f}s")

    _log_device_info()

    results = []
    for val in args.values:
        cfg = _make_cfg(args)
        ablation_mode = None

        if args.sweep == "seq_len":
            cfg.seq_len = int(val)
            exp_name = f"seq_{val}"

        elif args.sweep == "lr":
            parts = val.split(",")
            cfg.lr_g = float(parts[0])
            cfg.lr_d = float(parts[1])
            exp_name = f"lr_{parts[0]}_{parts[1]}"

        elif args.sweep == "loss_weights":
            parts = val.split(",")
            cfg.adv_weight = float(parts[0])
            cfg.l1_weight = float(parts[1])
            cfg.cls_weight = float(parts[2])
            cfg.q_weight = float(parts[3])
            exp_name = f"lw_a{parts[0]}_l{parts[1]}_c{parts[2]}_q{parts[3]}"

        elif args.sweep == "ablation":
            ablation_mode = val
            exp_name = f"abl_{val}"

        try:
            met = run_experiment(panel, cfg, exp_name, ablation_mode=ablation_mode)
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
        met["seq_len"] = cfg.seq_len
        met["lr_g"] = cfg.lr_g
        met["lr_d"] = cfg.lr_d
        met["generator"] = cfg.generator
        met["adv_weight"] = cfg.adv_weight
        met["l1_weight"] = cfg.l1_weight
        met["cls_weight"] = cfg.cls_weight
        met["q_weight"] = cfg.q_weight
        results.append(met)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    keys = ["experiment", "model_type", "seq_len", "lr_g", "lr_d", "generator",
            "adv_weight", "l1_weight", "cls_weight", "q_weight",
            "MAE", "MAPE", "DirAcc", "PinballLoss",
            "time_sec", "n_features", "train_samples", "test_samples"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*60}")
    print(f"Sweep complete! Results saved to {args.output}")
    print(f"{'='*60}")
    print(f"\n{'experiment':<30} {'MAE':>8} {'DirAcc':>8} {'Pinball':>8} {'Feat':>5} {'Time':>8}")
    print("-" * 72)
    for r in results:
        mae = r['MAE']
        da = r['DirAcc']
        pb = r['PinballLoss']
        mae_s = f"{mae:.4f}" if not (mae != mae) else "NaN"
        da_s = f"{da:.3f}" if not (da != da) else "NaN"
        pb_s = f"{pb:.4f}" if not (pb != pb) else "NaN"
        print(f"{r['experiment']:<30} {mae_s:>8} {da_s:>8} {pb_s:>8} "
              f"{r['n_features']:>5} {r['time_sec']:>7.1f}s")


if __name__ == "__main__":
    main()
