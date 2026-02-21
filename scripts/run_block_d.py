"""Run all Block D data & target experiments.

D1: Multi-horizon prediction
D2: Volatility-adjusted (ATR-normalized) target
D3: Volume feature ablation

Usage:
    python scripts/run_block_d.py
    python scripts/run_block_d.py --experiments D1a D1b
    python scripts/run_block_d.py --experiments D3a D3b D3c D3d
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

EXPERIMENTS = {
    # D1: Multi-Horizon
    "D1a": {
        "model_type": "tft", "d_model": 64, "nhead": 4, "num_layers": 2,
        "forecast_horizons": (1,),
        "desc": "TFT d=64 single horizon (baseline re-run)",
    },
    "D1b": {
        "model_type": "tft", "d_model": 64, "nhead": 4, "num_layers": 2,
        "forecast_horizons": (1, 2, 4),
        "desc": "TFT d=64 multi-horizon (1,2,4) joint training",
    },
    "D1c": {
        "model_type": "tft", "d_model": 64, "nhead": 4, "num_layers": 2,
        "forecast_horizons": (2,),
        "desc": "TFT d=64 2-bar-ahead only",
    },
    "D1d": {
        "model_type": "tft", "d_model": 64, "nhead": 4, "num_layers": 2,
        "forecast_horizons": (4,),
        "desc": "TFT d=64 4-bar-ahead only",
    },

    # D2: Volatility-Adjusted Target (delta / ATR)
    "D2a": {
        "model_type": "tft", "d_model": 64, "nhead": 4, "num_layers": 2,
        "use_atr_target": False,
        "desc": "TFT d=64 raw delta target (baseline = C4b)",
    },
    "D2b": {
        "model_type": "tft", "d_model": 64, "nhead": 4, "num_layers": 2,
        "use_atr_target": True, "atr_period": 14,
        "desc": "TFT d=64 delta/ATR(14) target",
    },
    "D2c": {
        "model_type": "tft", "d_model": 64, "nhead": 4, "num_layers": 2,
        "use_atr_target": True, "atr_period": 7,
        "desc": "TFT d=64 delta/ATR(7) target",
    },
    "D2d": {
        "model_type": "supervised", "generator": "lstm",
        "use_atr_target": True, "atr_period": 14,
        "desc": "Supervised LSTM delta/ATR(14) target",
    },

    # D3: Volume Feature Ablation
    "D3a": {
        "model_type": "tft", "d_model": 64, "nhead": 4, "num_layers": 2,
        "vol_keep": "all",
        "desc": "All volume (range+gap+vol+vwap+ATR+OBV)",
    },
    "D3b": {
        "model_type": "tft", "d_model": 64, "nhead": 4, "num_layers": 2,
        "vol_keep": "range_gap",
        "desc": "range + gap only (top-2 VSN features)",
    },
    "D3c": {
        "model_type": "tft", "d_model": 64, "nhead": 4, "num_layers": 2,
        "vol_keep": "range_gap_atr_obv",
        "desc": "range + gap + ATR + OBV (drop low-weight)",
    },
    "D3d": {
        "model_type": "tft", "d_model": 64, "nhead": 4, "num_layers": 2,
        "vol_keep": "none",
        "desc": "No volume features at all",
    },
}

# Volume feature column suffixes
VOL_SUFFIXES = ["_volume", "_range", "_vwap", "_gap", "_atr", "_obv"]

VOL_KEEP_MAP = {
    "all": {"_volume", "_range", "_vwap", "_gap", "_atr", "_obv"},
    "range_gap": {"_range", "_gap"},
    "range_gap_atr_obv": {"_range", "_gap", "_atr", "_obv"},
    "none": set(),
}


def _recompute_atr(panel, ticker, atr_period):
    """Recompute ATR with a different period and replace the column in-place."""
    import pandas as pd
    col = f"{ticker}_atr"
    if col not in panel.columns:
        return panel
    close = panel[ticker]
    prev_close = close.shift(1)
    high_col = f"{ticker}_range"
    if high_col in panel.columns:
        hl_range = panel[high_col]
    else:
        hl_range = close.diff().abs()
    tr1 = hl_range
    tr2 = (close - prev_close).abs()
    tr3 = (close - close.shift(1) - hl_range).abs() if high_col in panel.columns else tr2
    true_range = pd.concat([tr1, tr2], axis=1).max(axis=1)
    panel[col] = true_range.ewm(span=atr_period, adjust=False).mean()
    print(f"  Recomputed ATR({atr_period}) -> {col}")
    return panel


def _filter_volume_cols(panel, ticker, vol_keep):
    """Drop unwanted volume feature columns from panel."""
    if vol_keep == "all" or vol_keep not in VOL_KEEP_MAP:
        return panel

    keep_suffixes = VOL_KEEP_MAP[vol_keep]
    drop_cols = []
    for suffix in VOL_SUFFIXES:
        col = f"{ticker}{suffix}"
        if col in panel.columns and suffix not in keep_suffixes:
            drop_cols.append(col)

    if drop_cols:
        print(f"  Dropping volume columns: {drop_cols}")
        return panel.drop(columns=drop_cols)
    return panel


def _make_cfg(args, exp_params):
    """Create Config with experiment-specific parameters."""
    gen = exp_params.get("generator", "lstm")
    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_years=args.test_years, generator=gen,
        data_source=args.data_source, timeframe=args.timeframe,
        data_path=args.data_path, seq_len=args.seq_len,
    )
    cfg.model_type = exp_params["model_type"]
    cfg.loss_fn = exp_params.get("loss_fn", "huber")
    cfg.lr_g = args.lr_g
    cfg.lr_d = args.lr_d

    if "d_model" in exp_params:
        cfg.d_model = exp_params["d_model"]
    if "nhead" in exp_params:
        cfg.nhead = exp_params["nhead"]
    if "num_layers" in exp_params:
        cfg.num_layers = exp_params["num_layers"]
    if "forecast_horizons" in exp_params:
        cfg.forecast_horizons = exp_params["forecast_horizons"]
    if "use_atr_target" in exp_params:
        cfg.use_atr_target = exp_params["use_atr_target"]
    if "atr_period" in exp_params:
        cfg.atr_period = exp_params["atr_period"]

    cfg.apply_timeframe_defaults()
    return cfg


def run_experiment(panel, cfg, exp_name, exp_params):
    """Run one D-block experiment."""
    desc = exp_params["desc"]
    vol_keep = exp_params.get("vol_keep", "all")
    set_global_seed(SEED, deterministic=cfg.deterministic)

    t0 = time.time()

    work_panel = _filter_volume_cols(panel.copy(), cfg.ticker, vol_keep)

    atr_period = getattr(cfg, 'atr_period', 14)
    if atr_period != 14 and cfg.use_atr_target:
        work_panel = _recompute_atr(work_panel, cfg.ticker, atr_period)

    train_panel, test_panel = train_test_split_by_years(work_panel, cfg.test_years)

    horizons = getattr(cfg, 'forecast_horizons', (1,))
    is_multi = len(horizons) > 1

    print(f"\n{'='*65}")
    print(f"  {exp_name}: {desc}")
    print(f"  model={cfg.model_type}, horizons={horizons}, "
          f"atr_target={cfg.use_atr_target}")
    if vol_keep != "all":
        print(f"  vol_keep={vol_keep}")
    print(f"  Train: {train_panel.shape}, Test: {test_panel.shape}")
    print(f"{'='*65}")

    feat_train, feat_test = build_features_safe(train_panel, test_panel, cfg)

    raw_feat_train = feat_train.copy()
    raw_feat_test = feat_test.copy()

    feat_train, feat_test = fit_transforms(feat_train, feat_test, cfg)

    use_atr = getattr(cfg, 'use_atr_target', False)
    met, yte, y_pred, y_logit, y_q, model = run_one_split(
        feat_train, feat_test, cfg, verbose=True,
        raw_train_df=raw_feat_train if use_atr else None,
        raw_test_df=raw_feat_test if use_atr else None,
    )

    elapsed = time.time() - t0
    met["time_sec"] = round(elapsed, 1)
    met["n_features"] = feat_train.shape[1]
    met["train_samples"] = len(feat_train)
    met["test_samples"] = len(feat_test)

    if cfg.model_type == "tft" and hasattr(model, 'get_var_weights'):
        vsn_w = model.get_var_weights()
        if vsn_w is not None:
            col_list = list(feat_train.columns)
            print(f"\n  VSN Variable Importance Weights:")
            sorted_idx = np.argsort(-vsn_w)
            for idx in sorted_idx[:10]:
                name = col_list[idx] if idx < len(col_list) else f"var_{idx}"
                print(f"    {name:20s}: {vsn_w[idx]:.4f}")

    if is_multi:
        h_strs = [f"MAE_h{h}={met.get(f'MAE_h{h}', float('nan')):.4f}" for h in horizons]
        d_strs = [f"DirAcc_h{h}={met.get(f'DirAcc_h{h}', float('nan')):.3f}" for h in horizons]
        print(f"\n  Per-horizon: {', '.join(h_strs)}")
        print(f"               {', '.join(d_strs)}")
    else:
        m = met.get('MAE', float('nan'))
        d = met.get('DirAcc', float('nan'))
        ms = f"{m:.4f}" if m == m else "NaN"
        ds = f"{d:.3f}" if d == d else "NaN"
        p = met.get('PinballLoss', float('nan'))
        ps = f"{p:.4f}" if p == p else "NaN"
        print(f"\n  => MAE={ms}  DirAcc={ds}  Pinball={ps}  Time={elapsed:.1f}s")
        if "DirAcc_norm" in met:
            print(f"     DirAcc_norm (on ATR-normalized) = {met['DirAcc_norm']:.3f}")

    return met


def main():
    parser = argparse.ArgumentParser(description="Block D: Data & Target Experiments")
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2021-12-31")
    parser.add_argument("--test_years", type=int, default=1)
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--data_source", default="local")
    parser.add_argument("--data_path", default=r"G:\data2")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--lr_g", type=float, default=2e-3)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Specific experiments to run (e.g. D1a D1b). Default: all")
    parser.add_argument("--output", default="outputs/experiments/block_d_results.csv")
    args = parser.parse_args()

    exp_list = args.experiments or list(EXPERIMENTS.keys())
    print(f"Block D: Running {len(exp_list)} experiments on {args.ticker} {args.timeframe}")
    print(f"Experiments: {', '.join(exp_list)}")

    first_exp = EXPERIMENTS[exp_list[0]]
    cfg_load = _make_cfg(args, first_exp)
    print(f"\nLoading panel for {args.ticker} ({args.data_source}, {args.timeframe})...")
    t_load = time.time()
    panel = build_panel_auto(cfg_load)
    print(f"Panel loaded: {panel.shape} in {time.time()-t_load:.1f}s")
    print(f"Panel columns: {list(panel.columns)}")

    _log_device_info()

    results = []
    for exp_name in exp_list:
        if exp_name not in EXPERIMENTS:
            print(f"  WARNING: Unknown experiment '{exp_name}', skipping")
            continue

        exp_params = EXPERIMENTS[exp_name]
        cfg = _make_cfg(args, exp_params)

        try:
            met = run_experiment(panel, cfg, exp_name, exp_params)
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
        met["description"] = exp_params["desc"]
        horizons_str = str(getattr(cfg, 'forecast_horizons', (1,)))
        met["horizons"] = horizons_str
        met["atr_target"] = str(getattr(cfg, 'use_atr_target', False))
        met["vol_keep"] = exp_params.get("vol_keep", "all")
        results.append(met)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    keys = [
        "experiment", "model_type", "description",
        "horizons", "atr_target", "vol_keep",
        "MAE", "MAPE", "DirAcc", "PinballLoss",
        "DirAcc_norm",
        "MAE_h1", "MAE_h2", "MAE_h4", "DirAcc_h1", "DirAcc_h2", "DirAcc_h4",
        "MAE_avg", "DirAcc_avg",
        "time_sec", "n_features", "train_samples", "test_samples",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # Also append to master results log
    log_path = "outputs/experiments/results_log.csv"
    log_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        log_keys = ["experiment", "model_type", "MAE", "MAPE", "DirAcc",
                     "PinballLoss", "time_sec", "n_features",
                     "train_samples", "test_samples", "description"]
        w = csv.DictWriter(f, fieldnames=log_keys, extrasaction="ignore")
        if not log_exists:
            w.writeheader()
        w.writerows(results)

    print(f"\n{'='*90}")
    print(f"BLOCK D RESULTS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Exp':<6} {'Model':<12} {'Horizons':<12} {'ATR':>5} "
          f"{'MAE':>8} {'DirAcc':>8} {'Pinball':>8} {'Time':>8}")
    print("-" * 90)
    for r in results:
        m = r.get('MAE', float('nan'))
        d = r.get('DirAcc', float('nan'))
        p = r.get('PinballLoss', float('nan'))
        ms = f"{m:.4f}" if m == m else "NaN"
        ds = f"{d:.3f}" if d == d else "NaN"
        ps = f"{p:.4f}" if p == p else "NaN"
        print(f"{r['experiment']:<6} {r['model_type']:<12} "
              f"{r.get('horizons','(1,)'):<12} "
              f"{r.get('atr_target','False'):>5} "
              f"{ms:>8} {ds:>8} {ps:>8} "
              f"{r.get('time_sec', 0):>7.1f}s")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
