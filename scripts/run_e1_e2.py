"""E1: TFT h=4 on H1   +   E2: Walk-forward TFT h=4 on M5
    E3: Walk-forward TFT h=4 on H1.

Usage:
    python scripts/run_e1_e2.py --experiment E1
    python scripts/run_e1_e2.py --experiment E2
    python scripts/run_e1_e2.py --experiment E3
    python scripts/run_e1_e2.py                   # E1+E2
"""
import argparse
import csv
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import build_panel_auto
from src.dataset import train_test_split_by_years, walk_forward_splits
from src.train import (
    build_features_safe, fit_transforms, run_one_split,
    set_global_seed, _log_device_info,
)

SEED = 42
OUT_DIR = "outputs/experiments"


def run_e1(args):
    """E1: TFT h=4 on H1 (single-split, compare h=1 vs h=4)."""
    print("=" * 70)
    print("  E1: TFT h=4 on H1 -- does longer horizon help on hourly data?")
    print("=" * 70)

    horizons_list = [(1,), (4,), (2,), (8,)]
    results = []

    cfg_base = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_years=args.test_years, generator="lstm",
        data_source="local", timeframe="H1",
        data_path=args.data_path, seq_len=12,
        model_type="tft", loss_fn="huber",
    )
    cfg_base.d_model = 64
    cfg_base.nhead = 4
    cfg_base.num_layers = 2
    cfg_base.lr_g = 2e-3
    cfg_base.lr_d = 2e-4
    cfg_base.l1_weight = 0.4
    cfg_base.cls_weight = 0.2
    cfg_base.q_weight = 1.0
    cfg_base.apply_timeframe_defaults()

    print(f"\nLoading H1 panel...")
    t0 = time.time()
    panel = build_panel_auto(cfg_base)
    print(f"Panel: {panel.shape}, loaded in {time.time()-t0:.1f}s")
    _log_device_info()

    for horizons in horizons_list:
        set_global_seed(SEED)
        cfg = Config(
            ticker=args.ticker, start=args.start, end=args.end,
            test_years=args.test_years, generator="lstm",
            data_source="local", timeframe="H1",
            data_path=args.data_path, seq_len=12,
            model_type="tft", loss_fn="huber",
        )
        cfg.d_model = 64
        cfg.nhead = 4
        cfg.num_layers = 2
        cfg.lr_g = 2e-3
        cfg.lr_d = 2e-4
        cfg.l1_weight = 0.4
        cfg.cls_weight = 0.2
        cfg.q_weight = 1.0
        cfg.forecast_horizons = horizons
        cfg.apply_timeframe_defaults()

        label = f"h={horizons}"
        print(f"\n{'='*60}")
        print(f"  E1 run: TFT d=64 on H1, horizons={horizons}")
        print(f"{'='*60}")

        t1 = time.time()
        train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)
        print(f"  Train: {train_panel.shape}, Test: {test_panel.shape}")

        feat_train, feat_test = build_features_safe(train_panel, test_panel, cfg)
        feat_train, feat_test = fit_transforms(feat_train, feat_test, cfg)

        met, yte, y_pred, _, _, model = run_one_split(
            feat_train, feat_test, cfg, verbose=True)

        elapsed = time.time() - t1
        met["horizons"] = str(horizons)
        met["time_sec"] = round(elapsed, 1)
        met["n_features"] = feat_train.shape[1]

        if cfg.model_type == "tft" and hasattr(model, 'get_var_weights'):
            vsn_w = model.get_var_weights()
            if vsn_w is not None:
                col_list = list(feat_train.columns)
                print(f"\n  VSN top-5:")
                for idx in np.argsort(-vsn_w)[:5]:
                    name = col_list[idx] if idx < len(col_list) else f"var_{idx}"
                    print(f"    {name:20s}: {vsn_w[idx]:.4f}")

        m = met.get('MAE', float('nan'))
        d = met.get('DirAcc', float('nan'))
        print(f"\n  => {label}: MAE={m:.4f}  DirAcc={d:.3f}  Time={elapsed:.1f}s")
        results.append(met)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "e1_tft_h4_h1.csv")
    keys = ["horizons", "MAE", "MAPE", "DirAcc", "PinballLoss",
            "time_sec", "n_features"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*70}")
    print(f"E1 SUMMARY (H1, TFT d=64)")
    print(f"{'='*70}")
    print(f"{'Horizons':<15} {'MAE':>8} {'DirAcc':>8} {'Time':>8}")
    print("-" * 45)
    for r in results:
        m = r.get('MAE', float('nan'))
        d = r.get('DirAcc', float('nan'))
        print(f"{r['horizons']:<15} {m:>8.4f} {d:>8.3f} {r.get('time_sec',0):>7.1f}s")
    print(f"\nSaved to {out_path}")


def run_e2(args):
    """E2: Walk-forward TFT h=4 on M5 (5+ splits)."""
    print("=" * 70)
    print("  E2: Walk-forward TFT h=4 on M5 -- stability check")
    print("=" * 70)

    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_years=args.test_years, generator="lstm",
        data_source="local", timeframe="M5",
        data_path=args.data_path, seq_len=12,
        model_type="tft", loss_fn="huber",
    )
    cfg.d_model = 64
    cfg.nhead = 4
    cfg.num_layers = 2
    cfg.lr_g = 2e-3
    cfg.lr_d = 2e-4
    cfg.l1_weight = 0.4
    cfg.cls_weight = 0.2
    cfg.q_weight = 1.0
    cfg.forecast_horizons = (4,)
    cfg.apply_timeframe_defaults()

    print(f"\nLoading M5 panel...")
    t0 = time.time()
    panel = build_panel_auto(cfg)
    print(f"Panel: {panel.shape}, loaded in {time.time()-t0:.1f}s")
    print(f"Walk-forward: {cfg.wf_splits} splits, "
          f"min_train={cfg.wf_min_train}, step={cfg.wf_step}")
    _log_device_info()

    splits = list(walk_forward_splits(
        panel, n_splits=cfg.wf_splits,
        min_train=cfg.wf_min_train, step=cfg.wf_step,
    ))
    print(f"Generated {len(splits)} splits")

    metrics_all = []
    for i, (tr_idx, te_idx) in enumerate(splits, 1):
        set_global_seed(SEED)
        tr_panel = panel.iloc[tr_idx]
        te_panel = panel.iloc[te_idx]
        print(f"\n{'='*60}")
        print(f"  Split {i}/{len(splits)}: train={len(tr_panel)} test={len(te_panel)}")
        print(f"{'='*60}")

        t1 = time.time()
        feat_train, feat_test = build_features_safe(tr_panel, te_panel, cfg)
        feat_train, feat_test = fit_transforms(feat_train, feat_test, cfg)

        met, yte, y_pred, _, _, model = run_one_split(
            feat_train, feat_test, cfg, verbose=True)

        elapsed = time.time() - t1
        met["split"] = i
        met["train_bars"] = len(tr_panel)
        met["test_bars"] = len(te_panel)
        met["time_sec"] = round(elapsed, 1)

        m = met.get('MAE', float('nan'))
        d = met.get('DirAcc', float('nan'))
        print(f"\n  Split {i}: MAE={m:.4f}  DirAcc={d:.3f}  Time={elapsed:.1f}s")
        metrics_all.append(met)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "e2_wf_tft_h4_m5.csv")
    keys = ["split", "train_bars", "test_bars",
            "MAE", "MAPE", "DirAcc", "PinballLoss", "time_sec"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(metrics_all)

    # Summary
    dfm = pd.DataFrame(metrics_all)
    print(f"\n{'='*70}")
    print(f"E2 WALK-FORWARD SUMMARY (M5, TFT d=64, h=4)")
    print(f"{'='*70}")
    print(f"{'Split':<8} {'Train':>8} {'Test':>8} {'MAE':>8} {'DirAcc':>8} {'Time':>8}")
    print("-" * 55)
    for r in metrics_all:
        m = r.get('MAE', float('nan'))
        d = r.get('DirAcc', float('nan'))
        print(f"{r['split']:<8} {r['train_bars']:>8} {r['test_bars']:>8} "
              f"{m:>8.4f} {d:>8.3f} {r.get('time_sec',0):>7.1f}s")
    print("-" * 55)

    avg_mae = dfm['MAE'].mean()
    avg_dir = dfm['DirAcc'].mean()
    std_dir = dfm['DirAcc'].std()
    min_dir = dfm['DirAcc'].min()
    max_dir = dfm['DirAcc'].max()
    print(f"{'AVG':<8} {'':>8} {'':>8} {avg_mae:>8.4f} {avg_dir:>8.3f}")
    print(f"\nDirAcc: mean={avg_dir:.3f}, std={std_dir:.3f}, "
          f"min={min_dir:.3f}, max={max_dir:.3f}, spread={max_dir-min_dir:.3f}")
    print(f"Splits > 50%: {(dfm['DirAcc'] > 0.50).sum()}/{len(dfm)}")
    print(f"\nSaved to {out_path}")


def run_e3(args):
    """E3: Walk-forward TFT h=4 on H1 -- stability check for best result."""
    print("=" * 70)
    print("  E3: Walk-forward TFT h=4 on H1 -- stability check")
    print("=" * 70)

    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_years=args.test_years, generator="lstm",
        data_source="local", timeframe="H1",
        data_path=args.data_path, seq_len=12,
        model_type="tft", loss_fn="huber",
    )
    cfg.d_model = 64
    cfg.nhead = 4
    cfg.num_layers = 2
    cfg.lr_g = 2e-3
    cfg.lr_d = 2e-4
    cfg.l1_weight = 0.4
    cfg.cls_weight = 0.2
    cfg.q_weight = 1.0
    cfg.forecast_horizons = (4,)
    cfg.apply_timeframe_defaults()

    print(f"\nLoading H1 panel...")
    t0 = time.time()
    panel = build_panel_auto(cfg)
    print(f"Panel: {panel.shape}, loaded in {time.time()-t0:.1f}s")
    print(f"Walk-forward: {cfg.wf_splits} splits, "
          f"min_train={cfg.wf_min_train}, step={cfg.wf_step}")
    _log_device_info()

    splits = list(walk_forward_splits(
        panel, n_splits=cfg.wf_splits,
        min_train=cfg.wf_min_train, step=cfg.wf_step,
    ))
    print(f"Generated {len(splits)} splits")

    metrics_all = []
    for i, (tr_idx, te_idx) in enumerate(splits, 1):
        set_global_seed(SEED)
        tr_panel = panel.iloc[tr_idx]
        te_panel = panel.iloc[te_idx]
        print(f"\n{'='*60}")
        print(f"  Split {i}/{len(splits)}: train={len(tr_panel)} test={len(te_panel)}")
        print(f"{'='*60}")

        t1 = time.time()
        feat_train, feat_test = build_features_safe(tr_panel, te_panel, cfg)
        feat_train, feat_test = fit_transforms(feat_train, feat_test, cfg)

        met, yte, y_pred, _, _, model = run_one_split(
            feat_train, feat_test, cfg, verbose=True)

        elapsed = time.time() - t1
        met["split"] = i
        met["train_bars"] = len(tr_panel)
        met["test_bars"] = len(te_panel)
        met["time_sec"] = round(elapsed, 1)

        m = met.get('MAE', float('nan'))
        d = met.get('DirAcc', float('nan'))
        print(f"\n  Split {i}: MAE={m:.4f}  DirAcc={d:.3f}  Time={elapsed:.1f}s")
        metrics_all.append(met)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "e3_wf_tft_h4_h1.csv")
    keys = ["split", "train_bars", "test_bars",
            "MAE", "MAPE", "DirAcc", "PinballLoss", "time_sec"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(metrics_all)

    # Summary
    dfm = pd.DataFrame(metrics_all)
    print(f"\n{'='*70}")
    print(f"E3 WALK-FORWARD SUMMARY (H1, TFT d=64, h=4)")
    print(f"{'='*70}")
    print(f"{'Split':<8} {'Train':>8} {'Test':>8} {'MAE':>8} {'DirAcc':>8} {'Time':>8}")
    print("-" * 55)
    for r in metrics_all:
        m = r.get('MAE', float('nan'))
        d = r.get('DirAcc', float('nan'))
        print(f"{r['split']:<8} {r['train_bars']:>8} {r['test_bars']:>8} "
              f"{m:>8.4f} {d:>8.3f} {r.get('time_sec',0):>7.1f}s")
    print("-" * 55)

    avg_mae = dfm['MAE'].mean()
    avg_dir = dfm['DirAcc'].mean()
    std_dir = dfm['DirAcc'].std()
    min_dir = dfm['DirAcc'].min()
    max_dir = dfm['DirAcc'].max()
    print(f"{'AVG':<8} {'':>8} {'':>8} {avg_mae:>8.4f} {avg_dir:>8.3f}")
    print(f"\nDirAcc: mean={avg_dir:.3f}, std={std_dir:.3f}, "
          f"min={min_dir:.3f}, max={max_dir:.3f}, spread={max_dir-min_dir:.3f}")
    print(f"Splits > 50%: {(dfm['DirAcc'] > 0.50).sum()}/{len(dfm)}")
    print(f"\nSaved to {out_path}")


def _make_tft_cfg(args, timeframe="H1", horizons=(4,), n_classes=2,
                   l1_weight=0.4, cls_weight=0.2, q_weight=1.0,
                   loss_fn="huber"):
    """Helper: build a TFT d=64 config."""
    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_years=args.test_years, generator="lstm",
        data_source="local", timeframe=timeframe,
        data_path=args.data_path, seq_len=12,
        model_type="tft", loss_fn=loss_fn,
    )
    cfg.d_model = 64
    cfg.nhead = 4
    cfg.num_layers = 2
    cfg.lr_g = 2e-3
    cfg.lr_d = 2e-4
    cfg.l1_weight = l1_weight
    cfg.cls_weight = cls_weight
    cfg.q_weight = q_weight
    cfg.forecast_horizons = horizons
    cfg.n_classes = n_classes
    cfg.cls_threshold = 0.0  # auto-computed in run_one_split when n_classes>=3
    cfg.apply_timeframe_defaults()
    return cfg


def run_e3cls(args):
    """E3: TFT + 3-class head (up/flat/down) on H1 h=4."""
    print("=" * 70)
    print("  E3: TFT + 3-class head (up/flat/down) on H1 h=4")
    print("=" * 70)

    experiments = {
        "E3a_hybrid": {
            "desc": "TFT d=64, 3-class + regression (hybrid)",
            "n_classes": 3, "l1": 0.4, "cls": 0.2, "q": 1.0,
        },
        "E3b_pure_cls": {
            "desc": "TFT d=64, 3-class only (pure classifier)",
            "n_classes": 3, "l1": 0.0, "cls": 1.0, "q": 0.0,
        },
        "E3c_binary_base": {
            "desc": "TFT d=64, binary cls (baseline = E1 h=4)",
            "n_classes": 2, "l1": 0.4, "cls": 0.2, "q": 1.0,
        },
    }

    cfg_load = _make_tft_cfg(args)
    print(f"\nLoading H1 panel...")
    t0 = time.time()
    panel = build_panel_auto(cfg_load)
    print(f"Panel: {panel.shape}, loaded in {time.time()-t0:.1f}s")
    _log_device_info()

    results = []
    for exp_name, exp in experiments.items():
        set_global_seed(SEED)
        cfg = _make_tft_cfg(
            args, n_classes=exp["n_classes"],
            l1_weight=exp["l1"], cls_weight=exp["cls"], q_weight=exp["q"],
        )

        print(f"\n{'='*60}")
        print(f"  {exp_name}: {exp['desc']}")
        print(f"  n_classes={cfg.n_classes}, l1={cfg.l1_weight}, "
              f"cls={cfg.cls_weight}, q={cfg.q_weight}")
        print(f"{'='*60}")

        t1 = time.time()
        train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)
        print(f"  Train: {train_panel.shape}, Test: {test_panel.shape}")

        feat_train, feat_test = build_features_safe(train_panel, test_panel, cfg)
        feat_train, feat_test = fit_transforms(feat_train, feat_test, cfg)

        met, yte, y_pred, _, _, model = run_one_split(
            feat_train, feat_test, cfg, verbose=True)

        elapsed = time.time() - t1
        met["experiment"] = exp_name
        met["n_classes"] = exp["n_classes"]
        met["l1_weight"] = exp["l1"]
        met["cls_weight"] = exp["cls"]
        met["q_weight"] = exp["q"]
        met["time_sec"] = round(elapsed, 1)
        met["n_features"] = feat_train.shape[1]

        if cfg.model_type == "tft" and hasattr(model, 'get_var_weights'):
            vsn_w = model.get_var_weights()
            if vsn_w is not None:
                col_list = list(feat_train.columns)
                print(f"\n  VSN top-5:")
                for idx in np.argsort(-vsn_w)[:5]:
                    name = col_list[idx] if idx < len(col_list) else f"var_{idx}"
                    print(f"    {name:20s}: {vsn_w[idx]:.4f}")

        m = met.get('MAE', float('nan'))
        d = met.get('DirAcc', float('nan'))
        print(f"\n  => {exp_name}: MAE={m:.4f}  DirAcc={d:.3f}  Time={elapsed:.1f}s")
        results.append(met)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "e3_tft_3class.csv")
    keys = ["experiment", "n_classes", "l1_weight", "cls_weight", "q_weight",
            "MAE", "MAPE", "DirAcc", "PinballLoss", "time_sec", "n_features"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*70}")
    print(f"E3 SUMMARY (H1, TFT d=64, h=4, 3-class classification)")
    print(f"{'='*70}")
    print(f"{'Experiment':<20} {'Classes':>7} {'MAE':>8} {'DirAcc':>8} {'Time':>8}")
    print("-" * 60)
    for r in results:
        m = r.get('MAE', float('nan'))
        d = r.get('DirAcc', float('nan'))
        print(f"{r['experiment']:<20} {r['n_classes']:>7} {m:>8.4f} "
              f"{d:>8.3f} {r.get('time_sec',0):>7.1f}s")
    print(f"\nSaved to {out_path}")


def run_e4_ens(args):
    """E4: Ensemble TFT + Mean Reversion on H1 h=4."""
    print("=" * 70)
    print("  E4: Ensemble TFT + Mean Reversion on H1 h=4")
    print("=" * 70)

    cfg = _make_tft_cfg(args)

    print(f"\nLoading H1 panel...")
    t0 = time.time()
    panel = build_panel_auto(cfg)
    print(f"Panel: {panel.shape}, loaded in {time.time()-t0:.1f}s")
    _log_device_info()

    set_global_seed(SEED)
    train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)
    print(f"Train: {train_panel.shape}, Test: {test_panel.shape}")

    feat_train, feat_test = build_features_safe(train_panel, test_panel, cfg)
    feat_train, feat_test = fit_transforms(feat_train, feat_test, cfg)

    print("\nTraining TFT d=64 h=4...")
    t1 = time.time()
    met, yte, y_tft, _, _, model = run_one_split(
        feat_train, feat_test, cfg, verbose=True)
    tft_time = time.time() - t1
    print(f"TFT done in {tft_time:.1f}s: MAE={met['MAE']:.4f} DirAcc={met['DirAcc']:.3f}")

    # Mean Reversion predictions: pred[i] = -y_true[i-1]
    y_mr = np.zeros_like(yte)
    y_mr[1:] = -yte[:-1]

    # Evaluate on y[1:] (drop first where MR has no signal)
    yt = yte[1:]
    y_tft_trim = y_tft[1:]
    y_mr_trim = y_mr[1:]

    from src.utils.metrics import direction_accuracy, mae

    # Baseline metrics
    tft_dir = direction_accuracy(yt, y_tft_trim)
    mr_dir = direction_accuracy(yt, y_mr_trim)
    tft_mae = mae(yt, y_tft_trim)
    mr_mae = mae(yt, y_mr_trim)

    print(f"\n  TFT standalone:  MAE={tft_mae:.4f}  DirAcc={tft_dir:.3f}")
    print(f"  MR standalone:   MAE={mr_mae:.4f}  DirAcc={mr_dir:.3f}")

    # Ensemble strategies
    results = []

    # Strategy 1: Weighted average (50/50)
    for w_tft in [0.3, 0.5, 0.7]:
        y_avg = w_tft * y_tft_trim + (1 - w_tft) * y_mr_trim
        d = direction_accuracy(yt, y_avg)
        m = mae(yt, y_avg)
        label = f"Avg(TFT={w_tft:.1f},MR={1-w_tft:.1f})"
        print(f"  {label}:  MAE={m:.4f}  DirAcc={d:.3f}")
        results.append({"strategy": label, "MAE": m, "DirAcc": d,
                         "coverage": 1.0, "n_samples": len(yt)})

    # Strategy 2: Agreement filter
    tft_sign = np.sign(y_tft_trim)
    mr_sign = np.sign(y_mr_trim)
    agree_mask = (tft_sign == mr_sign) & (tft_sign != 0)

    agree_pct = agree_mask.mean()
    if agree_mask.sum() > 0:
        d_agree = direction_accuracy(yt[agree_mask], y_tft_trim[agree_mask])
        m_agree = mae(yt[agree_mask], y_tft_trim[agree_mask])
    else:
        d_agree = float('nan')
        m_agree = float('nan')
    print(f"\n  Agreement filter: coverage={agree_pct:.1%} ({agree_mask.sum()}/{len(yt)})")
    print(f"    On agreed subset: MAE={m_agree:.4f}  DirAcc={d_agree:.3f}")
    results.append({"strategy": "Agreement(TFT signal)", "MAE": m_agree,
                     "DirAcc": d_agree, "coverage": agree_pct,
                     "n_samples": int(agree_mask.sum())})

    # Strategy 3: Agreement filter -- use average prediction
    if agree_mask.sum() > 0:
        y_agree_avg = 0.5 * y_tft_trim[agree_mask] + 0.5 * y_mr_trim[agree_mask]
        d_agree_avg = direction_accuracy(yt[agree_mask], y_agree_avg)
        m_agree_avg = mae(yt[agree_mask], y_agree_avg)
    else:
        d_agree_avg = float('nan')
        m_agree_avg = float('nan')
    print(f"    On agreed (avg pred): MAE={m_agree_avg:.4f}  DirAcc={d_agree_avg:.3f}")
    results.append({"strategy": "Agreement(avg pred)", "MAE": m_agree_avg,
                     "DirAcc": d_agree_avg, "coverage": agree_pct,
                     "n_samples": int(agree_mask.sum())})

    # Strategy 4: Disagree filter (where MR overrules)
    disagree_mask = ~agree_mask
    if disagree_mask.sum() > 0:
        d_dis = direction_accuracy(yt[disagree_mask], y_mr_trim[disagree_mask])
        print(f"  Disagree subset: coverage={disagree_mask.mean():.1%} "
              f"MR DirAcc={d_dis:.3f}")
    else:
        d_dis = float('nan')

    # Strategy 5: Full ensemble: if agree -> TFT signal, if disagree -> 0 (skip)
    y_full = np.where(agree_mask, y_tft_trim, 0.0)
    d_full = direction_accuracy(yt, y_full)
    m_full = mae(yt, y_full)
    print(f"  Full ens (agree=TFT, disagree=0): MAE={m_full:.4f}  DirAcc={d_full:.3f}")
    results.append({"strategy": "Ens(agree=TFT,else=0)", "MAE": m_full,
                     "DirAcc": d_full, "coverage": 1.0,
                     "n_samples": len(yt)})

    # Add baselines
    results.insert(0, {"strategy": "TFT_standalone", "MAE": tft_mae,
                        "DirAcc": tft_dir, "coverage": 1.0,
                        "n_samples": len(yt)})
    results.insert(1, {"strategy": "MR_standalone", "MAE": mr_mae,
                        "DirAcc": mr_dir, "coverage": 1.0,
                        "n_samples": len(yt)})

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "e4_ensemble.csv")
    keys = ["strategy", "MAE", "DirAcc", "coverage", "n_samples"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*70}")
    print(f"E4 ENSEMBLE SUMMARY (H1, TFT d=64 h=4 + Mean Reversion)")
    print(f"{'='*70}")
    print(f"{'Strategy':<30} {'MAE':>8} {'DirAcc':>8} {'Cover':>8} {'N':>6}")
    print("-" * 65)
    for r in results:
        m = r.get('MAE', float('nan'))
        d = r.get('DirAcc', float('nan'))
        ms = f"{m:.4f}" if m == m else "NaN"
        ds = f"{d:.3f}" if d == d else "NaN"
        c = r.get('coverage', 0)
        print(f"{r['strategy']:<30} {ms:>8} {ds:>8} {c:>7.1%} {r['n_samples']:>6}")
    print(f"\nSaved to {out_path}")


def run_e5_lags(args):
    """E5: Delta lag features -- teach TFT the momentum signal."""
    print("=" * 70)
    print("  E5: Delta lag features (momentum signal) on H1 h=4")
    print("=" * 70)

    experiments = {
        "E5_no_lags": {
            "desc": "Baseline: TFT d=64 h=4, no delta lags",
            "use_delta_lags": False,
            "delta_lag_periods": (),
        },
        "E5_lag1": {
            "desc": "TFT + delta_lag1 only",
            "use_delta_lags": True,
            "delta_lag_periods": (1,),
        },
        "E5_lag124": {
            "desc": "TFT + delta_lag1,2,4",
            "use_delta_lags": True,
            "delta_lag_periods": (1, 2, 4),
        },
        "E5_lag1248": {
            "desc": "TFT + delta_lag1,2,4,8",
            "use_delta_lags": True,
            "delta_lag_periods": (1, 2, 4, 8),
        },
    }

    cfg_load = _make_tft_cfg(args)
    print(f"\nLoading H1 panel...")
    t0 = time.time()
    panel = build_panel_auto(cfg_load)
    print(f"Panel: {panel.shape}, loaded in {time.time()-t0:.1f}s")
    _log_device_info()

    results = []
    for exp_name, exp in experiments.items():
        set_global_seed(SEED)
        cfg = _make_tft_cfg(args)
        cfg.use_delta_lags = exp["use_delta_lags"]
        cfg.delta_lag_periods = exp["delta_lag_periods"]

        print(f"\n{'='*60}")
        print(f"  {exp_name}: {exp['desc']}")
        print(f"{'='*60}")

        t1 = time.time()
        train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)
        print(f"  Train: {train_panel.shape}, Test: {test_panel.shape}")

        feat_train, feat_test = build_features_safe(train_panel, test_panel, cfg)
        feat_train, feat_test = fit_transforms(feat_train, feat_test, cfg)
        print(f"  Features: {feat_train.shape[1]} cols")

        met, yte, y_pred, _, _, model = run_one_split(
            feat_train, feat_test, cfg, verbose=True)

        elapsed = time.time() - t1
        met["experiment"] = exp_name
        met["lags"] = str(exp["delta_lag_periods"])
        met["time_sec"] = round(elapsed, 1)
        met["n_features"] = feat_train.shape[1]

        # Compute momentum baseline for comparison
        y_mom = np.zeros_like(yte)
        y_mom[1:] = yte[:-1]
        from src.utils.metrics import direction_accuracy as da
        mom_dir = da(yte[1:], y_mom[1:])
        met["MomentumDirAcc"] = mom_dir

        if cfg.model_type == "tft" and hasattr(model, 'get_var_weights'):
            vsn_w = model.get_var_weights()
            if vsn_w is not None:
                col_list = list(feat_train.columns)
                print(f"\n  VSN top-7:")
                for idx in np.argsort(-vsn_w)[:7]:
                    name = col_list[idx] if idx < len(col_list) else f"var_{idx}"
                    print(f"    {name:20s}: {vsn_w[idx]:.4f}")

        m = met.get('MAE', float('nan'))
        d = met.get('DirAcc', float('nan'))
        print(f"\n  => {exp_name}: MAE={m:.4f}  DirAcc={d:.3f}  "
              f"(Momentum baseline: {mom_dir:.3f})  Time={elapsed:.1f}s")
        results.append(met)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "e5_delta_lags.csv")
    keys = ["experiment", "lags", "MAE", "MAPE", "DirAcc",
            "MomentumDirAcc", "PinballLoss", "time_sec", "n_features"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*70}")
    print(f"E5 SUMMARY (H1, TFT d=64, h=4, delta lag features)")
    print(f"{'='*70}")
    print(f"{'Experiment':<18} {'Lags':<16} {'#Feat':>5} {'MAE':>8} {'DirAcc':>8} "
          f"{'MomBase':>8} {'Time':>8}")
    print("-" * 78)
    for r in results:
        m = r.get('MAE', float('nan'))
        d = r.get('DirAcc', float('nan'))
        mb = r.get('MomentumDirAcc', float('nan'))
        print(f"{r['experiment']:<18} {r['lags']:<16} {r['n_features']:>5} "
              f"{m:>8.4f} {d:>8.3f} {mb:>8.3f} {r.get('time_sec',0):>7.1f}s")
    print(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="E1-E5 experiments")
    parser.add_argument("--experiment", nargs="*", default=["E1", "E2"],
                        choices=["E1", "E2", "E3", "E3cls", "E4", "E5"],
                        help="Which experiments to run")
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2021-12-31")
    parser.add_argument("--test_years", type=int, default=1)
    parser.add_argument("--data_path", default=r"G:\data2")
    args = parser.parse_args()

    if "E1" in args.experiment:
        run_e1(args)
    if "E2" in args.experiment:
        run_e2(args)
    if "E3" in args.experiment:
        run_e3(args)
    if "E3cls" in args.experiment:
        run_e3cls(args)
    if "E4" in args.experiment:
        run_e4_ens(args)
    if "E5" in args.experiment:
        run_e5_lags(args)


if __name__ == "__main__":
    main()
