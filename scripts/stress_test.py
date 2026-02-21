"""Stage 11: Stress tests for robustness of the GAN prediction system.

Tests:
  1. Repeatability: 3 runs with same params, check MAE spread < 15%
  2. Noise injection: add 1% Gaussian noise to prices, MAE degradation < 20%
  3. Short data: train on 3 years instead of 6, model still works
  4. Small batch: batch_size=32 instead of 256, training stable
  5. Fewer correlates: only 2 correlated tickers instead of 10

Usage:
    python scripts/stress_test.py --ticker SBER --data_source local --data_path G:\\data2 --timeframe M5
"""
import argparse
import csv
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import build_panel_auto
from src.dataset import train_test_split_by_years
from src.train import (
    build_features_safe, fit_transforms, run_one_split,
    set_global_seed,
)

SEED = 42


def run_single(panel, cfg, exp_name):
    """Run one experiment, return metrics dict or None on failure."""
    t0 = time.time()
    try:
        train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)
        tr_feat, te_feat = build_features_safe(train_panel, test_panel, cfg)
        tr_all, te_all = fit_transforms(tr_feat, te_feat, cfg)
        met, _, _, _, _, _ = run_one_split(tr_all, te_all, cfg)
        elapsed = time.time() - t0
        met["experiment"] = exp_name
        met["time_s"] = round(elapsed, 1)
        print(f"  {exp_name}: MAE={met['MAE']:.6f} DirAcc={met['DirAcc']:.4f} ({elapsed:.0f}s)")
        return met
    except Exception as e:
        print(f"  {exp_name}: FAILED -- {e}")
        traceback.print_exc()
        return {"experiment": exp_name, "MAE": float("nan"), "DirAcc": float("nan"),
                "MAPE": float("nan"), "PinballLoss": float("nan"),
                "time_s": round(time.time() - t0, 1), "error": str(e)}


def make_cfg(args, **overrides):
    """Create Config with base args + overrides."""
    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_years=args.test_years, data_source=args.data_source,
        timeframe=args.timeframe, data_path=args.data_path,
        local_raw_source=args.raw_source, generator=args.generator,
        seq_len=12, lr_g=1e-3, lr_d=5e-5,
        adv_weight=1.0, l1_weight=0.4, cls_weight=0.2, q_weight=1.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    if cfg.data_source == "local" or cfg.timeframe != "D1":
        cfg.apply_timeframe_defaults()
    return cfg


def test_repeatability(panel, args):
    """Test 1: 3 identical runs, check MAE spread < 15%."""
    print("\n" + "=" * 60)
    print("TEST 1: Repeatability (3 identical runs)")
    print("=" * 60)
    maes = []
    for i in range(3):
        cfg = make_cfg(args)
        set_global_seed(SEED + i)
        met = run_single(panel, cfg, f"repeat_{i+1}")
        if met and not np.isnan(met.get("MAE", float("nan"))):
            maes.append(met["MAE"])
    if len(maes) >= 2:
        spread = (max(maes) - min(maes)) / np.mean(maes) * 100
        passed = spread < 15
        print(f"  MAE values: {[f'{m:.6f}' for m in maes]}")
        print(f"  Spread: {spread:.1f}% (threshold: 15%) -- {'PASS' if passed else 'FAIL'}")
        return {"test": "Repeatability", "spread_pct": round(spread, 1),
                "pass": passed, "maes": maes}
    return {"test": "Repeatability", "spread_pct": None, "pass": False, "maes": maes}


def test_noise_injection(panel, args):
    """Test 2: Add 1% Gaussian noise to Close prices, check MAE degradation < 20%."""
    print("\n" + "=" * 60)
    print("TEST 2: Noise injection (1% Gaussian noise)")
    print("=" * 60)
    set_global_seed(SEED)
    cfg_clean = make_cfg(args)
    met_clean = run_single(panel, cfg_clean, "clean")

    noisy_panel = panel.copy()
    np.random.seed(SEED)
    for col in noisy_panel.columns:
        if "Close" in col or "Open" in col or "High" in col or "Low" in col:
            noise = np.random.randn(len(noisy_panel)) * 0.01 * noisy_panel[col].std()
            noisy_panel[col] = noisy_panel[col] + noise

    set_global_seed(SEED)
    cfg_noisy = make_cfg(args)
    met_noisy = run_single(noisy_panel, cfg_noisy, "noisy_1pct")

    if met_clean and met_noisy and not np.isnan(met_clean["MAE"]) and not np.isnan(met_noisy["MAE"]):
        degradation = (met_noisy["MAE"] - met_clean["MAE"]) / met_clean["MAE"] * 100
        passed = degradation < 20
        print(f"  Clean MAE: {met_clean['MAE']:.6f}, Noisy MAE: {met_noisy['MAE']:.6f}")
        print(f"  Degradation: {degradation:+.1f}% (threshold: 20%) -- {'PASS' if passed else 'FAIL'}")
        return {"test": "Noise injection", "degradation_pct": round(degradation, 1),
                "pass": passed, "clean_mae": met_clean["MAE"], "noisy_mae": met_noisy["MAE"]}
    return {"test": "Noise injection", "degradation_pct": None, "pass": False}


def test_short_data(args):
    """Test 3: Train on only 3 years (2019-2021) instead of 6."""
    print("\n" + "=" * 60)
    print("TEST 3: Short data (3 years: 2019-2021)")
    print("=" * 60)
    set_global_seed(SEED)
    cfg = make_cfg(args, start="2019-01-01", end="2021-12-31", test_years=1)
    panel_short = build_panel_auto(cfg)
    print(f"  Short panel shape: {panel_short.shape}")
    met = run_single(panel_short, cfg, "short_3y")
    passed = met is not None and not np.isnan(met.get("MAE", float("nan")))
    if passed:
        print(f"  PASS -- model trained on 3y data, MAE={met['MAE']:.6f}")
    else:
        print("  FAIL -- model could not train")
    return {"test": "Short data (3y)", "pass": passed,
            "mae": met.get("MAE") if met else None}


def test_small_batch(panel, args):
    """Test 4: batch_size=32, check stability."""
    print("\n" + "=" * 60)
    print("TEST 4: Small batch_size (32)")
    print("=" * 60)
    set_global_seed(SEED)
    cfg_base = make_cfg(args)
    met_base = run_single(panel, cfg_base, "batch_256")

    set_global_seed(SEED)
    cfg_small = make_cfg(args, batch_size=32)
    met_small = run_single(panel, cfg_small, "batch_32")

    if met_base and met_small and not np.isnan(met_base["MAE"]) and not np.isnan(met_small["MAE"]):
        delta = abs(met_small["MAE"] - met_base["MAE"]) / met_base["MAE"] * 100
        passed = delta < 30
        print(f"  Base MAE: {met_base['MAE']:.6f}, Small batch MAE: {met_small['MAE']:.6f}")
        print(f"  Delta: {delta:.1f}% (threshold: 30%) -- {'PASS' if passed else 'FAIL'}")
        return {"test": "Small batch", "delta_pct": round(delta, 1),
                "pass": passed, "base_mae": met_base["MAE"], "small_mae": met_small["MAE"]}
    return {"test": "Small batch", "pass": False}


def test_few_correlates(args):
    """Test 5: Only 2 correlated tickers instead of 10."""
    print("\n" + "=" * 60)
    print("TEST 5: Few correlates (only GMKN, LKOH)")
    print("=" * 60)
    set_global_seed(SEED)
    cfg = make_cfg(args, local_correlated=("GMKN", "LKOH"))
    panel_few = build_panel_auto(cfg)
    print(f"  Panel shape (2 correlates): {panel_few.shape}")
    met = run_single(panel_few, cfg, "few_corr_2")
    passed = met is not None and not np.isnan(met.get("MAE", float("nan")))
    if passed:
        print(f"  PASS -- model trained with 2 correlates, MAE={met['MAE']:.6f}")
    else:
        print("  FAIL -- model could not train with 2 correlates")
    return {"test": "Few correlates (2)", "pass": passed,
            "mae": met.get("MAE") if met else None}


def main():
    parser = argparse.ArgumentParser(description="Stage 11: Stress tests")
    parser.add_argument('--ticker', default='SBER')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2021-12-31')
    parser.add_argument('--test_years', type=int, default=1)
    parser.add_argument('--data_source', default='local', choices=['yfinance', 'local'])
    parser.add_argument('--timeframe', default='M5')
    parser.add_argument('--data_path', default='G:\\data2')
    parser.add_argument('--raw_source', default='m1', choices=['m1', 'ticks'])
    parser.add_argument('--generator', default='tst', choices=['lstm', 'tst'])
    args = parser.parse_args()

    print(f"Loading full panel: {args.ticker} {args.timeframe} [{args.start} .. {args.end}]")
    cfg_full = make_cfg(args)
    panel = build_panel_auto(cfg_full)
    print(f"Panel shape: {panel.shape}")

    results = []

    r1 = test_repeatability(panel, args)
    results.append(r1)

    r2 = test_noise_injection(panel, args)
    results.append(r2)

    r3 = test_short_data(args)
    results.append(r3)

    r4 = test_small_batch(panel, args)
    results.append(r4)

    r5 = test_few_correlates(args)
    results.append(r5)

    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r.get("pass"))
    total = len(results)
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        print(f"  [{status}] {r['test']}")
    print(f"\nTotal: {passed}/{total} passed")

    os.makedirs("outputs/experiments", exist_ok=True)
    out_path = "outputs/experiments/stage11_stress.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "pass", "detail"])
        w.writeheader()
        for r in results:
            detail = {k: v for k, v in r.items() if k not in ("test", "pass")}
            w.writerow({"test": r["test"], "pass": r.get("pass", False),
                        "detail": str(detail)})
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
