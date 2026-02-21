"""Run experiments across multiple tickers, loading each panel separately.

Usage:
    python scripts/multi_ticker_test.py --tickers GMKN LKOH GAZP RTSF
    python scripts/multi_ticker_test.py --tickers SBER --timeframes D1 H1 M5 M1
    python scripts/multi_ticker_test.py --tickers SBER --periods "2006-01-01,2012-12-31" "2012-01-01,2018-12-31" "2015-01-01,2021-12-31"
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


def run_single(ticker, start, end, test_years, generator, timeframe,
               data_source, data_path, exp_name,
               lr_g=None, lr_d=None, q_weight=None, seq_len=None):
    """Load data and run one full experiment for a ticker/timeframe/period."""
    set_global_seed(SEED, deterministic=False)

    cfg = Config(
        ticker=ticker, start=start, end=end,
        test_years=test_years, generator=generator,
        data_source=data_source, timeframe=timeframe,
        data_path=data_path,
    )
    if lr_g is not None:
        cfg.lr_g = lr_g
    if lr_d is not None:
        cfg.lr_d = lr_d
    if q_weight is not None:
        cfg.q_weight = q_weight
    if seq_len is not None:
        cfg.seq_len = seq_len
    cfg.apply_timeframe_defaults()

    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"  ticker={ticker}, tf={timeframe}, period={start}..{end}, gen={generator}")
    print(f"{'='*60}")

    print(f"  Loading panel...")
    panel = build_panel_auto(cfg)
    print(f"  Panel: {panel.shape}, loaded in {time.time()-t0:.1f}s")

    train_panel, test_panel = train_test_split_by_years(panel, cfg.test_years)
    print(f"  Train: {train_panel.shape}, Test: {test_panel.shape}")

    feat_train, feat_test = build_features_safe(train_panel, test_panel, cfg)
    feat_train, feat_test = fit_transforms(feat_train, feat_test, cfg)

    met, _, _, _, _, _ = run_one_split(feat_train, feat_test, cfg)

    elapsed = time.time() - t0
    met["time_sec"] = round(elapsed, 1)
    met["n_features"] = feat_train.shape[1]
    met["n_bars"] = panel.shape[0]
    met["train_samples"] = len(feat_train)
    met["test_samples"] = len(feat_test)

    print(f"  -> MAE={met['MAE']:.4f}  MAPE={met['MAPE']:.4f}  "
          f"DirAcc={met['DirAcc']:.3f}  Pinball={met['PinballLoss']:.4f}  "
          f"Time={elapsed:.1f}s")

    return met


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["SBER"])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2021-12-31")
    parser.add_argument("--test_years", type=int, default=1)
    parser.add_argument("--generator", default="tst")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--timeframes", nargs="+", default=None,
                        help="Sweep timeframes (overrides --timeframe)")
    parser.add_argument("--periods", nargs="+", default=None,
                        help="Sweep periods as 'start,end' pairs")
    parser.add_argument("--data_source", default="local")
    parser.add_argument("--data_path", default=r"G:\data2")
    parser.add_argument("--lr_g", type=float, default=None)
    parser.add_argument("--lr_d", type=float, default=None)
    parser.add_argument("--q_weight", type=float, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--output", default="outputs/experiments/multi_ticker.csv")
    args = parser.parse_args()

    _log_device_info()

    results = []

    if args.timeframes:
        # Timeframe sweep mode
        for tf in args.timeframes:
            start = args.start
            end = args.end
            test_years = args.test_years
            # For M1 use shorter period to avoid OOM
            if tf == "M1":
                start = "2019-01-01"
                test_years = 1
            exp_name = f"{args.tickers[0]}_{tf}"
            try:
                met = run_single(
                    args.tickers[0], start, end, test_years,
                    args.generator, tf, args.data_source, args.data_path,
                    exp_name, lr_g=args.lr_g, lr_d=args.lr_d,
                    q_weight=args.q_weight, seq_len=args.seq_len,
                )
            except Exception as e:
                print(f"  !! FAILED: {e}")
                traceback.print_exc()
                met = {"MAE": float("nan"), "MAPE": float("nan"),
                       "DirAcc": float("nan"), "PinballLoss": float("nan"),
                       "time_sec": 0, "n_features": 0, "n_bars": 0,
                       "train_samples": 0, "test_samples": 0}
            met["experiment"] = exp_name
            met["ticker"] = args.tickers[0]
            met["timeframe"] = tf
            met["period"] = f"{start}..{end}"
            met["generator"] = args.generator
            results.append(met)

    elif args.periods:
        # Period sweep mode
        for period_str in args.periods:
            parts = period_str.split(",")
            start, end = parts[0], parts[1]
            exp_name = f"{args.tickers[0]}_{start[:4]}_{end[:4]}"
            try:
                met = run_single(
                    args.tickers[0], start, end, args.test_years,
                    args.generator, args.timeframe, args.data_source,
                    args.data_path, exp_name, lr_g=args.lr_g, lr_d=args.lr_d,
                    q_weight=args.q_weight, seq_len=args.seq_len,
                )
            except Exception as e:
                print(f"  !! FAILED: {e}")
                traceback.print_exc()
                met = {"MAE": float("nan"), "MAPE": float("nan"),
                       "DirAcc": float("nan"), "PinballLoss": float("nan"),
                       "time_sec": 0, "n_features": 0, "n_bars": 0,
                       "train_samples": 0, "test_samples": 0}
            met["experiment"] = exp_name
            met["ticker"] = args.tickers[0]
            met["timeframe"] = args.timeframe
            met["period"] = f"{start}..{end}"
            met["generator"] = args.generator
            results.append(met)

    else:
        # Multi-ticker mode
        for ticker in args.tickers:
            exp_name = f"{ticker}_{args.timeframe}"
            try:
                met = run_single(
                    ticker, args.start, args.end, args.test_years,
                    args.generator, args.timeframe, args.data_source,
                    args.data_path, exp_name, lr_g=args.lr_g, lr_d=args.lr_d,
                    q_weight=args.q_weight, seq_len=args.seq_len,
                )
            except Exception as e:
                print(f"  !! FAILED: {e}")
                traceback.print_exc()
                met = {"MAE": float("nan"), "MAPE": float("nan"),
                       "DirAcc": float("nan"), "PinballLoss": float("nan"),
                       "time_sec": 0, "n_features": 0, "n_bars": 0,
                       "train_samples": 0, "test_samples": 0}
            met["experiment"] = exp_name
            met["ticker"] = ticker
            met["timeframe"] = args.timeframe
            met["period"] = f"{args.start}..{args.end}"
            met["generator"] = args.generator
            results.append(met)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    keys = ["experiment", "ticker", "timeframe", "period", "generator",
            "MAE", "MAPE", "DirAcc", "PinballLoss",
            "time_sec", "n_features", "n_bars", "train_samples", "test_samples"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*60}")
    print(f"Complete! Results saved to {args.output}")
    print(f"{'='*60}")
    print(f"\n{'experiment':<25} {'MAE':>8} {'DirAcc':>8} {'Bars':>8} {'Time':>8}")
    print("-" * 62)
    for r in results:
        mae = r['MAE']
        da = r['DirAcc']
        mae_s = f"{mae:.4f}" if not (mae != mae) else "NaN"
        da_s = f"{da:.3f}" if not (da != da) else "NaN"
        print(f"{r['experiment']:<25} {mae_s:>8} {da_s:>8} {r['n_bars']:>8} {r['time_sec']:>7.1f}s")


if __name__ == "__main__":
    main()
