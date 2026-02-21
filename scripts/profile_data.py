"""Data profiling script for local MOEX data in G:\\data2.

Scans the date-based hierarchy, catalogs tickers, date ranges, row counts,
detects gaps, anomalies, and outputs a summary CSV.

Usage:
    python scripts/profile_data.py --data_path G:\\data2
    python scripts/profile_data.py --data_path G:\\data2 --ticker SBRF
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

M1_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]
TICK_COLUMNS = ["date", "time", "price", "volume"]


def scan_tickers(data_path: str) -> dict:
    """Walk the data hierarchy and catalog all tickers with metadata.

    Returns dict: ticker -> {"dates": [...], "m1_files": int, "tick_files": int}
    """
    tickers = defaultdict(lambda: {"dates": [], "m1_files": 0, "tick_files": 0})

    for year_dir in sorted(os.listdir(data_path)):
        year_path = os.path.join(data_path, year_dir)
        if not os.path.isdir(year_path) or not year_dir.isdigit():
            continue
        for month_dir in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month_dir)
            if not os.path.isdir(month_path):
                continue
            for day_dir in sorted(os.listdir(month_path)):
                day_path = os.path.join(month_path, day_dir)
                if not os.path.isdir(day_path):
                    continue
                try:
                    dt = datetime(int(year_dir), int(month_dir), int(day_dir))
                except ValueError:
                    continue
                for ticker_dir in os.listdir(day_path):
                    ticker_path = os.path.join(day_path, ticker_dir)
                    if not os.path.isdir(ticker_path):
                        continue
                    tickers[ticker_dir]["dates"].append(dt)
                    m1_dir = os.path.join(ticker_path, "M1")
                    tick_dir = os.path.join(ticker_path, "ticks")
                    if os.path.isdir(m1_dir):
                        for fn in ("data.txt", "data.csv"):
                            if os.path.isfile(os.path.join(m1_dir, fn)):
                                tickers[ticker_dir]["m1_files"] += 1
                                break
                    if os.path.isdir(tick_dir):
                        for fn in ("data.txt", "data.csv"):
                            if os.path.isfile(os.path.join(tick_dir, fn)):
                                tickers[ticker_dir]["tick_files"] += 1
                                break

    return dict(tickers)


def read_m1_file(filepath: str) -> pd.DataFrame:
    """Read a single M1 CSV file (no header)."""
    try:
        df = pd.read_csv(
            filepath, header=None, names=M1_COLUMNS,
            dtype={"date": str, "time": str, "open": np.float32,
                   "high": np.float32, "low": np.float32,
                   "close": np.float32, "volume": np.int64},
        )
        return df
    except Exception:
        return pd.DataFrame()


def profile_ticker_m1(data_path: str, ticker: str,
                      dates: list, max_sample_days: int = 30) -> dict:
    """Profile M1 data for a single ticker: row counts, anomalies, gaps."""
    total_rows = 0
    bars_per_day = []
    anomalies = []
    sample_dates = sorted(dates)

    if len(sample_dates) > max_sample_days:
        step = len(sample_dates) // max_sample_days
        sample_dates = sample_dates[::step][:max_sample_days]

    for dt in sample_dates:
        m1_dir = os.path.join(
            data_path, str(dt.year), str(dt.month), str(dt.day),
            ticker, "M1",
        )
        m1_path = None
        for fn in ("data.txt", "data.csv"):
            candidate = os.path.join(m1_dir, fn)
            if os.path.isfile(candidate):
                m1_path = candidate
                break
        if m1_path is None:
            continue
        df = read_m1_file(m1_path)
        if df.empty:
            continue

        n = len(df)
        total_rows += n
        bars_per_day.append(n)

        if (df["close"] <= 0).any():
            anomalies.append(f"{dt.date()}: zero/negative close prices")
        if n > 0 and df["high"].max() > 0:
            pct_change = df["close"].pct_change().dropna().abs()
            extreme = pct_change[pct_change > 0.10]
            if len(extreme) > 0:
                anomalies.append(
                    f"{dt.date()}: {len(extreme)} bars with >10% change"
                )

    all_dates = sorted(dates)
    missing_days = []
    if len(all_dates) > 1:
        date_set = set(d.date() for d in all_dates)
        current = all_dates[0].date()
        end_date = all_dates[-1].date()
        while current <= end_date:
            if current.weekday() < 5 and current not in date_set:
                missing_days.append(current)
            current += timedelta(days=1)

    return {
        "ticker": ticker,
        "first_date": min(dates).strftime("%Y-%m-%d"),
        "last_date": max(dates).strftime("%Y-%m-%d"),
        "total_trading_days": len(dates),
        "m1_sampled_rows": total_rows,
        "avg_bars_per_day": round(np.mean(bars_per_day), 1) if bars_per_day else 0,
        "min_bars_per_day": min(bars_per_day) if bars_per_day else 0,
        "max_bars_per_day": max(bars_per_day) if bars_per_day else 0,
        "missing_weekdays": len(missing_days),
        "anomaly_count": len(anomalies),
        "anomalies_sample": "; ".join(anomalies[:5]),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile local MOEX data")
    parser.add_argument("--data_path", default=r"G:\data2",
                        help="Path to data hierarchy")
    parser.add_argument("--ticker", default=None,
                        help="Profile a specific ticker only")
    parser.add_argument("--output", default="outputs/data_profile.csv",
                        help="Output CSV path")
    parser.add_argument("--max_sample", type=int, default=30,
                        help="Max days to sample per ticker for detailed profiling")
    args = parser.parse_args()

    print(f"Scanning {args.data_path} ...")
    catalog = scan_tickers(args.data_path)
    print(f"Found {len(catalog)} tickers\n")

    if not catalog:
        print("ERROR: No tickers found. Check --data_path.")
        sys.exit(1)

    if args.ticker:
        if args.ticker not in catalog:
            print(f"ERROR: Ticker '{args.ticker}' not found. "
                  f"Available: {sorted(catalog.keys())}")
            sys.exit(1)
        catalog = {args.ticker: catalog[args.ticker]}

    print(f"{'Ticker':<15} {'First':>12} {'Last':>12} {'Days':>6} "
          f"{'M1 files':>9} {'Tick files':>11}")
    print("-" * 70)
    for tk in sorted(catalog.keys()):
        info = catalog[tk]
        dates = info["dates"]
        first = min(dates).strftime("%Y-%m-%d") if dates else "N/A"
        last = max(dates).strftime("%Y-%m-%d") if dates else "N/A"
        print(f"{tk:<15} {first:>12} {last:>12} {len(dates):>6} "
              f"{info['m1_files']:>9} {info['tick_files']:>11}")

    print(f"\nProfiling M1 data (sampling up to {args.max_sample} days per ticker)...")
    profiles = []
    for tk in sorted(catalog.keys()):
        info = catalog[tk]
        if info["m1_files"] == 0:
            print(f"  {tk}: no M1 files, skipping")
            continue
        print(f"  {tk}...", end=" ", flush=True)
        prof = profile_ticker_m1(args.data_path, tk, info["dates"],
                                 max_sample_days=args.max_sample)
        profiles.append(prof)
        print(f"avg {prof['avg_bars_per_day']} bars/day, "
              f"{prof['anomaly_count']} anomalies")

    if profiles:
        df_prof = pd.DataFrame(profiles)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df_prof.to_csv(args.output, index=False)
        print(f"\nProfile saved: {args.output}")

        print("\n" + "=" * 70)
        print("SUMMARY: Tickers suitable for experiments (>100 trading days, M1 data)")
        print("=" * 70)
        suitable = df_prof[df_prof["total_trading_days"] >= 100]
        if suitable.empty:
            print("  No tickers with >= 100 trading days found.")
        else:
            for _, row in suitable.iterrows():
                status = "OK" if row["anomaly_count"] < 5 else "WARN (anomalies)"
                print(f"  {row['ticker']:<15} {row['first_date']} -> "
                      f"{row['last_date']}  ({row['total_trading_days']} days)  "
                      f"[{status}]")
    else:
        print("\nNo M1 data found for profiling.")


if __name__ == "__main__":
    main()
