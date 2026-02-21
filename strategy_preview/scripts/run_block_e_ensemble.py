import pandas as pd
import numpy as np
import time
import os
import sys

# Ensure imports work from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_local import load_m1_bars
from src.strategy.momentum_trend import StrategyParams, prepare_strategy_data, generate_signals
from src.strategy.backtester import simulate_combined_trades, compute_metrics, trades_to_dataframe
from scripts.konkop_analysis import analyse, format_report

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments")
PARAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params")

def get_top_diverse_params(csv_name, direction, n=5):
    df = pd.read_csv(os.path.join(PARAMS_DIR, csv_name))
    df = df.sort_values("avg_trade_1c", ascending=False)
    
    top_sets = []
    seen_tfs = set()
    
    for _, row in df.iterrows():
        tf_pair = (row["tf1"], row["tf2"])
        if tf_pair not in seen_tfs:
            seen_tfs.add(tf_pair)
            top_sets.append(row)
            if len(top_sets) == n:
                break
                
    params_list = []
    for row in top_sets:
        p = {
            "tf1_minutes": int(row["tf1"]),
            "tf2_minutes": int(row["tf2"]),
            "lookback": int(row["lookback"]),
            "length": int(row["length"]),
            "lookback2": int(row["lookback2"]),
            "length2": int(row["length2"]),
            "min_s": int(row["min_s"]),
            "max_s": int(row["max_s"]),
            "koeff1": float(row["koeff1"]),
            "koeff2": float(row["koeff2"]),
            "mmcoff": int(row["mmcoff"]),
            "exitday": int(row["exitday"]),
            "sdel_day": int(row["sdel_day"]),
            "capital": 1_000_000,
            "point_value_mult": 300.0,
            "direction": direction,
            "adx_threshold": 0.0,
            "adx_action": "skip"
        }
        params_list.append(StrategyParams(**p))
    return params_list

def _compute_1c_pnl(trades_df: pd.DataFrame) -> np.ndarray:
    signs = np.where(trades_df["direction"] == "short", -1.0, 1.0)
    return signs * (trades_df["exit_price"].values - trades_df["entry_price"].values)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="RTSF")
    ap.add_argument("--data_path", default=r"G:\data2")
    ap.add_argument("--train_start", default="2007-01-01")
    ap.add_argument("--train_end", default="2018-12-31")
    ap.add_argument("--test_start", default="2019-01-01")
    ap.add_argument("--test_end", default="2026-02-28")
    ap.add_argument("--commission", type=float, default=0.01)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading Top-5 diverse parameter sets for LONG and SHORT...")
    long_params = get_top_diverse_params(f"block_e_gpu_trials.csv", "long", n=5)
    short_params = get_top_diverse_params(f"block_e_short_gpu_trials.csv", "short", n=5)
    
    print("\n--- LONG PARAMS ---")
    for p in long_params:
        print(f"TF1={p.tf1_minutes} TF2={p.tf2_minutes} | lookback={p.lookback} koeff1={p.koeff1:.4f}")
    
    print("\n--- SHORT PARAMS ---")
    for p in short_params:
        print(f"TF1={p.tf1_minutes} TF2={p.tf2_minutes} | lookback={p.lookback} koeff1={p.koeff1:.4f}")

    print(f"\nLoading M1 bars for {args.ticker}...")
    t0 = time.time()
    df_m1_full = load_m1_bars(args.data_path, args.ticker, args.train_start, args.test_end)
    print(f"Loaded {len(df_m1_full):,} M1 bars in {time.time()-t0:.1f}s")
    
    train_end_dt = pd.Timestamp(args.train_end)
    test_start_dt = pd.Timestamp(args.test_start)

    all_results = []

    for period, df_m1 in [
        ("train", df_m1_full[df_m1_full.index <= train_end_dt]),
        ("test", df_m1_full[df_m1_full.index >= test_start_dt]),
    ]:
        print(f"\n--- {period.upper()} PERIOD ---")
        t0 = time.time()
        
        period_trades = []
        for i in range(5):
            print(f"  Simulating sub-strategy {i+1}/5...")
            pL = long_params[i]
            pS = short_params[i]
            
            df_sig_L = prepare_strategy_data(df_m1.copy(), pL)
            df_sig_L = generate_signals(df_sig_L, pL)
            
            df_sig_S = prepare_strategy_data(df_m1.copy(), pS)
            df_sig_S = generate_signals(df_sig_S, pS)
            
            trades = simulate_combined_trades(
                df_sig_L, df_sig_S, pL, pS,
                max_contracts=0,
                max_contracts_long=10,
                max_contracts_short=4,
                flip_mode="close_loss",
                leverage=2.0
            )
            period_trades.extend(trades)
            
        print(f"  Sorting and analyzing ensemble trades...")
        period_trades.sort(key=lambda x: x.exit_time)
        
        tdf = trades_to_dataframe(period_trades)
        tdf["pnl_1c"] = _compute_1c_pnl(tdf)
        tdf["period"] = period
        all_results.append(tdf)
        
        m = analyse(tdf, f"ENSEMBLE {period.upper()}", 5_000_000, args.commission)
        rep = format_report(m)
        safe = rep.replace("\u2500", "-")
        try:
            print(safe)
        except UnicodeEncodeError:
            print(safe.encode("ascii", errors="replace").decode("ascii"))
        
        print(f"  Elapsed: {time.time()-t0:.1f}s")

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.sort_values("exit_time", inplace=True)
    out_csv = os.path.join(OUT_DIR, "block_e_ensemble_all_trades.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df_all)} ensemble trades to {out_csv}")

if __name__ == "__main__":
    main()