import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.konkop_analysis import analyse, format_report

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments")

def main():
    # Load ML test results (now contains ALL data 2006-2026)
    results_path = os.path.join(OUT_DIR, "block_e_ml_test_results.csv")
    df_res = pd.read_csv(results_path, parse_dates=["entry_time"])
    
    # Load the original ensemble trades
    trades_path = os.path.join(OUT_DIR, "block_e_combined_all_trades.csv")
    df_trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    
    # Sort both to ensure alignment
    df_res = df_res.sort_values("entry_time").reset_index(drop=True)
    df_trades = df_trades.sort_values("entry_time").reset_index(drop=True)
    
    # Add predictions
    df_trades["ml_proba"] = df_res["ml_proba"]
    
    print("==============================================================")
    print("  BASE STRATEGY (ALL DATA 2006-2026)")
    print("==============================================================")
    m_base = analyse(df_trades.copy(), "Base", 5_000_000, 0.01)
    rep_base = format_report(m_base)
    print(rep_base.replace("\u2500", "-"))
    
    print(f"\n==============================================================")
    print(f"  ML STRATEGY (Threshold 0.5, Leverage x4.0) ALL DATA")
    print(f"==============================================================")
    
    df_ml = df_trades[df_trades["ml_proba"] >= 0.5].copy()
    
    # Apply Leverage x4.0 (Base is x2.0, so multiplier is 2.0)
    df_ml["contracts"] = np.clip((df_ml["contracts"] * 2.0).astype(int), 1, None)
    df_ml["pnl"] = df_ml["pnl_1c"] * df_ml["contracts"]
    
    trades_taken = len(df_ml)
    trades_skipped = len(df_trades) - trades_taken
    pct_taken = trades_taken / len(df_trades)
    
    print(f"Trades Taken: {trades_taken} ({pct_taken:.1%}) | Skipped: {trades_skipped}")
    
    if trades_taken > 0:
        m_ml = analyse(df_ml, "ML_Filtered_Lev_x4", 5_000_000, 0.01)
        rep_ml = format_report(m_ml)
        print(rep_ml.replace("\u2500", "-"))
    else:
        print("No trades taken at this threshold.")

if __name__ == "__main__":
    main()