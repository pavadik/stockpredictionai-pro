import pandas as pd
import numpy as np
import os
import sys

# Ensure imports work from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.konkop_analysis import analyse, format_report

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments")

def evaluate_ml_impact():
    # Load ML test results
    results_path = os.path.join(OUT_DIR, "block_e_ml_test_results.csv")
    df_res = pd.read_csv(results_path, parse_dates=["entry_time"])
    
    # We need to map these predictions back to the original trades DataFrame
    # Let's load the original ensemble trades
    trades_path = os.path.join(OUT_DIR, "block_e_ensemble_all_trades.csv")
    df_trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    
    # Filter to OOS only
    train_end_dt = pd.Timestamp("2012-12-31")
    df_trades_oos = df_trades[df_trades["entry_time"] > train_end_dt].copy()
    
    # Sort both to ensure alignment
    df_res = df_res.sort_values("entry_time").reset_index(drop=True)
    df_trades_oos = df_trades_oos.sort_values("entry_time").reset_index(drop=True)
    
    # Add predictions
    df_trades_oos["ml_proba"] = df_res["ml_proba"]
    
    print("==============================================================")
    print("  BASE ENSEMBLE STRATEGY (OOS 2019-2026)")
    print("==============================================================")
    m_base = analyse(df_trades_oos.copy(), "Base", 5_000_000, 0.01)
    rep_base = format_report(m_base)
    print(rep_base.replace("\u2500", "-"))
    
    thresholds = [0.49, 0.50, 0.51]
    
    for thresh in thresholds:
        print(f"\n==============================================================")
        print(f"  ML ENSEMBLE STRATEGY (Threshold {thresh})")
        print(f"==============================================================")
        
        # Filter trades based on prediction
        df_trades_ml = df_trades_oos[df_trades_oos["ml_proba"] >= thresh].copy()
        
        trades_taken = len(df_trades_ml)
        trades_skipped = len(df_trades_oos) - trades_taken
        pct_taken = trades_taken / len(df_trades_oos)
        
        print(f"Trades Taken: {trades_taken} ({pct_taken:.1%}) | Skipped: {trades_skipped}")
        
        if trades_taken > 0:
            m_ml = analyse(df_trades_ml, "ML_Filtered", 5_000_000, 0.01)
            rep_ml = format_report(m_ml)
            print(rep_ml.replace("\u2500", "-"))
        else:
            print("No trades taken at this threshold.")

if __name__ == "__main__":
    evaluate_ml_impact()