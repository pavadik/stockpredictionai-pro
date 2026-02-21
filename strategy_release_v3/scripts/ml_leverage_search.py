import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.konkop_analysis import analyse, format_report

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments")

def main():
    # Load ML test results
    results_path = os.path.join(OUT_DIR, "block_e_ml_test_results.csv")
    df_res = pd.read_csv(results_path, parse_dates=["entry_time"])
    
    # Load the original ensemble trades (these were generated with leverage=2.0)
    trades_path = os.path.join(OUT_DIR, "block_e_combined_all_trades.csv")
    df_trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    
    # Filter to OOS only
    train_end_dt = pd.Timestamp("2018-12-31")
    df_trades_oos = df_trades[df_trades["entry_time"] > train_end_dt].copy()
    
    # Sort both to ensure alignment
    df_res = df_res.sort_values("entry_time").reset_index(drop=True)
    df_trades_oos = df_trades_oos.sort_values("entry_time").reset_index(drop=True)
    
    # Add predictions
    df_trades_oos["ml_proba"] = df_res["ml_proba"]
    
    # Filter to only taken trades (Threshold 0.5)
    df_ml = df_trades_oos[df_trades_oos["ml_proba"] >= 0.5].copy()
    df_ml["contracts_base"] = df_ml["contracts"]
    
    print(f"Total ML-filtered trades: {len(df_ml)}")
    print("Base trades were generated with leverage=2.0.\n")
    
    # We will test multipliers on top of the base leverage 2.0.
    # mult=1.0 -> lev 2.0
    # mult=1.5 -> lev 3.0
    # mult=2.0 -> lev 4.0
    # mult=2.5 -> lev 5.0
    # mult=3.0 -> lev 6.0
    
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    results = []
    
    for mult in multipliers:
        df_test = df_ml.copy()
        
        # Apply multiplier
        df_test["contracts"] = np.clip((df_test["contracts_base"] * mult).astype(int), 1, None)
        df_test["pnl"] = df_test["pnl_1c"] * df_test["contracts"]
        
        # We need to simulate the total capital and commission.
        # Konkop analysis calculates net profit, max DD, etc.
        m = analyse(df_test, f"ML_Lev_x{2.0 * mult}", 5_000_000, 0.01)
        
        effective_leverage = 2.0 * mult
        
        results.append({
            "Leverage": f"x{effective_leverage:.1f}",
            "Net Profit": m["net_profit"],
            "Max DD %": m["max_dd_pct"],
            "Max DD (abs)": m["max_dd_abs"],
            "Profit Factor": m["profit_factor"],
            "Sharpe (1c)": m["sharpe_1c"],
            "Ann.Ret/MaxDD": m["ann_ret_max_dd"]
        })
        
        print(f"\n==============================================================")
        print(f"  ML ENSEMBLE STRATEGY (Leverage x{effective_leverage:.1f})")
        print(f"==============================================================")
        rep = format_report(m)
        print(rep.replace("\u2500", "-"))
        
    print("\n\n--- SUMMARY TABLE ---")
    df_summary = pd.DataFrame(results)
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()