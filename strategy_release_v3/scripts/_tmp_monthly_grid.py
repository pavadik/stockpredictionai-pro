import pandas as pd
import numpy as np
import os
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments")

def main():
    # Load ML test results
    results_path = os.path.join(OUT_DIR, "block_e_ml_test_results.csv")
    df_res = pd.read_csv(results_path, parse_dates=["entry_time"])
    
    # Load the original ensemble trades
    trades_path = os.path.join(OUT_DIR, "block_e_combined_all_trades.csv")
    df_trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    
    df_res = df_res.sort_values("entry_time").reset_index(drop=True)
    df_trades = df_trades.sort_values("entry_time").reset_index(drop=True)
    
    df_trades["ml_proba"] = df_res["ml_proba"]
    
    # Filter ML trades with proba >= 0.5
    df_ml = df_trades[df_trades["ml_proba"] >= 0.5].copy()
    
    # Apply Leverage x4.0 (multiplier = 2.0 on top of base 2.0)
    df_ml["contracts"] = np.clip((df_ml["contracts"] * 2.0).astype(int), 1, None)
    
    # Calc PnL net
    comm = 0.01 / 100.0 * (df_ml["entry_price"] + df_ml["exit_price"]) * df_ml["contracts"]
    df_ml["pnl_net"] = (df_ml["pnl_1c"] * df_ml["contracts"]) - comm
    
    df_ml = df_ml.sort_values("exit_time")
    
    # Generate Equity Curve trade-by-trade
    df_ml["equity"] = 5_000_000 + df_ml["pnl_net"].cumsum()
    df_ml["peak"] = df_ml["equity"].cummax()
    df_ml["dd_pct"] = (df_ml["peak"] - df_ml["equity"]) / df_ml["peak"] * 100.0
    
    # We want a breakdown by month
    df_ml["month_idx"] = df_ml["exit_time"].dt.to_period("M")
    
    # Create an empty template from 2006-01 to 2025-12
    all_months = pd.period_range("2006-01", "2025-12", freq="M")
    
    # Sum PnL for each month
    monthly_pnl = df_ml.groupby("month_idx")["pnl_net"].sum().reindex(all_months, fill_value=0)
    
    # Max DD for each month (approximation based on trade exit times)
    monthly_dd = df_ml.groupby("month_idx")["dd_pct"].max().reindex(all_months, fill_value=0)
    
    # Monthly Return % based on beginning of month equity
    # First, let's get End of Month Equity
    eom_equity = 5_000_000 + monthly_pnl.cumsum()
    
    # Beginning of month equity is shifted by 1
    bom_equity = eom_equity.shift(1).fillna(5_000_000)
    
    # Monthly Return
    monthly_ret = (monthly_pnl / bom_equity) * 100.0
    
    # Build dataframe for pivot
    df_monthly = pd.DataFrame({
        "month_idx": all_months,
        "Year": all_months.year,
        "Month": all_months.strftime("%b"),
        "PnL": monthly_pnl.values,
        "Return": monthly_ret.values,
        "MaxDD": monthly_dd.values
    })
    
    # Pivot for Return
    pivot_ret = df_monthly.pivot(index="Year", columns="Month", values="Return")
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_ret = pivot_ret.reindex(columns=months_order)
    
    # Annual Total Return (using product of 1 + r)
    pivot_ret["Year_Ret"] = (eom_equity[eom_equity.index.month == 12].values / bom_equity[bom_equity.index.month == 1].values - 1) * 100.0
    
    # Annual Max DD (from the highest DD recorded that year)
    yearly_dd = df_monthly.groupby("Year")["MaxDD"].max()
    pivot_ret["Max_DD"] = yearly_dd.values
    
    print("\n==========================================================================================================")
    print("  MONTHLY RETURNS (%) - ML Filtered (Threshold 0.5, Leverage x4.0) 2006-2025")
    print("==========================================================================================================")
    
    # Format and print
    header = f"{'Year':>5} |" + "".join([f"{m:>7} |" for m in months_order]) + f"{'Year%':>9} |{'MaxDD%':>9} |"
    print(header)
    print("-" * len(header))
    
    for year, row in pivot_ret.iterrows():
        line = f"{year:>5} |"
        for m in months_order:
            val = row[m]
            if pd.isna(val) or val == 0:
                line += f"{'0.00':>7} |"
            else:
                line += f"{val:>7.2f} |"
        line += f"{row['Year_Ret']:>8.2f}% |"
        line += f"{row['Max_DD']:>8.2f}% |"
        print(line)
        
    print("-" * len(header))

if __name__ == "__main__":
    main()