"""
Detailed analysis of ML Strategy (Long-only Combined) using Konkop metrics.
Adapts the ML strategy trades CSV to the format expected by konkop_analysis.py.
Includes compounding simulation.
"""
import sys
import os
import pandas as pd
import numpy as np
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the analysis logic from the release script
release_scripts_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "strategy_release", "scripts"
)
sys.path.insert(0, release_scripts_path)

from konkop_analysis import analyse, format_report

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "strategy"
)

def apply_compounding(sub_df, start_cap, comm_pct, leverage=1.0):
    """
    Simulate compounding by recalculating contracts and PnL based on running equity.
    Returns a new DataFrame with updated 'contracts' and 'pnl' columns.
    """
    local_df = sub_df.copy().reset_index(drop=True)
    equity = start_cap
    contracts_col = []
    pnl_col = []
    
    # We need gross_pnl_1c from the dataframe
    
    for i in range(len(local_df)):
        entry_p = local_df.at[i, 'entry_price']
        exit_p = local_df.at[i, 'exit_price']
        gross_1c = local_df.at[i, 'gross_pnl_1c']
        
        # Position sizing: All-in (Leverage x)
        if equity <= 0:
            contracts = 0
        else:
            contracts = int((equity * leverage) / entry_p)
            if contracts < 1: contracts = 0 # Can't afford 1 share
        
        # Calculate net result to update equity
        gross_pnl = contracts * gross_1c
        comm_val = (comm_pct / 100.0) * (entry_p + exit_p) * contracts
        net_pnl = gross_pnl - comm_val
        
        equity += net_pnl
        
        contracts_col.append(contracts)
        pnl_col.append(gross_pnl) # Konkop expects gross PnL in 'pnl' column
        
    local_df['contracts'] = contracts_col
    local_df['pnl'] = pnl_col
    # Ensure pnl_1c is present for analyse function (it expects gross per unit)
    local_df['pnl_1c'] = local_df['gross_pnl_1c']
    return local_df

def main():
    # 1. Load ML strategy trades
    trades_path = os.path.join(OUT_DIR, "strategy_long_only_combined_trades.csv")
    if not os.path.exists(trades_path):
        print(f"Error: {trades_path} not found.")
        return

    df = pd.read_csv(trades_path)
    print(f"Loaded {len(df)} trades from {trades_path}")

    # Simulation parameters
    INITIAL_CAPITAL = 1_000_000
    COMMISSION_PCT = 0.03 # 0.03% requested
    LEVERAGE = 1.0 # 100% of equity used per trade

    # Reconstruct Gross PnL per unit
    # pnl_pts in CSV is NET of ORIGINAL_COMMISSION (0.02). 
    ORIGINAL_COMMISSION_PCT = 0.02
    df["orig_comm_1c"] = (ORIGINAL_COMMISSION_PCT / 100.0) * (df["entry_price"] + df["exit_price"])
    df["gross_pnl_1c"] = df["pnl_pts"] + df["orig_comm_1c"]
    
    # Period split
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["year"] = df["entry_time"].dt.year
    
    # Define Train/Test split
    df_train_raw = df[df["year"] <= 2018].copy()
    df_test_raw = df[df["year"] > 2018].copy()
    
    # Apply compounding independently for isolated analysis
    # This allows viewing "Test" performance as if we started with 1M fresh.
    df_train_comp = apply_compounding(df_train_raw, INITIAL_CAPITAL, COMMISSION_PCT, LEVERAGE)
    df_test_comp = apply_compounding(df_test_raw, INITIAL_CAPITAL, COMMISSION_PCT, LEVERAGE)
    
    # For "ALL", we simulate continuous compounding
    df_all_comp = apply_compounding(df, INITIAL_CAPITAL, COMMISSION_PCT, LEVERAGE)
    
    # 3. Analyze
    m_train = analyse(df_train_comp, "TRAIN (2009 - 2018) [Compounding]", INITIAL_CAPITAL, COMMISSION_PCT)
    m_test = analyse(df_test_comp, "TEST / OOS (2019 - 2021) [Compounding]", INITIAL_CAPITAL, COMMISSION_PCT)
    m_all = analyse(df_all_comp, "ALL (2009 - 2021) [Compounding]", INITIAL_CAPITAL, COMMISSION_PCT)
    
    # 4. Generate Report
    header = [
        "=" * 62,
        "  ML STRATEGY: TFT + Momentum (Long-only) — SBER H1",
        "  Konkop Xpress-style Analysis (WITH REINVESTMENT)",
        "=" * 62,
        "",
        "  ─── Strategy Parameters ─────────────────────────────────",
        "  Model: TFT (d=64, h=4, delta_lag1)",
        "  Filter: Momentum (h=4, atr_thresh=0.3)",
        "  Logic: Long if TFT > 0 AND Momentum > 0",
        f"  Start Capital = {INITIAL_CAPITAL:,.0f}",
        f"  Position Sizing = 100% Equity (Leverage {LEVERAGE}x)",
        f"  Commission = {COMMISSION_PCT}% (round-trip)",
        "",
    ]
    
    report = "\n".join(header) + "\n"
    for m in [m_train, m_test, m_all]:
        if m:
            report += "\n" + format_report(m) + "\n"
            
    # OOS comparison
    if m_train and m_test:
        decay = (1 - m_test["avg_trade_1c"] / m_train["avg_trade_1c"]) * 100 \
            if m_train["avg_trade_1c"] != 0 else 0
        report += "\n" + "=" * 62 + "\n"
        report += "  OOS COMPARISON (Independent Compounding)\n"
        report += "=" * 62 + "\n"
        report += f"                             {'TRAIN':>12}  {'TEST':>12}\n"
        report += f"  Avg trade (1c)         {m_train['avg_trade_1c']:>12.2f}  {m_test['avg_trade_1c']:>12.2f}\n"
        report += f"  Profit factor (1c)     {m_train['pf_1c']:>12.2f}  {m_test['pf_1c']:>12.2f}\n"
        report += f"  Sharpe (1c)            {m_train['sharpe_1c']:>12.2f}  {m_test['sharpe_1c']:>12.2f}\n"
        report += f"  Win rate               {m_train['perc_winning']:>11.2f}%  {m_test['perc_winning']:>11.2f}%\n"
        report += f"  Trades                 {m_train['total_trades']:>12}  {m_test['total_trades']:>12}\n"
        report += f"  Ann. return            {m_train['annual_return_pct']:>11.2f}%  {m_test['annual_return_pct']:>11.2f}%\n"
        report += f"  Max DD                 {m_train['max_dd_pct']:>11.2f}%  {m_test['max_dd_pct']:>11.2f}%\n"
        report += f"  OOS decay (1c)         {decay:>11.1f}%\n"
        report += "=" * 62 + "\n"

    # Year-by-year (using df_all_comp for continuous compounding view)
    df_yearly = df_all_comp.copy()
    
    # We need to manually calculate pnl_1c for the report because apply_compounding didn't set it 
    # (it only set 'contracts' and 'pnl' which is gross total).
    # But 'gross_pnl_1c' is in the DF.
    df_yearly['pnl_1c'] = df_yearly['gross_pnl_1c']
    
    years_list = sorted(df_yearly["year"].unique())
    yearly_rows = []
    
    for yr in years_list:
        dy = df_yearly[df_yearly["year"] == yr]
        n = len(dy)
        pnl_arr = dy["pnl"].values # Gross Total
        p1c = dy["pnl_1c"].values # Gross 1c
        contracts_arr = dy["contracts"].values
        entry_p = dy["entry_price"].values
        exit_p = dy["exit_price"].values

        comm = COMMISSION_PCT / 100.0 * (entry_p + exit_p) * contracts_arr
        pnl_net = pnl_arr - comm

        wins = pnl_net[pnl_net > 0]
        losses = pnl_net[pnl_net < 0]

        gp = float(wins.sum()) if len(wins) > 0 else 0.0
        gl = float(losses.sum()) if len(losses) > 0 else 0.0
        pf = abs(gp / gl) if gl != 0 else float("inf")
        net = gp + gl
        wr = len(wins) / n * 100 if n > 0 else 0.0
        avg_t = float(pnl_net.mean()) if n > 0 else 0.0
        
        # Calculate NET 1c for the report
        comm_1c = (COMMISSION_PCT / 100.0) * (entry_p + exit_p)
        p1c_net = p1c - comm_1c
        avg_1c_net = float(p1c_net.mean()) if n > 0 else 0.0
        total_1c_net = float(p1c_net.sum())
        
        gp_1c = float(p1c_net[p1c_net > 0].sum()) if (p1c_net > 0).any() else 0.0
        gl_1c = float(p1c_net[p1c_net < 0].sum()) if (p1c_net < 0).any() else 0.0
        pf_1c = abs(gp_1c / gl_1c) if gl_1c != 0 else float("inf")

        # Max DD within the year (on sized equity)
        eq = np.cumsum(pnl_net)
        rm = np.maximum.accumulate(eq)
        dd = rm - eq
        mdd = float(dd.max()) if len(dd) > 0 else 0.0

        avg_c = float(contracts_arr.mean()) if n > 0 else 0.0
        is_train = "TR" if yr <= 2018 else "TE"
        
        yearly_rows.append({
            "year": yr, "split": is_train, "trades": n,
            "net_pnl": net, "pf": pf, "win_rate": wr,
            "avg_trade": avg_t, "avg_1c": avg_1c_net,
            "total_1c": total_1c_net, "pf_1c": pf_1c,
            "max_dd": mdd, "avg_contracts": avg_c,
        })

    report += "\n" + "=" * 96 + "\n"
    report += "  YEAR-BY-YEAR BREAKDOWN (With Reinvestment)\n"
    report += "=" * 96 + "\n"
    hdr = (f"  {'Year':>4} {'':>2} {'#Tr':>4}  {'Net PnL':>12}  {'PF':>5}  "
           f"{'WR%':>6}  {'Avg trade':>10}  {'Avg 1c':>8}  "
           f"{'Tot 1c':>9}  {'PF 1c':>5}  {'MaxDD':>10}  {'AvgC':>5}")
    report += hdr + "\n"
    report += "  " + "-" * 92 + "\n"

    for r in yearly_rows:
        pf_s = f"{r['pf']:5.2f}" if r['pf'] < 100 else "  inf"
        pf1_s = f"{r['pf_1c']:5.2f}" if r['pf_1c'] < 100 else "  inf"
        line = (f"  {r['year']:>4} {r['split']:>2} {r['trades']:>4}  "
                f"{r['net_pnl']:>12,.0f}  {pf_s}  "
                f"{r['win_rate']:>5.1f}%  {r['avg_trade']:>10,.0f}  "
                f"{r['avg_1c']:>8.2f}  {r['total_1c']:>9,.0f}  "
                f"{pf1_s}  {r['max_dd']:>10,.0f}  {r['avg_contracts']:>5.1f}")
        report += line + "\n"
    report += "  " + "-" * 92 + "\n"
    
    # Totals
    t_trades = sum(r["trades"] for r in yearly_rows)
    t_net = sum(r["net_pnl"] for r in yearly_rows)
    t_1c = sum(r["total_1c"] for r in yearly_rows)
    t_avg1c = t_1c / t_trades if t_trades > 0 else 0
    report += (f"  {'ALL':>4} {'':>2} {t_trades:>4}  {t_net:>12,.0f}  "
               f"{'':>5}  {'':>6}  {'':>10}  {t_avg1c:>8.2f}  "
               f"{t_1c:>9,.0f}\n")
    report += "=" * 96 + "\n"

    # Output
    report_path = os.path.join(OUT_DIR, "STRATEGY_ML_KONKOP_COMPOUND.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    # Print to console
    safe = report.replace("\u2500", "-")
    try:
        print(safe)
    except UnicodeEncodeError:
        print(safe.encode("ascii", errors="replace").decode("ascii"))
        
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
