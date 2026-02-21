"""Generate strategy comparison report with charts."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "outputs/strategy"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

strategies = [
    ("momentum_only", "Momentum L+S", "#9E9E9E", "--"),
    ("tft_only", "TFT L+S", "#FF9800", "--"),
    ("combined", "TFT+Mom agree L+S", "#795548", "--"),
    ("long_only_mom", "Momentum Long-only", "#2196F3", "-"),
    ("long_only_tft", "TFT Long-only", "#4CAF50", "-"),
    ("long_only_combined", "TFT+Mom Long-only", "#F44336", "-"),
]

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# 1) Equity curves comparison
ax_eq = fig.add_subplot(gs[0, :])
legend_items = []
for name, label, color, ls in strategies:
    path = os.path.join(OUT_DIR, f"strategy_{name}_equity.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            ax_eq.plot(df["trade_num"], df["equity_pct"],
                       label=label, color=color, linestyle=ls, linewidth=1.5, alpha=0.85)
ax_eq.set_title("Equity Curves: All Strategies (SBER H1 h=4)", fontsize=13, fontweight="bold")
ax_eq.set_xlabel("Trade #")
ax_eq.set_ylabel("Cumulative PnL (%)")
ax_eq.legend(fontsize=9, loc="upper left")
ax_eq.grid(True, alpha=0.3)
ax_eq.axhline(y=0, color="black", linewidth=0.5)

# 2) Per-strategy bar chart: PnL%, Sharpe, MaxDD
comp_path = os.path.join(OUT_DIR, "strategy_comparison.csv")
if os.path.exists(comp_path):
    comp = pd.read_csv(comp_path)
    if "mode" not in comp.columns:
        comp["mode"] = [s[0] for s in strategies[:len(comp)]]

    # Aggregate all runs
    all_modes = {}
    for name, label, color, ls in strategies:
        sub = comp[comp["mode"] == name]
        if not sub.empty:
            row = sub.iloc[-1]
            all_modes[label] = {
                "PnL%": row.get("total_pnl_pct", 0),
                "Sharpe": row.get("sharpe", 0),
                "MaxDD%": -row.get("max_drawdown_pct", 0),
                "WR%": row.get("win_rate", 0) * 100,
                "PF": row.get("profit_factor", 0),
                "Trades": row.get("num_trades", 0),
                "color": color,
            }

    if all_modes:
        labels = list(all_modes.keys())
        colors = [all_modes[l]["color"] for l in labels]

        ax_pnl = fig.add_subplot(gs[1, 0])
        vals = [all_modes[l]["PnL%"] for l in labels]
        bars = ax_pnl.barh(labels, vals, color=colors, alpha=0.8)
        ax_pnl.set_xlabel("Total PnL (%)")
        ax_pnl.set_title("Total PnL by Strategy", fontsize=11, fontweight="bold")
        ax_pnl.axvline(x=0, color="black", linewidth=0.5)
        ax_pnl.grid(True, alpha=0.2, axis="x")
        for bar, val in zip(bars, vals):
            ax_pnl.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontsize=8)

        ax_risk = fig.add_subplot(gs[1, 1])
        sharpes = [all_modes[l]["Sharpe"] for l in labels]
        bars = ax_risk.barh(labels, sharpes, color=colors, alpha=0.8)
        ax_risk.set_xlabel("Sharpe Ratio")
        ax_risk.set_title("Sharpe Ratio by Strategy", fontsize=11, fontweight="bold")
        ax_risk.axvline(x=0, color="black", linewidth=0.5)
        ax_risk.grid(True, alpha=0.2, axis="x")
        for bar, val in zip(bars, sharpes):
            ax_risk.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                         f'{val:.2f}', va='center', fontsize=8)

        # 3) MaxDD and Win Rate
        ax_dd = fig.add_subplot(gs[2, 0])
        dds = [abs(all_modes[l]["MaxDD%"]) for l in labels]
        bars = ax_dd.barh(labels, dds, color=colors, alpha=0.8)
        ax_dd.set_xlabel("Max Drawdown (%)")
        ax_dd.set_title("Max Drawdown by Strategy", fontsize=11, fontweight="bold")
        ax_dd.grid(True, alpha=0.2, axis="x")
        for bar, val in zip(bars, dds):
            ax_dd.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', va='center', fontsize=8)

        ax_wr = fig.add_subplot(gs[2, 1])
        wrs = [all_modes[l]["WR%"] for l in labels]
        bars = ax_wr.barh(labels, wrs, color=colors, alpha=0.8)
        ax_wr.set_xlabel("Win Rate (%)")
        ax_wr.set_title("Win Rate by Strategy", fontsize=11, fontweight="bold")
        ax_wr.axvline(x=50, color="gray", linewidth=0.5, linestyle="--")
        ax_wr.grid(True, alpha=0.2, axis="x")
        for bar, val in zip(bars, wrs):
            ax_wr.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', va='center', fontsize=8)

chart_path = os.path.join(OUT_DIR, "strategy_full_comparison.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Full comparison chart saved: {chart_path}")

# Print summary table
print("\n" + "=" * 85)
print("  FULL STRATEGY COMPARISON: SBER H1 h=4")
print("=" * 85)
print(f"  {'Strategy':<22} {'Trades':>7} {'PnL%':>8} {'AvgTrade%':>10} "
      f"{'WR':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>8}")
print(f"  {'-'*80}")
for label in (all_modes if all_modes else {}):
    m = all_modes[label]
    print(f"  {label:<22} {int(m['Trades']):>7} "
          f"{m['PnL%']:>7.1f}% "
          f"{'n/a':>10} "
          f"{m['WR%']:>5.1f}% {m['PF']:>6.2f} "
          f"{m['Sharpe']:>7.2f} {abs(m['MaxDD%']):>7.1f}%")
