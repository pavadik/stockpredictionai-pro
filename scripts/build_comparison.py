"""Rebuild strategy_comparison.csv from per-mode split files and generate report."""
import os
import sys
import numpy as np
import pandas as pd

OUT_DIR = "outputs/strategy"

modes = {
    "momentum_only":      "Momentum L+S",
    "tft_only":           "TFT L+S",
    "combined":           "TFT+Mom agree L+S",
    "long_only_mom":      "Momentum Long-only",
    "long_only_tft":      "TFT Long-only",
    "long_only_combined": "TFT+Mom Long-only",
}

rows = []
for mode, label in modes.items():
    eq_path = os.path.join(OUT_DIR, f"strategy_{mode}_equity.csv")
    tr_path = os.path.join(OUT_DIR, f"strategy_{mode}_trades.csv")
    if not os.path.exists(eq_path) or not os.path.exists(tr_path):
        continue
    eq = pd.read_csv(eq_path)
    tr = pd.read_csv(tr_path)
    if eq.empty:
        rows.append({"mode": mode, "label": label, "num_trades": 0,
                      "total_pnl_pct": 0, "avg_trade_pct": 0, "win_rate": 0,
                      "profit_factor": 0, "max_drawdown_pct": 0, "sharpe": 0})
        continue

    pnls = eq["pnl_pct"].values
    n = len(pnls)
    total = pnls.sum()
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    wr = len(wins) / n if n > 0 else 0
    gp = wins.sum() if len(wins) else 0
    gl = abs(losses.sum()) if len(losses) else 0
    pf = gp / gl if gl > 0 else float("inf")
    cum = np.cumsum(pnls)
    rmax = np.maximum.accumulate(cum)
    mdd = (rmax - cum).max()
    sharpe = (pnls.mean() / pnls.std()) * np.sqrt(min(250, n)) if n > 1 and pnls.std() > 0 else 0

    long_n = (tr["direction"] == "long").sum() if "direction" in tr.columns else 0
    short_n = (tr["direction"] == "short").sum() if "direction" in tr.columns else 0
    long_pnl = tr.loc[tr["direction"] == "long", "pnl_pct"].sum() if "direction" in tr.columns else 0
    short_pnl = tr.loc[tr["direction"] == "short", "pnl_pct"].sum() if "direction" in tr.columns else 0

    rows.append({
        "mode": mode, "label": label,
        "num_trades": n, "total_pnl_pct": round(total, 2),
        "avg_trade_pct": round(total / n, 4) if n > 0 else 0,
        "win_rate": round(wr, 4), "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(mdd, 2), "sharpe": round(sharpe, 3),
        "long_trades": long_n, "short_trades": short_n,
        "long_pnl_pct": round(long_pnl, 2), "short_pnl_pct": round(short_pnl, 2),
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "strategy_comparison.csv"), index=False)

print("=" * 90)
print("  FULL STRATEGY COMPARISON: SBER H1 h=4 | Walk-forward 5 splits")
print("=" * 90)
print(f"  {'Strategy':<24} {'Trades':>7} {'PnL%':>8} {'Avg%':>8} "
      f"{'WR':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>8}")
print(f"  {'-'*82}")
for _, r in df.iterrows():
    pnl = r.get("total_pnl_pct", 0)
    print(f"  {r['label']:<24} {int(r['num_trades']):>7} "
          f"{pnl:>7.1f}% {r['avg_trade_pct']:>7.4f}% "
          f"{r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} "
          f"{r['sharpe']:>7.2f} {r['max_drawdown_pct']:>7.1f}%")

print(f"\n  WINNER: ", end="")
best = df.loc[df["sharpe"].idxmax()]
print(f"{best['label']} (Sharpe {best['sharpe']:.2f}, PnL {best['total_pnl_pct']:.1f}%)")
