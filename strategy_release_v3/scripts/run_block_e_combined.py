"""Block E Combined: unified LONG+SHORT simulation with flip logic.

Runs both LONG and SHORT signal generation on M1 bars, then feeds
them into a single backtester that holds ONE position at a time.
When an opposing entry fires, the current position is closed ("flip")
and the new direction is opened.  Capital is shared (5M total).

Usage:
    python scripts/run_block_e_combined.py
    python scripts/run_block_e_combined.py --capital 5000000 --commission 0.01
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data_local import load_m1_bars
from src.strategy.momentum_trend import (
    StrategyParams,
    prepare_strategy_data,
    generate_signals,
)
from src.strategy.backtester import (
    simulate_combined_trades,
    compute_metrics,
    trades_to_dataframe,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "experiments",
)


def _build_params(json_dict: dict, capital: float, point_mult: float,
                  direction: str) -> StrategyParams:
    remap = {"tf1": "tf1_minutes", "tf2": "tf2_minutes"}
    d = {remap.get(k, k): v for k, v in json_dict.items()}
    d["capital"] = capital
    d["point_value_mult"] = point_mult
    d["direction"] = direction
    d.pop("exit_week", None)
    return StrategyParams(**d)


def _compute_1c_pnl(trades_df: pd.DataFrame) -> np.ndarray:
    """Per-1-contract PnL respecting direction."""
    signs = np.where(trades_df["direction"] == "short", -1.0, 1.0)
    return signs * (trades_df["exit_price"].values - trades_df["entry_price"].values)


def _yearly_breakdown(df: pd.DataFrame, cpct: float) -> list:
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["year"] = df["entry_time"].dt.year
    rows = []
    for yr in sorted(df["year"].unique()):
        dy = df[df["year"] == yr]
        n = len(dy)
        pnl_arr = dy["pnl"].values
        p1c = dy["pnl_1c"].values
        contracts_arr = dy["contracts"].values
        entry_p = dy["entry_price"].values
        exit_p = dy["exit_price"].values

        comm = cpct / 100.0 * (entry_p + exit_p) * contracts_arr
        pnl_net = pnl_arr - comm

        n_long = int((dy["direction"] == "long").sum())
        n_short = int((dy["direction"] == "short").sum())
        n_flip = int((dy["exit_reason"] == "flip").sum())

        wins = pnl_net[pnl_net > 0]
        losses = pnl_net[pnl_net < 0]
        gp = float(wins.sum()) if len(wins) > 0 else 0.0
        gl = float(losses.sum()) if len(losses) > 0 else 0.0
        pf = abs(gp / gl) if gl != 0 else float("inf")
        net = gp + gl
        wr = len(wins) / n * 100 if n > 0 else 0.0
        avg_t = float(pnl_net.mean()) if n > 0 else 0.0
        avg_1c = float(p1c.mean()) if n > 0 else 0.0
        total_1c = float(p1c.sum())

        gp_1c = float(p1c[p1c > 0].sum()) if (p1c > 0).any() else 0.0
        gl_1c = float(p1c[p1c < 0].sum()) if (p1c < 0).any() else 0.0
        pf_1c = abs(gp_1c / gl_1c) if gl_1c != 0 else float("inf")

        eq = np.cumsum(pnl_net)
        rm = np.maximum.accumulate(eq)
        dd = rm - eq
        mdd = float(dd.max()) if len(dd) > 0 else 0.0
        avg_c = float(contracts_arr.mean()) if n > 0 else 0.0

        split = "TR" if yr <= 2018 else "TE"
        rows.append({
            "year": yr, "split": split, "trades": n,
            "n_long": n_long, "n_short": n_short, "n_flip": n_flip,
            "net_pnl": net, "pf": pf, "win_rate": wr,
            "avg_trade": avg_t, "avg_1c": avg_1c,
            "total_1c": total_1c, "pf_1c": pf_1c,
            "max_dd": mdd, "avg_contracts": avg_c,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Block E Combined LONG+SHORT with flip logic")
    ap.add_argument("--ticker", default="RTSF")
    ap.add_argument("--data_path", default=r"G:\data2")
    ap.add_argument("--train_start", default="2007-01-01")
    ap.add_argument("--train_end", default="2018-12-31")
    ap.add_argument("--test_start", default="2019-01-01")
    ap.add_argument("--test_end", default="2022-12-31")
    ap.add_argument("--capital", type=float, default=5_000_000)
    ap.add_argument("--point_mult", type=float, default=300.0)
    ap.add_argument("--commission", type=float, default=0.01)
    ap.add_argument("--max_contracts", type=int, default=0,
                    help="Cap max contracts per trade (0 = no cap)")
    ap.add_argument("--cap_long", type=int, default=0,
                    help="Cap max contracts for LONG (overrides --max_contracts)")
    ap.add_argument("--cap_short", type=int, default=0,
                    help="Cap max contracts for SHORT (overrides --max_contracts)")
    ap.add_argument("--flip_mode", default="flip",
                    choices=["flip", "ignore", "close_only",
                             "flip_profit", "close_loss",
                             "flip_long_close_short"],
                    help="How to handle opposing entry signals")
    ap.add_argument("--leverage", type=float, default=1.0,
                    help="Leverage multiplier for position sizing (1.0 = no leverage)")
    args = ap.parse_args()
    capital = args.capital
    cpct = args.commission

    os.makedirs(OUT_DIR, exist_ok=True)
    sep = "=" * 62

    print(f"\n{sep}")
    print("  BLOCK E: Combined LONG+SHORT (Flip Mode)")
    print(f"{sep}")

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params", "v3_joint_best_long.json")) as f:
        long_json = json.load(f)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params", "v3_joint_best_short.json")) as f:
        short_json = json.load(f)

    params_L = _build_params(long_json, capital, args.point_mult, "long")
    params_S = _build_params(short_json, capital, args.point_mult, "short")

    print(f"  LONG:  tf1={params_L.tf1_minutes} tf2={params_L.tf2_minutes}  "
          f"lookback={params_L.lookback} length={params_L.length}")
    print(f"  SHORT: tf1={params_S.tf1_minutes} tf2={params_S.tf2_minutes}  "
          f"lookback={params_S.lookback} length={params_S.length}")
    cl = args.cap_long or args.max_contracts
    cs = args.cap_short or args.max_contracts
    cap_parts = []
    if cl > 0:
        cap_parts.append(f"L<={cl}")
    if cs > 0:
        cap_parts.append(f"S<={cs}")
    cap_label = ", ".join(cap_parts) if cap_parts else "no cap"
    lev_label = f", leverage={args.leverage:.1f}x" if args.leverage != 1.0 else ""
    print(f"  Capital = {capital:,.0f}  (shared, single position, {cap_label}{lev_label})")
    print(f"  Flip mode: {args.flip_mode}")

    # --- Load M1 data ---
    print(f"\n[1/4] Loading M1 bars ...")
    t0 = time.time()
    df_m1_full = load_m1_bars(args.data_path, args.ticker,
                              args.train_start, args.test_end)
    print(f"  {len(df_m1_full):,} M1 bars in {time.time()-t0:.1f}s")

    train_end_dt = pd.Timestamp(args.train_end)
    test_start_dt = pd.Timestamp(args.test_start)

    results = {}
    for period, df_m1 in [
        ("train", df_m1_full[df_m1_full.index <= train_end_dt]),
        ("test", df_m1_full[df_m1_full.index >= test_start_dt]),
    ]:
        label = "TRAIN" if period == "train" else "TEST (OOS)"
        step = "2" if period == "train" else "3"
        print(f"\n[{step}/4] {label}: generating signals & simulating ...")
        t0 = time.time()

        df_sig_L = prepare_strategy_data(df_m1.copy(), params_L)
        df_sig_L = generate_signals(df_sig_L, params_L)

        df_sig_S = prepare_strategy_data(df_m1.copy(), params_S)
        df_sig_S = generate_signals(df_sig_S, params_S)

        trades = simulate_combined_trades(
            df_sig_L, df_sig_S, params_L, params_S,
            max_contracts=args.max_contracts,
            max_contracts_long=args.cap_long,
            max_contracts_short=args.cap_short,
            flip_mode=args.flip_mode,
            leverage=args.leverage)
        elapsed = time.time() - t0

        metrics = compute_metrics(trades)
        tdf = trades_to_dataframe(trades)
        tdf["pnl_1c"] = _compute_1c_pnl(tdf)
        tdf["period"] = period

        n_long = sum(1 for t in trades if t.direction == "long")
        n_short = sum(1 for t in trades if t.direction == "short")
        n_flip = sum(1 for t in trades if t.exit_reason == "flip")

        print(f"  {len(trades)} trades ({n_long}L + {n_short}S, "
              f"{n_flip} flips) in {elapsed:.1f}s")
        print(f"  avg_trade={metrics['avg_trade']:,.0f}  "
              f"PF={metrics['profit_factor']:.2f}  "
              f"WR={metrics['win_rate']:.1%}  "
              f"maxDD={metrics['max_drawdown']:,.0f}")

        results[period] = {"trades": trades, "metrics": metrics, "df": tdf}

    # --- Combine train + test ---
    print(f"\n[4/4] Building report ...")
    df_all = pd.concat([results["train"]["df"], results["test"]["df"]],
                       ignore_index=True)
    df_all.sort_values("exit_time", inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    comb_csv = os.path.join(OUT_DIR, "block_e_combined_all_trades.csv")
    df_all.to_csv(comb_csv, index=False)
    print(f"  Saved {len(df_all)} trades -> {comb_csv}")

    # --- Konkop analysis (reuse from scripts/konkop_analysis.py) ---
    from scripts.konkop_analysis import analyse, format_report

    df_train = df_all[df_all["period"] == "train"].copy()
    df_test = df_all[df_all["period"] == "test"].copy()

    m_train = analyse(df_train, "COMBINED TRAIN (2007-2018)", capital, cpct)
    m_test = analyse(df_test, "COMBINED TEST / OOS (2019-2022)", capital, cpct)
    m_all = analyse(df_all, "COMBINED ALL (2007-2022)", capital, cpct)

    report_lines = [
        sep,
        "  BLOCK E: Combined Strategy (LONG + SHORT) -- RTSF",
        "  Unified Simulation with Flip Logic",
        sep,
        "",
        f"  Mode: single position, flip_mode={args.flip_mode}",
        f"  Capital: {capital:,.0f} (shared)  Leverage: {args.leverage:.1f}x",
        f"  Max contracts: {cap_label}",
        "",
        "  --- LONG Parameters ---",
        f"  tf1={long_json.get('tf1_minutes', long_json.get('tf1'))}  tf2={long_json.get('tf2_minutes', long_json.get('tf2'))}  "
        f"lookback={long_json['lookback']}  length={long_json['length']}",
        f"  lookback2={long_json['lookback2']}  length2={long_json['length2']}  "
        f"min_s={long_json['min_s']}  max_s={long_json['max_s']}",
        f"  koeff1={long_json['koeff1']:.4f}  koeff2={long_json['koeff2']:.4f}  "
        f"mmcoff={long_json['mmcoff']}  sdel_day={long_json['sdel_day']}",
        "",
        "  --- SHORT Parameters ---",
        f"  tf1={short_json.get('tf1_minutes', short_json.get('tf1'))}  tf2={short_json.get('tf2_minutes', short_json.get('tf2'))}  "
        f"lookback={short_json['lookback']}  length={short_json['length']}",
        f"  lookback2={short_json['lookback2']}  length2={short_json['length2']}  "
        f"min_s={short_json['min_s']}  max_s={short_json['max_s']}",
        f"  koeff1={short_json['koeff1']:.4f}  koeff2={short_json['koeff2']:.4f}  "
        f"mmcoff={short_json['mmcoff']}  sdel_day={short_json['sdel_day']}",
        "",
    ]
    report = "\n".join(report_lines) + "\n"
    for m in [m_train, m_test, m_all]:
        report += "\n" + format_report(m) + "\n"

    if m_train and m_test:
        decay = (1 - m_test["avg_trade_1c"] / m_train["avg_trade_1c"]) * 100 \
            if m_train["avg_trade_1c"] != 0 else 0
        report += f"\n{sep}\n  OOS COMPARISON\n{sep}\n"
        report += f"                             {'TRAIN':>12}  {'TEST':>12}\n"
        report += f"  Avg trade (1c)         {m_train['avg_trade_1c']:>12.1f}  {m_test['avg_trade_1c']:>12.1f}\n"
        report += f"  Profit factor (1c)     {m_train['pf_1c']:>12.2f}  {m_test['pf_1c']:>12.2f}\n"
        report += f"  Sharpe (1c)            {m_train['sharpe_1c']:>12.2f}  {m_test['sharpe_1c']:>12.2f}\n"
        report += f"  Win rate               {m_train['perc_winning']:>11.2f}%  {m_test['perc_winning']:>11.2f}%\n"
        report += f"  Trades                 {m_train['total_trades']:>12}  {m_test['total_trades']:>12}\n"
        report += f"  Net profit             {m_train['net_profit']:>12,.0f}  {m_test['net_profit']:>12,.0f}\n"
        report += f"  Ann. return            {m_train['annual_return_pct']:>11.2f}%  {m_test['annual_return_pct']:>11.2f}%\n"
        report += f"  Max DD                 {m_train['max_dd_pct']:>11.2f}%  {m_test['max_dd_pct']:>11.2f}%\n"
        report += f"  Ann.ret / maxDD        {m_train['ann_ret_max_dd']:>12.2f}  {m_test['ann_ret_max_dd']:>12.2f}\n"
        report += f"  OOS decay (1c)         {decay:>11.1f}%\n"
        report += f"{sep}\n"

    # --- Yearly breakdown ---
    yr_rows = _yearly_breakdown(df_all, cpct)
    yr_df = pd.DataFrame(yr_rows)
    yr_csv = os.path.join(OUT_DIR, "block_e_combined_yearly.csv")
    yr_df.to_csv(yr_csv, index=False)

    w = 118
    report += f"\n{'=' * w}\n  YEAR-BY-YEAR BREAKDOWN (COMBINED, FLIP MODE)\n{'=' * w}\n"
    hdr = (f"  {'Year':>4} {'':>2} {'#Tr':>4} {'L':>3} {'S':>3} {'Flp':>3}  "
           f"{'Net PnL':>12}  {'PF':>5}  {'WR%':>6}  {'Avg trade':>10}  "
           f"{'Avg 1c':>8}  {'Tot 1c':>9}  {'PF 1c':>5}  {'MaxDD':>10}  {'AvgC':>5}")
    report += hdr + "\n"
    report += "  " + "-" * (w - 4) + "\n"

    for r in yr_rows:
        pf_s = f"{r['pf']:5.2f}" if r["pf"] < 100 else "  inf"
        pf1_s = f"{r['pf_1c']:5.2f}" if r["pf_1c"] < 100 else "  inf"
        line = (f"  {r['year']:>4} {r['split']:>2} {r['trades']:>4} "
                f"{r['n_long']:>3} {r['n_short']:>3} {r['n_flip']:>3}  "
                f"{r['net_pnl']:>12,.0f}  {pf_s}  "
                f"{r['win_rate']:>5.1f}%  {r['avg_trade']:>10,.0f}  "
                f"{r['avg_1c']:>8,.1f}  {r['total_1c']:>9,.0f}  "
                f"{pf1_s}  {r['max_dd']:>10,.0f}  {r['avg_contracts']:>5.1f}")
        report += line + "\n"

    report += "  " + "-" * (w - 4) + "\n"
    t_trades = sum(r["trades"] for r in yr_rows)
    t_net = sum(r["net_pnl"] for r in yr_rows)
    t_1c = sum(r["total_1c"] for r in yr_rows)
    t_avg1c = t_1c / t_trades if t_trades > 0 else 0
    t_long = sum(r["n_long"] for r in yr_rows)
    t_short = sum(r["n_short"] for r in yr_rows)
    t_flip = sum(r["n_flip"] for r in yr_rows)
    report += (f"  {'ALL':>4} {'':>2} {t_trades:>4} {t_long:>3} {t_short:>3} "
               f"{t_flip:>3}  {t_net:>12,.0f}  {'':>5}  {'':>6}  {'':>10}  "
               f"{t_avg1c:>8,.1f}  {t_1c:>9,.0f}\n")
    report += "=" * w + "\n"

    # --- Flip stats ---
    report += f"\n{sep}\n  FLIP STATISTICS\n{sep}\n"
    n_total = len(df_all)
    n_flip = int((df_all["exit_reason"] == "flip").sum())
    n_sig = int((df_all["exit_reason"] == "signal_reverse").sum())
    n_eod = int((df_all["exit_reason"] == "end_of_day").sum())
    n_exp = int((df_all["exit_reason"] == "expiration").sum())
    n_other = n_total - n_flip - n_sig - n_eod - n_exp
    report += f"  Total trades:       {n_total}\n"
    report += f"  Flips:              {n_flip} ({n_flip/n_total*100:.1f}%)\n"
    report += f"  Signal reverse:     {n_sig}\n"
    report += f"  End of day:         {n_eod}\n"
    report += f"  Expiration:         {n_exp}\n"
    if n_other > 0:
        report += f"  Other:              {n_other}\n"
    report += f"  Overlapping L+S:    0 (eliminated by flip logic)\n"
    report += f"{sep}\n"

    report_path = os.path.join(OUT_DIR, "BLOCK_E_COMBINED_KONKOP.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    safe = report.replace("\u2500", "-")
    try:
        print(safe)
    except UnicodeEncodeError:
        print(safe.encode("ascii", errors="replace").decode("ascii"))

    print(f"\nSaved -> {report_path}")
    print(f"Saved -> {yr_csv}")
    print(f"Saved -> {comb_csv}")


if __name__ == "__main__":
    main()
