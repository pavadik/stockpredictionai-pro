"""
Konkop Xpress-style detailed trade analysis for Block E strategy.
Outputs metrics for TRAIN, TEST, and ALL periods.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "experiments",
)


def analyse(trades: pd.DataFrame, label: str, capital: float,
            commission_pct: float = 0.01) -> dict:
    """Compute Konkop-style metrics from a trades DataFrame."""
    n = len(trades)
    if n == 0:
        return {}

    pnl = trades["pnl"].values
    pnl_1c = trades["pnl_1c"].values
    contracts = trades["contracts"].values
    entry_p = trades["entry_price"].values
    exit_p = trades["exit_price"].values

    # Commission per round-trip = commission_pct% of (entry+exit)*contracts
    comm_per_trade = commission_pct / 100.0 * (entry_p + exit_p) * contracts
    pnl_net = pnl - comm_per_trade
    pnl_1c_net = pnl_1c - commission_pct / 100.0 * (entry_p + exit_p)

    # Equity curve
    equity_curve = np.cumsum(pnl_net)
    start_equity = capital
    end_equity = capital + equity_curve[-1]

    # Win / Loss
    wins_mask = pnl_net > 0
    loss_mask = pnl_net < 0
    flat_mask = pnl_net == 0
    n_win = int(wins_mask.sum())
    n_loss = int(loss_mask.sum())
    n_flat = int(flat_mask.sum())

    gross_profit = float(pnl_net[wins_mask].sum()) if n_win > 0 else 0.0
    gross_loss = float(pnl_net[loss_mask].sum()) if n_loss > 0 else 0.0
    net_profit = gross_profit + gross_loss
    total_commission = float(comm_per_trade.sum())

    pf = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    avg_win = float(pnl_net[wins_mask].mean()) if n_win > 0 else 0.0
    avg_loss = float(pnl_net[loss_mask].mean()) if n_loss > 0 else 0.0
    avg_trade = float(pnl_net.mean())
    std_trade = float(pnl_net.std(ddof=1)) if n > 1 else 0.0

    avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    avg_trade_stddev = avg_trade / std_trade if std_trade != 0 else 0.0

    largest_win = float(pnl_net[wins_mask].max()) if n_win > 0 else 0.0
    largest_loss = float(pnl_net[loss_mask].min()) if n_loss > 0 else 0.0

    # Max consecutive wins/losses
    def max_consecutive(mask):
        best = 0
        cur = 0
        for v in mask:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    max_cw = max_consecutive(wins_mask)
    max_cl = max_consecutive(loss_mask)

    # Max Drawdown (from equity curve, absolute and %)
    running_max = np.maximum.accumulate(capital + equity_curve)
    drawdowns = running_max - (capital + equity_curve)
    max_dd_abs = float(drawdowns.max())
    dd_pct_arr = drawdowns / running_max
    max_dd_pct = float(dd_pct_arr.max()) * 100

    # Period
    entry_times = pd.to_datetime(trades["entry_time"])
    exit_times = pd.to_datetime(trades["exit_time"])
    start_date = entry_times.min()
    end_date = exit_times.max()
    years = max((end_date - start_date).days / 365.25, 0.01)

    total_return_pct = net_profit / capital * 100
    annual_return_pct = ((1 + net_profit / capital) ** (1 / years) - 1) * 100
    ann_ret_max_dd = annual_return_pct / max_dd_pct if max_dd_pct != 0 else 0.0

    # Math expectation as %
    math_exp_pct = avg_trade / capital * 100

    # Per-1-contract stats
    wins_1c_mask = pnl_1c > 0
    loss_1c_mask = pnl_1c < 0
    n_win_1c = int(wins_1c_mask.sum())
    n_loss_1c = int(loss_1c_mask.sum())
    avg_win_1c = float(pnl_1c[wins_1c_mask].mean()) if n_win_1c > 0 else 0.0
    avg_loss_1c = float(pnl_1c[loss_1c_mask].mean()) if n_loss_1c > 0 else 0.0
    avg_trade_1c = float(pnl_1c.mean())
    std_trade_1c = float(pnl_1c.std(ddof=1)) if n > 1 else 0.0
    gp_1c = float(pnl_1c[wins_1c_mask].sum()) if n_win_1c > 0 else 0.0
    gl_1c = float(pnl_1c[loss_1c_mask].sum()) if n_loss_1c > 0 else 0.0
    pf_1c = abs(gp_1c / gl_1c) if gl_1c != 0 else float("inf")
    sharpe_1c = avg_trade_1c / std_trade_1c * np.sqrt(n) if std_trade_1c != 0 else 0.0

    # Holding period
    hold_bars = (exit_times - entry_times)
    avg_hold_hours = hold_bars.mean().total_seconds() / 3600
    max_hold_hours = hold_bars.max().total_seconds() / 3600
    min_hold_hours = hold_bars.min().total_seconds() / 3600

    # Avg contracts
    avg_contracts = float(contracts.mean())
    min_contracts = int(contracts.min())
    max_contracts = int(contracts.max())

    return {
        "label": label,
        "start_date": start_date.strftime("%d.%m.%Y"),
        "end_date": end_date.strftime("%d.%m.%Y"),
        "start_equity": capital,
        "end_equity": end_equity,
        "total_trades": n,
        "win_trades": n_win,
        "loss_trades": n_loss,
        "flat_trades": n_flat,
        "perc_winning": n_win / n * 100,
        "perc_losing": n_loss / n * 100,
        "profit_factor": pf,
        "net_profit": net_profit,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "total_commission": total_commission,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_trade": avg_trade,
        "std_trade": std_trade,
        "math_exp_pct": math_exp_pct,
        "avg_win_loss": avg_win_loss_ratio,
        "avg_trade_stddev": avg_trade_stddev,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "max_conseq_win": max_cw,
        "max_conseq_loss": max_cl,
        "max_dd_abs": max_dd_abs,
        "max_dd_pct": max_dd_pct,
        "total_return_pct": total_return_pct,
        "annual_return_pct": annual_return_pct,
        "ann_ret_max_dd": ann_ret_max_dd,
        "avg_trade_1c": avg_trade_1c,
        "std_trade_1c": std_trade_1c,
        "pf_1c": pf_1c,
        "sharpe_1c": sharpe_1c,
        "total_pnl_1c": gp_1c + gl_1c,
        "avg_hold_hours": avg_hold_hours,
        "max_hold_hours": max_hold_hours,
        "min_hold_hours": min_hold_hours,
        "avg_contracts": avg_contracts,
        "min_contracts": min_contracts,
        "max_contracts": max_contracts,
        "years": years,
    }


def format_report(m: dict) -> str:
    """Format a single period analysis into a Konkop-style text block."""
    sep = "=" * 62
    lines = [
        sep,
        f"  {m['label']}",
        sep,
        "",
        f"  Start date       {m['start_date']:>14}    End date      {m['end_date']:>14}",
        f"  Start equity   {m['start_equity']:>14,.0f}    End equity  {m['end_equity']:>14,.0f}",
        f"  Commission (round-trip)               {m['total_commission']:>14,.0f}",
        "",
        "  ─── Trade Statistics ─────────────────────────────────────",
        f"  Total trades          {m['total_trades']:>8}    Profit factor          {m['profit_factor']:>8.2f}",
        f"  Math. expectation  {m['math_exp_pct']:>8.2f} %    Avg. win/loss          {m['avg_win_loss']:>8.2f}",
        f"  StdDev. trade      {m['std_trade']:>11,.0f}    Avg.trade/StdDev       {m['avg_trade_stddev']:>8.2f}",
        "",
        "  ─── Profit / Loss ───────────────────────────────────────",
        f"  Net profit       {m['net_profit']:>14,.0f}    Max DD             {m['max_dd_pct']:>8.2f} %",
        f"  Gross profit     {m['gross_profit']:>14,.0f}    Gross loss     {m['gross_loss']:>14,.0f}",
        "",
        "  ─── Win / Loss Breakdown ────────────────────────────────",
        f"  Win. trades           {m['win_trades']:>8}    Los. trades            {m['loss_trades']:>8}",
        f"  Perc. winning      {m['perc_winning']:>8.2f} %    Perc. losing         {m['perc_losing']:>8.2f} %",
        f"  Avg. win         {m['avg_win']:>14,.0f}    Avg. loss          {m['avg_loss']:>11,.0f}",
        f"  Largest win      {m['largest_win']:>14,.0f}    Largest loss       {m['largest_loss']:>11,.0f}",
        f"  Max conseq.win        {m['max_conseq_win']:>8}    Max conseq.loss        {m['max_conseq_loss']:>8}",
        "",
        "  ─── Returns ─────────────────────────────────────────────",
        f"  Total return       {m['total_return_pct']:>8.2f} %",
        f"  Annual return      {m['annual_return_pct']:>8.2f} %",
        f"  Ann.ret/max.DD     {m['ann_ret_max_dd']:>8.2f}",
        f"  Max DD (abs)     {m['max_dd_abs']:>14,.0f}",
        "",
        "  ─── Per 1 Contract ──────────────────────────────────────",
        f"  Avg trade (1c)     {m['avg_trade_1c']:>11,.1f} pts   Total PnL (1c)   {m['total_pnl_1c']:>11,.0f} pts",
        f"  StdDev (1c)        {m['std_trade_1c']:>11,.1f} pts   Profit factor(1c)    {m['pf_1c']:>8.2f}",
        f"  Sharpe (1c)            {m['sharpe_1c']:>8.2f}",
        "",
        "  ─── Position Sizing ─────────────────────────────────────",
        f"  Avg contracts      {m['avg_contracts']:>11.1f}    Min / Max        {m['min_contracts']:>5} / {m['max_contracts']:<5}",
        "",
        "  ─── Holding Period ──────────────────────────────────────",
        f"  Avg hold           {m['avg_hold_hours']:>8.1f} hrs   Min hold         {m['min_hold_hours']:>8.1f} hrs",
        f"  Max hold           {m['max_hold_hours']:>8.1f} hrs   Period           {m['years']:>8.2f} yrs",
        sep,
    ]
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="long", choices=["long", "short"])
    cli = ap.parse_args()

    tag = "_short" if cli.direction == "short" else ""
    trades_path = os.path.join(OUT_DIR, f"block_e{tag}_all_trades.csv")
    params_path = os.path.join(OUT_DIR, f"block_e{tag}_best_params.json")

    df = pd.read_csv(trades_path)
    with open(params_path, "r") as f:
        params = json.load(f)

    capital = 5_000_000
    commission_pct = 0.01

    df_train = df[df["period"] == "train"].copy()
    df_test = df[df["period"] == "test"].copy()

    m_train = analyse(df_train, "TRAIN  (2007 - 2018)", capital, commission_pct)
    m_test = analyse(df_test, "TEST / OOS  (2019 - 2022)", capital, commission_pct)
    m_all = analyse(df, "ALL  (2007 - 2022)", capital, commission_pct)

    dir_label = cli.direction.upper()
    header = [
        "=" * 62,
        f"  BLOCK E: Momentum-Trend Strategy — RTSF ({dir_label})",
        "  Konkop Xpress-style Analysis",
        "=" * 62,
        "",
        "  ─── Strategy Parameters ─────────────────────────────────",
        f"  tf1 = {params.get('tf1', '?')} min    tf2 = {params.get('tf2', '?')} min",
        f"  lookback = {params.get('lookback')}    length = {params.get('length')}",
        f"  lookback2 = {params.get('lookback2')}   length2 = {params.get('length2')}",
        f"  min_s = {params.get('min_s')}  max_s = {params.get('max_s')}",
        f"  koeff1 = {params.get('koeff1', 0):.4f}  koeff2 = {params.get('koeff2', 0):.4f}",
        f"  mmcoff = {params.get('mmcoff')}  exitday = {params.get('exitday')}  "
        f"sdel_day = {params.get('sdel_day')}  exit_week = {params.get('exit_week')}",
        f"  Capital = {capital:,.0f}    Commission = {commission_pct}% (round-trip)",
        f"  Position sizing: floor(capital / (300 * ATR({params.get('mmcoff')})))",
        "",
    ]

    report = "\n".join(header) + "\n"
    for m in [m_train, m_test, m_all]:
        report += "\n" + format_report(m) + "\n"

    # OOS comparison
    if m_train and m_test:
        decay = (1 - m_test["avg_trade_1c"] / m_train["avg_trade_1c"]) * 100 \
            if m_train["avg_trade_1c"] != 0 else 0
        report += "\n" + "=" * 62 + "\n"
        report += "  OOS COMPARISON\n"
        report += "=" * 62 + "\n"
        report += f"                             {'TRAIN':>12}  {'TEST':>12}\n"
        report += f"  Avg trade (1c)         {m_train['avg_trade_1c']:>12.1f}  {m_test['avg_trade_1c']:>12.1f}\n"
        report += f"  Profit factor (1c)     {m_train['pf_1c']:>12.2f}  {m_test['pf_1c']:>12.2f}\n"
        report += f"  Sharpe (1c)            {m_train['sharpe_1c']:>12.2f}  {m_test['sharpe_1c']:>12.2f}\n"
        report += f"  Win rate               {m_train['perc_winning']:>11.2f}%  {m_test['perc_winning']:>11.2f}%\n"
        report += f"  Trades                 {m_train['total_trades']:>12}  {m_test['total_trades']:>12}\n"
        report += f"  Ann. return            {m_train['annual_return_pct']:>11.2f}%  {m_test['annual_return_pct']:>11.2f}%\n"
        report += f"  Max DD                 {m_train['max_dd_pct']:>11.2f}%  {m_test['max_dd_pct']:>11.2f}%\n"
        report += f"  OOS decay (1c)         {decay:>11.1f}%\n"
        report += "=" * 62 + "\n"

    # --- Year-by-year breakdown ---
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["year"] = df["entry_time"].dt.year
    years_list = sorted(df["year"].unique())

    yearly_rows = []
    for yr in years_list:
        dy = df[df["year"] == yr]
        n = len(dy)
        pnl_arr = dy["pnl"].values
        p1c = dy["pnl_1c"].values
        contracts_arr = dy["contracts"].values
        entry_p = dy["entry_price"].values
        exit_p = dy["exit_price"].values

        comm = commission_pct / 100.0 * (entry_p + exit_p) * contracts_arr
        pnl_net = pnl_arr - comm

        wins = pnl_net[pnl_net > 0]
        losses = pnl_net[pnl_net < 0]

        gp = float(wins.sum()) if len(wins) > 0 else 0.0
        gl = float(losses.sum()) if len(losses) > 0 else 0.0
        pf = abs(gp / gl) if gl != 0 else float("inf")
        net = gp + gl
        wr = len(wins) / n * 100 if n > 0 else 0.0
        avg_t = float(pnl_net.mean()) if n > 0 else 0.0
        avg_1c = float(p1c.mean()) if n > 0 else 0.0
        std_1c = float(p1c.std(ddof=1)) if n > 1 else 0.0
        total_1c = float(p1c.sum())

        gp_1c = float(p1c[p1c > 0].sum()) if (p1c > 0).any() else 0.0
        gl_1c = float(p1c[p1c < 0].sum()) if (p1c < 0).any() else 0.0
        pf_1c = abs(gp_1c / gl_1c) if gl_1c != 0 else float("inf")

        # Max DD within the year (on sized equity, starting from 0)
        eq = np.cumsum(pnl_net)
        rm = np.maximum.accumulate(eq)
        dd = rm - eq
        mdd = float(dd.max()) if len(dd) > 0 else 0.0

        avg_c = float(contracts_arr.mean()) if n > 0 else 0.0

        is_train = "TR" if yr <= 2018 else "TE"
        yearly_rows.append({
            "year": yr, "split": is_train, "trades": n,
            "net_pnl": net, "pf": pf, "win_rate": wr,
            "avg_trade": avg_t, "avg_1c": avg_1c,
            "total_1c": total_1c, "pf_1c": pf_1c,
            "max_dd": mdd, "avg_contracts": avg_c,
        })

    report += "\n" + "=" * 96 + "\n"
    report += "  YEAR-BY-YEAR BREAKDOWN\n"
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
                f"{r['avg_1c']:>8,.1f}  {r['total_1c']:>9,.0f}  "
                f"{pf1_s}  {r['max_dd']:>10,.0f}  {r['avg_contracts']:>5.1f}")
        report += line + "\n"

    report += "  " + "-" * 92 + "\n"

    # Totals row
    t_trades = sum(r["trades"] for r in yearly_rows)
    t_net = sum(r["net_pnl"] for r in yearly_rows)
    t_1c = sum(r["total_1c"] for r in yearly_rows)
    t_avg1c = t_1c / t_trades if t_trades > 0 else 0
    report += (f"  {'ALL':>4} {'':>2} {t_trades:>4}  {t_net:>12,.0f}  "
               f"{'':>5}  {'':>6}  {'':>10}  {t_avg1c:>8,.1f}  "
               f"{t_1c:>9,.0f}\n")
    report += "=" * 96 + "\n"

    # Save yearly CSV
    yearly_df = pd.DataFrame(yearly_rows)
    yearly_csv = os.path.join(OUT_DIR, f"block_e{tag}_yearly.csv")
    yearly_df.to_csv(yearly_csv, index=False)

    report_path = os.path.join(OUT_DIR, f"BLOCK_E{tag.upper()}_KONKOP.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    safe = report.replace("\u2500", "-")
    try:
        print(safe)
    except UnicodeEncodeError:
        print(safe.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nSaved -> {report_path}")
    print(f"Saved -> {yearly_csv}")


if __name__ == "__main__":
    main()
