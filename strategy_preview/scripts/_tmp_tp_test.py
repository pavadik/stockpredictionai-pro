import subprocess
import os

def parse_report(output):
    """Simple parser to extract OOS metrics from the script output."""
    lines = output.split('\n')
    metrics = {}
    in_oos = False
    
    for line in lines:
        if 'OOS COMPARISON' in line:
            in_oos = True
            continue
            
        if in_oos:
            if 'Avg trade (1c)' in line:
                metrics['avg_trade'] = float(line.split()[-1])
            elif 'Profit factor (1c)' in line:
                metrics['pf'] = float(line.split()[-1])
            elif 'Sharpe (1c)' in line:
                metrics['sharpe'] = float(line.split()[-1])
            elif 'Win rate' in line:
                metrics['wr'] = float(line.split()[-1].replace('%', ''))
            elif 'Net profit' in line:
                metrics['net_pnl'] = float(line.split()[-1].replace(',', ''))
            elif 'Max DD' in line:
                metrics['max_dd'] = float(line.split()[-1].replace('%', ''))
            elif 'Ann.ret / maxDD' in line:
                metrics['ann_ret_dd'] = float(line.split()[-1])
                break
                
    return metrics

def P(msg=""):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode())

# Base parameters
cmd_base = [
    "python", "scripts/run_block_e_combined.py",
    "--cap_long", "50", "--cap_short", "20",
    "--flip_mode", "close_loss",
    "--train_start", "2006-01-01", "--test_end", "2026-02-28",
    "--commission", "0.01"
]

# We will test closing 50% of position at 2, 3, 4, 5, 6 ATRs.
tp_pct = 0.5
tp_multipliers = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0]

results = []

P("Running Take-Profit Scale-out Analysis...")
P("=" * 100)

for atr_mult in tp_multipliers:
    P(f"Testing TP: {atr_mult} ATR (Close {tp_pct*100}% of position)")
    
    cmd = cmd_base.copy()
    if atr_mult > 0:
        cmd.extend(["--tp_atr_mult", str(atr_mult), "--tp_pct", str(tp_pct)])
        
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        P(f"Error running cmd: {res.stderr}")
        continue
        
    metrics = parse_report(res.stdout)
    if not metrics:
        P("Failed to parse metrics!")
        continue
        
    metrics['atr_mult'] = atr_mult
    results.append(metrics)

P("=" * 100)
hdr = (
    f"{'TP ATR':>6} | {'OOS PnL':>11} | {'Avg 1c':>8} | {'PF 1c':>6} | "
    f"{'Sharpe':>6} | {'WR%':>5} | {'MaxDD%':>7} | {'Ret/DD':>6}"
)
P(hdr)
P("-" * 100)

for r in results:
    label = "None" if r['atr_mult'] == 0 else f"{r['atr_mult']:.1f}"
    line = (
        f"{label:>6} | {r['net_pnl']:>11,.0f} | {r['avg_trade']:>8.1f} | "
        f"{r['pf']:>6.2f} | {r['sharpe']:>6.2f} | {r['wr']:>4.1f}% | "
        f"{r['max_dd']:>6.1f}% | {r['ann_ret_dd']:>6.2f}"
    )
    P(line)
