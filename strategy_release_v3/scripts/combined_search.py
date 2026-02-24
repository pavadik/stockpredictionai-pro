"""
Combined search: ML threshold + leverage + partial TP.

For each combination, runs backtest and measures total PnL and MaxDD.
The goal: find the combo that maximizes PnL while keeping MaxDD <= 20%.
"""
import subprocess
import sys
import re
import os

BASE_CMD = [
    sys.executable, "-u", "scripts/run_block_e_combined.py",
    "--train_start", "2006-01-01", "--test_end", "2026-02-28",
    "--commission", "0.01", "--flip_mode", "close_loss",
    "--max_contracts", "100",
]

rx_all = re.compile(r"ALL\s+(\d+)\s+\d+\s+\d+\s+\d+\s+([\d,\-]+)")
rx_net_test = re.compile(r"Net profit\s+[\d,]+\s+([\d,]+)")
rx_dd_test  = re.compile(r"Max DD\s+[\d.]+%\s+([\d.]+)%")
rx_dd_all   = re.compile(r"Max DD\s+([\d.]+)%")
rx_pf_test  = re.compile(r"Profit factor \(1c\)\s+[\d.]+\s+([\d.]+)")

def parse(output):
    m = {}
    lines = output.split("\n")
    for line in lines:
        g = rx_all.search(line)
        if g:
            m["total_trades"] = int(g.group(1))
            m["total_pnl"] = int(g.group(2).replace(",", ""))
        g = rx_net_test.search(line)
        if g:
            m["test_pnl"] = int(g.group(1).replace(",", ""))
        g = rx_dd_test.search(line)
        if g:
            m["test_dd"] = float(g.group(1))
        g = rx_pf_test.search(line)
        if g:
            m["test_pf"] = float(g.group(1))

    for line in lines:
        g = rx_dd_all.search(line)
        if g:
            m.setdefault("train_dd", float(g.group(1)))
    return m


LEVERAGES = [1.0, 2.0, 3.0, 4.0]
TP_CONFIGS = [
    (0.0, 0.0, "noTP"),
    (3.0, 0.5, "TP3x50%"),
    (4.0, 0.5, "TP4x50%"),
    (5.0, 0.5, "TP5x50%"),
    (8.0, 0.25, "TP8x25%"),
]

results = []

for lev in LEVERAGES:
    for tp_mult, tp_pct, tp_label in TP_CONFIGS:
        label = f"L{lev:.0f}x_{tp_label}"
        cmd = BASE_CMD + ["--leverage", str(lev)]
        if tp_mult > 0:
            cmd += ["--tp_atr_mult", str(tp_mult), "--tp_pct", str(tp_pct)]

        print(f">>> {label} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    ERROR: {r.stderr[:200]}")
            continue
        m = parse(r.stdout)
        m["label"] = label
        results.append(m)
        total = m.get("total_pnl", 0)
        test_dd = m.get("test_dd", 0)
        test_pf = m.get("test_pf", 0)
        trades = m.get("total_trades", 0)
        print(f"    Total={total:>12,}  TestDD={test_dd:.1f}%  TestPF={test_pf:.2f}  Trades={trades}")

print("\n" + "=" * 120)
hdr = f"{'Config':>18} | {'Total PnL':>12} | {'Test PnL':>11} | {'TestDD':>6} | {'TestPF':>6} | {'Trades':>6}"
print(hdr)
print("-" * 120)
for r in sorted(results, key=lambda x: -x.get("total_pnl", 0)):
    dd_flag = " !!!" if r.get("test_dd", 0) > 20 else ""
    print(f"{r['label']:>18} | {r.get('total_pnl',0):>12,} | {r.get('test_pnl',0):>11,} | {r.get('test_dd',0):>5.1f}% | {r.get('test_pf',0):>6.2f} | {r.get('total_trades',0):>6}{dd_flag}")
print("=" * 120)
