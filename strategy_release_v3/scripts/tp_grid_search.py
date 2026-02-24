"""Grid search for partial take-profit parameters (tp_atr_mult, tp_pct)."""
import subprocess, sys, re

BASE_CMD = [
    sys.executable, "-u", "scripts/run_block_e_combined.py",
    "--train_start", "2006-01-01", "--test_end", "2026-02-28",
    "--commission", "0.01", "--flip_mode", "close_loss",
    "--leverage", "2.0", "--max_contracts", "100",
]

TP_MULTS = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
TP_PCTS  = [0.25, 0.5, 0.75]

rx_net   = re.compile(r"Net profit\s+([\d,]+)\s+([\d,]+)")
rx_maxdd = re.compile(r"Max DD\s+([\d.]+)%\s+([\d.]+)%")
rx_pf    = re.compile(r"Profit factor \(1c\)\s+([\d.]+)\s+([\d.]+)")
rx_tr    = re.compile(r"Trades\s+(\d+)\s+(\d+)")
rx_all_net = re.compile(r"ALL\s+\d+\s+\d+\s+\d+\s+\d+\s+([\d,\-]+)")

def parse(output):
    m = {}
    for line in output.split("\n"):
        g = rx_net.search(line)
        if g:
            m["train_pnl"] = int(g.group(1).replace(",", ""))
            m["test_pnl"]  = int(g.group(2).replace(",", ""))
        g = rx_maxdd.search(line)
        if g:
            m["train_dd"] = float(g.group(1))
            m["test_dd"]  = float(g.group(2))
        g = rx_pf.search(line)
        if g:
            m["train_pf"] = float(g.group(1))
            m["test_pf"]  = float(g.group(2))
        g = rx_tr.search(line)
        if g:
            m["train_tr"] = int(g.group(1))
            m["test_tr"]  = int(g.group(2))
        g = rx_all_net.search(line)
        if g:
            m["total_pnl"] = int(g.group(1).replace(",", ""))
    return m

results = []

for mult in TP_MULTS:
    for pct in TP_PCTS:
        if mult == 0.0 and pct != TP_PCTS[0]:
            continue
        label = f"TP={mult:.0f}ATR x{pct:.0%}" if mult > 0 else "NONE"
        cmd = BASE_CMD.copy()
        if mult > 0:
            cmd += ["--tp_atr_mult", str(mult), "--tp_pct", str(pct)]
        print(f">>> Running {label} ...")
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
        print(f"    Total={total:>12,}  TestDD={test_dd:.1f}%  TestPF={test_pf:.2f}")

print("\n" + "=" * 110)
hdr = f"{'Config':>18} | {'Total PnL':>12} | {'Train PnL':>12} | {'Test PnL':>11} | {'TestDD':>6} | {'TestPF':>6} | {'Trades':>6}"
print(hdr)
print("-" * 110)
for r in results:
    print(f"{r['label']:>18} | {r.get('total_pnl',0):>12,} | {r.get('train_pnl',0):>12,} | {r.get('test_pnl',0):>11,} | {r.get('test_dd',0):>5.1f}% | {r.get('test_pf',0):>6.2f} | {r.get('train_tr',0)+r.get('test_tr',0):>6}")
print("=" * 110)
