"""Quick check: Momentum vs Mean Reversion baselines across horizons on H1."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import build_panel_auto
from src.dataset import train_test_split_by_years
from src.dataset import make_sequences_multi
from src.utils.metrics import direction_accuracy, mae

cfg = Config(
    ticker="SBER", start="2015-01-01", end="2021-12-31",
    test_years=1, data_source="local", timeframe="H1",
    data_path=r"G:\data2", generator="lstm",
)
cfg.apply_timeframe_defaults()
panel = build_panel_auto(cfg)
_, test_panel = train_test_split_by_years(panel, cfg.test_years)

print(f"Test panel: {test_panel.shape}")
print()

header = f"{'Horizon':<10} {'MR DirAcc':>10} {'Mom DirAcc':>11} {'Naive(0)':>10} {'N':>6}"
print(header)
print("-" * len(header))

for h in [1, 2, 4, 8, 12]:
    _, yte = make_sequences_multi(test_panel, "SBER", 12, (h,))
    yte = yte.squeeze(-1)

    # Mean Reversion: pred[i] = -y_true[i-1]
    mr = np.zeros_like(yte)
    mr[1:] = -yte[:-1]

    # Momentum: pred[i] = +y_true[i-1]
    mom = np.zeros_like(yte)
    mom[1:] = yte[:-1]

    # Naive: pred = 0
    naive = np.zeros_like(yte)

    y = yte[1:]
    mr_dir = direction_accuracy(y, mr[1:])
    mom_dir = direction_accuracy(y, mom[1:])
    naive_dir = direction_accuracy(y, naive[1:])

    print(f"h={h:<8} {mr_dir:>10.3f} {mom_dir:>11.3f} {naive_dir:>10.3f} {len(y):>6}")

print()
print("MR + Mom should ~= 1.0 (complementary)")
print("If Mom >> 0.5: momentum dominates at this horizon")
print("If MR >> 0.5: mean reversion dominates at this horizon")
