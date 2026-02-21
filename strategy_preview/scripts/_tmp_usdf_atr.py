import sys
sys.path.insert(0, ".")
import pandas as pd
from src.data_local import load_m1_bars

df = load_m1_bars(r"G:\data2", "USDF", "2015-01-01", "2015-12-31")
df_d1 = df.groupby(df.index.date).agg({"high": "max", "low": "min", "close": "last"})
prev_close = df_d1["close"].shift(1)
tr = pd.concat([
    df_d1["high"] - df_d1["low"],
    (df_d1["high"] - prev_close).abs(),
    (df_d1["low"] - prev_close).abs()
], axis=1).max(axis=1)

print(f"Mean D1 ATR for USDF (2015): {tr.mean():.1f} pts")
print(f"Mean Price for USDF (2015): {df_d1['close'].mean():.1f}")
