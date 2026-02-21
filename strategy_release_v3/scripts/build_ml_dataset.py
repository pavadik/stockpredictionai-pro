import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_local import load_m1_bars

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments")

def create_ml_dataset():
    trades_path = os.path.join(OUT_DIR, "block_e_combined_all_trades.csv")
    print(f"Loading trades from {trades_path}...")
    trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
    
    print(f"Loading M1 bars for RTSF to extract features...")
    # Load slightly more data to allow for history features
    m1 = load_m1_bars(r"G:\data2", "RTSF", "2005-12-01", "2026-02-28")
    
    print("Building features...")
    # Daily aggregation for ADX and macro features
    d1 = m1.groupby(m1.index.date).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    })
    
    # Simple ADX calculation
    prev_close = d1["close"].shift(1)
    tr = pd.concat([
        d1["high"] - d1["low"],
        (d1["high"] - prev_close).abs(),
        (d1["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    
    up_move = d1["high"] - d1["high"].shift(1)
    down_move = d1["low"].shift(1) - d1["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=d1.index).ewm(alpha=1/14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=d1.index).ewm(alpha=1/14, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx_d1 = dx.ewm(alpha=1/14, adjust=False).mean()
    
    features = []
    labels = []
    
    for _, trade in trades.iterrows():
        entry_time = trade["entry_time"]
        entry_date = entry_time.date()
        
        # Target: 1 if profitable trade, 0 if losing trade
        is_profitable = 1 if trade["pnl"] > 0 else 0
        labels.append(is_profitable)
        
        # --- Time Features ---
        f = {}
        f["hour"] = entry_time.hour
        f["minute"] = entry_time.minute
        f["day_of_week"] = entry_time.weekday()
        f["month"] = entry_time.month
        
        # --- Macro / D1 Features ---
        # Get yesterday's D1 data
        try:
            loc = d1.index.get_loc(entry_date)
            if loc > 10:
                yest = d1.iloc[loc - 1]
                f["d1_volume"] = yest["volume"]
                f["d1_tr"] = tr.iloc[loc - 1]
                f["d1_adx"] = adx_d1.iloc[loc - 1]
                f["d1_ret_1d"] = yest["close"] / d1.iloc[loc - 2]["close"] - 1.0
                f["d1_ret_5d"] = yest["close"] / d1.iloc[loc - 6]["close"] - 1.0
            else:
                f["d1_volume"] = 0
                f["d1_tr"] = 0
                f["d1_adx"] = 0
                f["d1_ret_1d"] = 0
                f["d1_ret_5d"] = 0
        except KeyError:
            f["d1_volume"] = 0
            f["d1_tr"] = 0
            f["d1_adx"] = 0
            f["d1_ret_1d"] = 0
            f["d1_ret_5d"] = 0
            
        # --- Micro / M1 Features (At entry) ---
        # Get the bar exactly at entry
        try:
            m1_loc = m1.index.get_loc(entry_time)
            if m1_loc > 120:
                m1_entry = m1.iloc[m1_loc]
                f["m1_volume"] = m1_entry["volume"]
                f["m1_ret_15m"] = m1_entry["close"] / m1.iloc[m1_loc - 15]["close"] - 1.0
                f["m1_ret_60m"] = m1_entry["close"] / m1.iloc[m1_loc - 60]["close"] - 1.0
                f["m1_ret_120m"] = m1_entry["close"] / m1.iloc[m1_loc - 120]["close"] - 1.0
            else:
                f["m1_volume"] = 0
                f["m1_ret_15m"] = 0
                f["m1_ret_60m"] = 0
                f["m1_ret_120m"] = 0
        except Exception:
            f["m1_volume"] = 0
            f["m1_ret_15m"] = 0
            f["m1_ret_60m"] = 0
            f["m1_ret_120m"] = 0
            
        # Direction flag (1 for long, 0 for short)
        f["is_long"] = 1 if trade["direction"] == "long" else 0
        
        # Add metadata for tracking
        f["entry_time"] = entry_time
        f["period"] = trade["period"]
        f["pnl_net"] = trade["pnl"]
        
        features.append(f)
        
    df_f = pd.DataFrame(features)
    df_f["target"] = labels
    
    # Save dataset
    out_path = os.path.join(OUT_DIR, "block_e_ml_dataset.csv")
    df_f.to_csv(out_path, index=False)
    print(f"\nML Dataset created: {out_path} ({len(df_f)} rows)")
    print(f"Target distribution: {df_f['target'].mean():.1%} Profitable trades")
    
if __name__ == "__main__":
    create_ml_dataset()