import pandas as pd
import numpy as np
import os
import sys
import time
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_local import load_m1_bars
from src.strategy.momentum_trend import StrategyParams, prepare_strategy_data, generate_signals
from src.strategy.backtester import simulate_trades, trades_to_dataframe

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments")
PARAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params")

def precompute_features(df_m1: pd.DataFrame):
    print("Precomputing ML features for entire dataset to speed up joint optimization...")
    
    # D1 Features
    df_d1 = df_m1.groupby(df_m1.index.date).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    })
    
    prev_close = df_d1["close"].shift(1)
    tr = pd.concat([
        df_d1["high"] - df_d1["low"],
        (df_d1["high"] - prev_close).abs(),
        (df_d1["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    
    up_move = df_d1["high"] - df_d1["high"].shift(1)
    down_move = df_d1["low"].shift(1) - df_d1["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df_d1.index).ewm(alpha=1/14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df_d1.index).ewm(alpha=1/14, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx_d1 = dx.ewm(alpha=1/14, adjust=False).mean()
    
    # We need "yesterday's" data to avoid lookahead
    df_d1["d1_volume"] = df_d1["volume"].shift(1)
    df_d1["d1_tr"] = tr.shift(1)
    df_d1["d1_adx"] = adx_d1.shift(1)
    df_d1["d1_ret_1d"] = df_d1["close"].shift(1) / df_d1["close"].shift(2) - 1.0
    df_d1["d1_ret_5d"] = df_d1["close"].shift(1) / df_d1["close"].shift(6) - 1.0
    
    # Map D1 to M1
    d1_mapped = df_d1[["d1_volume", "d1_tr", "d1_adx", "d1_ret_1d", "d1_ret_5d"]]
    d1_mapped.index = pd.to_datetime(d1_mapped.index)
    
    # M1 Features
    df_m1["m1_volume"] = df_m1["volume"]
    df_m1["m1_ret_15m"] = df_m1["close"] / df_m1["close"].shift(15) - 1.0
    df_m1["m1_ret_60m"] = df_m1["close"] / df_m1["close"].shift(60) - 1.0
    df_m1["m1_ret_120m"] = df_m1["close"] / df_m1["close"].shift(120) - 1.0
    
    # Time features
    df_m1["hour"] = df_m1.index.hour
    df_m1["minute"] = df_m1.index.minute
    df_m1["day_of_week"] = df_m1.index.weekday
    df_m1["month"] = df_m1.index.month
    
    # Add date column for joining with D1
    df_m1["_date"] = pd.to_datetime(df_m1.index.date)
    
    # Merge D1 into M1
    df_features = pd.merge(df_m1, d1_mapped, left_on="_date", right_index=True, how="left")
    df_features.index = df_m1.index
    df_features.drop(columns=["_date", "open", "high", "low", "close", "volume"], inplace=True)
    
    return df_features

def extract_features_for_trades(trades_df: pd.DataFrame, df_features: pd.DataFrame, direction: str) -> pd.DataFrame:
    # Get entry times
    entry_times = trades_df["entry_time"].values
    
    # Extract precomputed features for exactly these times
    # Note: Using .reindex or .loc is fast
    features = df_features.reindex(entry_times).copy()
    
    # Add trade specific labels
    features["target"] = (trades_df["pnl_1c"] > 0).astype(int).values
    features["is_long"] = 1 if direction == "long" else 0
    
    return features.dropna()

def train_and_eval_xgboost(features: pd.DataFrame, trades_df: pd.DataFrame) -> tuple:
    # Need at least 60 trades to even try training
    if len(features) < 60:
        return 0, 0
        
    feat_cols = [
        "hour", "minute", "day_of_week", "month",
        "d1_volume", "d1_tr", "d1_adx", "d1_ret_1d", "d1_ret_5d",
        "m1_volume", "m1_ret_15m", "m1_ret_60m", "m1_ret_120m",
        "is_long"
    ]
    
    X = features[feat_cols]
    y = features["target"]
    
    # Calculate class weight
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    if pos_count == 0: return 0, 0
    scale_pos_weight = neg_count / pos_count
    
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=100, # slightly faster for search
        max_depth=3,      # shallower to prevent overfitting on small sets
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )
    
    model.fit(X, y)
    
    # Predict (In-sample evaluation for parameter selection)
    probs = model.predict_proba(X)[:, 1]
    
    # Filter trades
    # We join back to trades_df to calculate the actual metric
    # The indexes are perfectly aligned because we built `features` from `trades_df["entry_time"]`
    mask = probs >= 0.5
    
    taken_trades = trades_df.loc[features.index[mask]]
    
    num_taken = len(taken_trades)
    if num_taken < 60:
        return 0, num_taken
        
    avg_trade = taken_trades["pnl_1c"].mean()
    
    return avg_trade, num_taken

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_n", type=int, default=100, help="Number of top GPU trials to test")
    args = ap.parse_args()
    
    print(f"Loading top {args.top_n} trials from GPU logs...")
    
    gpu_csv_path = r"G:\StockPredict\GIT\outputs\experiments\block_e_gpu_trials.csv"
    short_csv_path = r"G:\StockPredict\GIT\outputs\experiments\block_e_short_gpu_trials.csv"
    
    df_long_trials = pd.read_csv(gpu_csv_path)
    df_short_trials = pd.read_csv(short_csv_path)
    
    df_long_trials = df_long_trials[df_long_trials["num_trades"] >= 100].sort_values("avg_trade_1c", ascending=False).head(args.top_n)
    df_short_trials = df_short_trials[df_short_trials["num_trades"] >= 100].sort_values("avg_trade_1c", ascending=False).head(args.top_n)
    
    print(f"Loading M1 bars for RTSF (2006-2018)...")
    df_m1_full = load_m1_bars(r"G:\data2", "RTSF", "2006-01-01", "2018-12-31")
    
    # Vectorized feature precomputation
    t0 = time.time()
    df_features = precompute_features(df_m1_full.copy())
    print(f"Precomputed features in {time.time()-t0:.1f}s")
    
    # Let's run the joint optimization for LONG
    print("\n--- OPTIMIZING LONG STRATEGY + ML ---")
    best_long_avg = 0
    best_long_params = None
    best_long_trades = 0
    
    for idx, row in df_long_trials.iterrows():
        p = StrategyParams(
            tf1_minutes=int(row["tf1"]), tf2_minutes=int(row["tf2"]),
            lookback=int(row["lookback"]), length=int(row["length"]),
            lookback2=int(row["lookback2"]), length2=int(row["length2"]),
            min_s=int(row["min_s"]), max_s=int(row["max_s"]),
            koeff1=float(row["koeff1"]), koeff2=float(row["koeff2"]),
            mmcoff=int(row["mmcoff"]), exitday=int(row["exitday"]), sdel_day=int(row["sdel_day"]),
            direction="long"
        )
        
        df_sig = prepare_strategy_data(df_m1_full.copy(), p)
        df_sig = generate_signals(df_sig, p)
        
        # Fast extraction without flip logic overhead (simulate_combined_trades supports single direction)
        trades = simulate_trades(df_sig, p)
        
        if not trades: continue
        
        tdf = trades_to_dataframe(trades)
        tdf["pnl_1c"] = tdf["exit_price"] - tdf["entry_price"]
        
        feat = extract_features_for_trades(tdf, df_features, "long")
        # Ensure indices align perfectly
        tdf.set_index("entry_time", inplace=True)
        tdf = tdf.loc[feat.index]
        
        avg_tr, n_tr = train_and_eval_xgboost(feat, tdf)
        
        if n_tr >= 60 and avg_tr > best_long_avg:
            best_long_avg = avg_tr
            best_long_params = p
            best_long_trades = n_tr
            print(f"New Best LONG! Avg: {avg_tr:.1f} pts | Trades: {n_tr} | Params: tf1={p.tf1_minutes} lookback={p.lookback} koeff1={p.koeff1:.4f}")
            
    # Same for SHORT
    print("\n--- OPTIMIZING SHORT STRATEGY + ML ---")
    best_short_avg = 0
    best_short_params = None
    best_short_trades = 0
    
    for idx, row in df_short_trials.iterrows():
        p = StrategyParams(
            tf1_minutes=int(row["tf1"]), tf2_minutes=int(row["tf2"]),
            lookback=int(row["lookback"]), length=int(row["length"]),
            lookback2=int(row["lookback2"]), length2=int(row["length2"]),
            min_s=int(row["min_s"]), max_s=int(row["max_s"]),
            koeff1=float(row["koeff1"]), koeff2=float(row["koeff2"]),
            mmcoff=int(row["mmcoff"]), exitday=int(row["exitday"]), sdel_day=int(row["sdel_day"]),
            direction="short"
        )
        
        df_sig = prepare_strategy_data(df_m1_full.copy(), p)
        df_sig = generate_signals(df_sig, p)
        
        trades = simulate_trades(df_sig, p)
        
        if not trades: continue
        
        tdf = trades_to_dataframe(trades)
        tdf["pnl_1c"] = tdf["entry_price"] - tdf["exit_price"] # Short PnL
        
        feat = extract_features_for_trades(tdf, df_features, "short")
        tdf.set_index("entry_time", inplace=True)
        tdf = tdf.loc[feat.index]
        
        avg_tr, n_tr = train_and_eval_xgboost(feat, tdf)
        
        if n_tr >= 60 and avg_tr > best_short_avg:
            best_short_avg = avg_tr
            best_short_params = p
            best_short_trades = n_tr
            print(f"New Best SHORT! Avg: {avg_tr:.1f} pts | Trades: {n_tr} | Params: tf1={p.tf1_minutes} lookback={p.lookback} koeff1={p.koeff1:.4f}")

    print("\n--- JOINT OPTIMIZATION RESULTS ---")
    print(f"Best LONG : Avg {best_long_avg:.1f} pts ({best_long_trades} trades)")
    print(f"Best SHORT: Avg {best_short_avg:.1f} pts ({best_short_trades} trades)")
    
    if best_long_params:
        import json
        with open(os.path.join(PARAMS_DIR, "v3_joint_best_long.json"), "w") as f:
            json.dump(vars(best_long_params), f, indent=4)
            
    if best_short_params:
        import json
        with open(os.path.join(PARAMS_DIR, "v3_joint_best_short.json"), "w") as f:
            json.dump(vars(best_short_params), f, indent=4)
            
if __name__ == "__main__":
    main()