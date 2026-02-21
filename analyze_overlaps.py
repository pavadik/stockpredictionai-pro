"""
Analyze LONG+SHORT overlaps in combined trades CSV.
"""
import pandas as pd

CSV_PATH = "outputs/experiments/block_e_combined_all_trades.csv"
df = pd.read_csv(CSV_PATH)
df["entry_time"] = pd.to_datetime(df["entry_time"])
df["exit_time"] = pd.to_datetime(df["exit_time"])

longs = df[df["direction"] == "long"].copy()
shorts = df[df["direction"] == "short"].copy()

print("=" * 70)
print("LONG+SHORT OVERLAP ANALYSIS")
print("=" * 70)
print(f"\nTotal trades: {len(df)} (LONG: {len(longs)}, SHORT: {len(shorts)})")

overlaps = []
for _, long_row in longs.iterrows():
    l_start, l_end = long_row["entry_time"], long_row["exit_time"]
    for _, short_row in shorts.iterrows():
        s_start, s_end = short_row["entry_time"], short_row["exit_time"]
        if l_start < s_end and s_start < l_end:
            overlap_start = max(l_start, s_start)
            overlap_end = min(l_end, s_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds() / 3600
            overlaps.append({
                "long_idx": long_row.name, "short_idx": short_row.name,
                "long_entry": l_start, "long_exit": l_end,
                "short_entry": s_start, "short_exit": s_end,
                "overlap_start": overlap_start, "overlap_end": overlap_end,
                "overlap_hours": overlap_duration,
                "long_pnl": long_row["pnl"], "short_pnl": short_row["pnl"],
                "combined_pnl": long_row["pnl"] + short_row["pnl"],
            })

overlap_df = pd.DataFrame(overlaps)
if len(overlap_df) == 0:
    print("\nNo LONG+SHORT overlaps found.")
else:
    unique_pairs = overlap_df.drop_duplicates(subset=["long_idx", "short_idx"])
    n_unique_pairs = len(unique_pairs)
    print(f"\n1. OVERLAP COUNT")
    print(f"   Total overlap instances (unique LONG-SHORT pairs): {n_unique_pairs}")

    durations = unique_pairs["overlap_hours"]
    print(f"\n2. OVERLAP DURATION (hours)")
    print(f"   Mean: {durations.mean():.2f} h  Median: {durations.median():.2f} h")
    print(f"   Min: {durations.min():.2f} h  Max: {durations.max():.2f} h  Std: {durations.std():.2f} h")

    pairs = unique_pairs
    long_pnls = pairs["long_pnl"].values
    short_pnls = pairs["short_pnl"].values
    combined = pairs["combined_pnl"].values
    same_sign = (long_pnls > 0) == (short_pnls > 0)
    opposite_sign = ~same_sign
    n_same = same_sign.sum()
    n_opposite = opposite_sign.sum()
    print(f"\n3. PnL DURING OVERLAPS - HEDGE vs COMPOUND")
    print(f"   Same sign (compound): {n_same}  Opposite sign (hedge): {n_opposite}")
    print(f"   Hedge ratio: {100*n_opposite/n_unique_pairs:.1f}%")
    print(f"   Mean LONG PnL: {long_pnls.mean():.2f}  Mean SHORT PnL: {short_pnls.mean():.2f}")
    print(f"   Mean COMBINED PnL: {combined.mean():.2f}")
    denom = (abs(long_pnls).mean() + abs(short_pnls).mean())
    if denom > 0:
        print(f"   PnL magnitude reduction (hedging): {100*(1 - abs(combined).mean()/denom):.1f}%")

    n_ex = min(10, len(pairs))
    examples = pairs.nlargest(n_ex, "overlap_hours")
    print(f"\n4. EXAMPLES (top {n_ex} by overlap duration)")
    print("-" * 70)
    for idx, (_, row) in enumerate(examples.iterrows(), 1):
        print(f"\n--- Example {idx} ---")
        print(f"  LONG:  {row['long_entry']} -> {row['long_exit']}  PnL: {row['long_pnl']:,.0f}")
        print(f"  SHORT: {row['short_entry']} -> {row['short_exit']}  PnL: {row['short_pnl']:,.0f}")
        print(f"  Overlap: {row['overlap_start']} to {row['overlap_end']} ({row['overlap_hours']:.1f}h)")
        print(f"  Combined PnL: {row['combined_pnl']:,.0f}")
