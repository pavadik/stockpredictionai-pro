"""ML Trading Strategy: TFT + Momentum on SBER H1.

Walk-forward approach:
  1. Train TFT d=64 h=4 on rolling windows
  2. Generate direction predictions on out-of-sample bars
  3. Combine with momentum signal
  4. Simulate trades with signal filtering, ATR thresholds
  5. Compute equity curve, Sharpe, MaxDD, yearly breakdown

Usage:
    python scripts/run_ml_strategy.py
    python scripts/run_ml_strategy.py --mode momentum_only
    python scripts/run_ml_strategy.py --mode tft_only
    python scripts/run_ml_strategy.py --mode combined
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data import build_panel_auto
from src.dataset import make_sequences_multi, walk_forward_splits
from src.train import (
    build_features_safe, fit_transforms, run_one_split,
    set_global_seed, _log_device_info,
)
from src.utils.metrics import direction_accuracy

SEED = 42
OUT_DIR = "outputs/strategy"


# -----------------------------------------------------------------------
# Trade record
# -----------------------------------------------------------------------
@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    direction: str
    pnl_pct: float        # PnL as % of entry price
    pnl_pts: float        # PnL in price points
    exit_reason: str
    signal_strength: float = 0.0


# -----------------------------------------------------------------------
# ATR computation
# -----------------------------------------------------------------------
def compute_atr(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Simple ATR proxy from close-to-close changes (no OHLC available)."""
    atr = np.zeros(len(closes))
    abs_diff = np.abs(np.diff(closes, prepend=closes[0]))
    for i in range(len(closes)):
        start = max(0, i - period + 1)
        atr[i] = abs_diff[start:i + 1].mean()
    return atr


# -----------------------------------------------------------------------
# Signal generation
# -----------------------------------------------------------------------
def generate_signals_momentum(closes: np.ndarray, atr: np.ndarray,
                              horizon: int = 4,
                              atr_threshold: float = 0.3) -> np.ndarray:
    """Momentum signal with ATR filter.

    Signal = sign(close[i] - close[i - horizon]) if |delta| > atr_threshold * ATR.
    """
    signals = np.zeros(len(closes))
    for i in range(horizon, len(closes)):
        delta = closes[i] - closes[i - horizon]
        if atr[i] > 0 and abs(delta) > atr_threshold * atr[i]:
            signals[i] = np.sign(delta)
    return signals


def generate_signals_tft(y_pred: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """TFT model signals from predicted deltas with optional threshold."""
    pred = y_pred.flatten()
    sigs = np.zeros(len(pred))
    mask = np.abs(pred) > threshold
    sigs[mask] = np.sign(pred[mask])
    return sigs


def combine_signals(sig_tft: np.ndarray, sig_mom: np.ndarray,
                    mode: str = "combined",
                    w_mom: float = 0.7, w_tft: float = 0.3) -> np.ndarray:
    """Combine TFT and momentum signals.

    Modes:
      - momentum_only: only momentum
      - tft_only: only TFT
      - combined: both must agree, else 0
      - weighted: w_mom*Momentum + w_tft*TFT
    """
    if mode == "momentum_only":
        return sig_mom.copy()
    if mode == "tft_only":
        return sig_tft.copy()
    if mode == "combined":
        agree = (np.sign(sig_tft) == np.sign(sig_mom)) & (sig_tft != 0) & (sig_mom != 0)
        out = np.zeros_like(sig_mom)
        out[agree] = sig_mom[agree]
        return out
    if mode == "weighted":
        w = w_mom * sig_mom + w_tft * sig_tft
        return np.sign(w) * (np.abs(w) > 0.3).astype(float)

    # Long-only variants: only take long signals, flat otherwise
    if mode == "long_only_mom":
        return np.where(sig_mom > 0, 1.0, 0.0)
    if mode == "long_only_tft":
        return np.where(sig_tft > 0, 1.0, 0.0)
    if mode == "long_only_combined":
        agree_long = (sig_tft > 0) & (sig_mom > 0)
        return np.where(agree_long, 1.0, 0.0)

    return sig_mom.copy()


# -----------------------------------------------------------------------
# Trade simulation
# -----------------------------------------------------------------------
def simulate_trades(
    closes: np.ndarray,
    times: np.ndarray,
    signals: np.ndarray,
    atr: np.ndarray,
    hold_bars: int = 4,
    commission_pct: float = 0.0002,
    max_hold_mult: float = 2.0,
    trailing_atr_mult: float = 0.0,
) -> List[Trade]:
    """Bar-by-bar trade simulation on H1 bars.

    Rules:
      - Enter on signal != 0 when flat (at the bar's close price)
      - Hold for hold_bars, then exit
      - Exit early on signal reversal or trailing stop
      - No pyramiding
      - Commission: pct of trade value on entry + exit
    """
    trades = []
    n = len(closes)
    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_time = ""
    entry_signal = 0.0
    best_price = 0.0

    for i in range(n):
        sig = signals[i]

        if position != 0:
            bars_held = i - entry_bar

            # Track best price for trailing stop
            if position > 0:
                best_price = max(best_price, closes[i])
            else:
                best_price = min(best_price, closes[i])

            should_exit = False
            exit_reason = ""

            # Mandatory exit after hold_bars
            max_hold = int(hold_bars * max_hold_mult)
            if bars_held >= hold_bars:
                should_exit = True
                exit_reason = "hold_expired"

            # Trailing stop (if configured)
            if trailing_atr_mult > 0 and atr[i] > 0 and not should_exit:
                if position > 0:
                    retrace = best_price - closes[i]
                else:
                    retrace = closes[i] - best_price
                if retrace > trailing_atr_mult * atr[entry_bar]:
                    should_exit = True
                    exit_reason = "trailing_stop"

            # Signal reversal
            if sig != 0 and sig != position and not should_exit:
                should_exit = True
                exit_reason = "signal_reverse"

            if should_exit:
                exit_price = closes[i]
                raw_pnl = position * (exit_price - entry_price)
                cost = commission_pct * (entry_price + exit_price)
                net_pnl = raw_pnl - cost
                pnl_pct = net_pnl / entry_price * 100 if entry_price > 0 else 0

                trades.append(Trade(
                    entry_bar=entry_bar, exit_bar=i,
                    entry_time=str(times[entry_bar]),
                    exit_time=str(times[i]),
                    entry_price=entry_price, exit_price=exit_price,
                    direction="long" if position > 0 else "short",
                    pnl_pct=pnl_pct, pnl_pts=net_pnl,
                    exit_reason=exit_reason,
                    signal_strength=entry_signal,
                ))
                position = 0

        # Open new position
        if position == 0 and sig != 0:
            position = int(sig)
            entry_price = closes[i]
            entry_bar = i
            entry_time = str(times[i])
            entry_signal = sig
            best_price = entry_price

    if position != 0:
        exit_price = closes[-1]
        raw_pnl = position * (exit_price - entry_price)
        cost = commission_pct * (entry_price + exit_price)
        net_pnl = raw_pnl - cost
        pnl_pct = net_pnl / entry_price * 100 if entry_price > 0 else 0
        trades.append(Trade(
            entry_bar=entry_bar, exit_bar=n - 1,
            entry_time=str(times[entry_bar]),
            exit_time=str(times[-1]),
            entry_price=entry_price, exit_price=exit_price,
            direction="long" if position > 0 else "short",
            pnl_pct=pnl_pct, pnl_pts=net_pnl,
            exit_reason="end_of_data", signal_strength=entry_signal,
        ))

    return trades


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------
def compute_strategy_metrics(trades: List[Trade], annualize_n: int = 252) -> dict:
    if not trades:
        return {k: 0 for k in [
            "num_trades", "total_pnl_pct", "total_pnl_pts", "avg_trade_pct",
            "win_rate", "profit_factor", "max_drawdown_pct", "sharpe",
            "long_trades", "short_trades", "long_pnl_pct", "short_pnl_pct",
            "avg_bars_held", "max_consecutive_loss",
        ]}

    pnls_pct = np.array([t.pnl_pct for t in trades])
    pnls_pts = np.array([t.pnl_pts for t in trades])
    n = len(pnls_pct)
    total_pct = float(pnls_pct.sum())
    total_pts = float(pnls_pts.sum())

    wins = pnls_pct[pnls_pct > 0]
    losses = pnls_pct[pnls_pct < 0]
    wr = len(wins) / n if n > 0 else 0
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else float("inf")

    cum = np.cumsum(pnls_pct)
    rmax = np.maximum.accumulate(cum)
    mdd = float((rmax - cum).max()) if n else 0.0

    sharpe = 0.0
    if n > 1 and pnls_pct.std() > 0:
        sharpe = float((pnls_pct.mean() / pnls_pct.std()) * np.sqrt(min(annualize_n, n)))

    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]

    bars_held = [t.exit_bar - t.entry_bar for t in trades]

    # Max consecutive losses
    max_cl = 0
    cl = 0
    for p in pnls_pct:
        if p < 0:
            cl += 1
            max_cl = max(max_cl, cl)
        else:
            cl = 0

    return {
        "num_trades": n,
        "total_pnl_pct": round(total_pct, 4),
        "total_pnl_pts": round(total_pts, 4),
        "avg_trade_pct": round(total_pct / n, 6),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(mdd, 4),
        "sharpe": round(sharpe, 3),
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "long_pnl_pct": round(sum(t.pnl_pct for t in longs), 4),
        "short_pnl_pct": round(sum(t.pnl_pct for t in shorts), 4),
        "avg_bars_held": round(np.mean(bars_held), 1) if bars_held else 0,
        "max_consecutive_loss": max_cl,
    }


def yearly_breakdown(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    df = pd.DataFrame([{
        "year": t.entry_time[:4], "pnl_pct": t.pnl_pct,
        "pnl_pts": t.pnl_pts, "direction": t.direction,
    } for t in trades])
    for year, grp in df.groupby("year"):
        pnls = grp["pnl_pct"].values
        pts = grp["pnl_pts"].values
        n = len(pnls)
        rows.append({
            "year": year, "trades": n,
            "total_pnl_pct": round(pnls.sum(), 4),
            "total_pnl_pts": round(pts.sum(), 4),
            "avg_trade_pct": round(pnls.mean(), 6),
            "win_rate": round((pnls > 0).mean(), 4),
            "profit_factor": round(
                pnls[pnls > 0].sum() / max(abs(pnls[pnls < 0].sum()), 1e-9), 3
            ),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Align TFT predictions to bar index
# -----------------------------------------------------------------------
def align_tft_to_bars(y_pred: np.ndarray, feat_test_idx, te_panel_idx,
                      seq_len: int, max_h: int,
                      n_bars: int, threshold: float = 0.0) -> np.ndarray:
    """Map TFT predictions back to test panel bar positions."""
    offset_in_feat = seq_len + max_h - 1
    n_preds = len(y_pred)
    sig = np.zeros(n_bars)

    for j in range(n_preds):
        feat_pos = offset_in_feat + j
        if feat_pos >= len(feat_test_idx):
            break
        pred_ts = feat_test_idx[feat_pos]
        if pred_ts in te_panel_idx:
            pos = te_panel_idx.get_loc(pred_ts)
            if isinstance(pos, int) and 0 <= pos < n_bars:
                val = y_pred[j] if y_pred.ndim == 1 else y_pred[j, 0]
                if abs(val) > threshold:
                    sig[pos] = np.sign(val)
    return sig


# -----------------------------------------------------------------------
# Walk-forward ML strategy runner
# -----------------------------------------------------------------------
def run_strategy(args, mode="combined"):
    """Run the full walk-forward ML strategy."""
    print("=" * 70)
    print(f"  ML Strategy: {args.ticker} H1 h={args.hold_bars} | mode={mode}")
    print(f"  ATR threshold: {args.atr_threshold}")
    print("=" * 70)

    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end,
        test_years=args.test_years, generator="lstm",
        data_source="local", timeframe="H1",
        data_path=args.data_path, seq_len=12,
        model_type="tft", loss_fn="huber",
    )
    cfg.d_model = 64
    cfg.nhead = 4
    cfg.num_layers = 2
    cfg.lr_g = 2e-3
    cfg.lr_d = 2e-4
    cfg.l1_weight = 0.4
    cfg.cls_weight = 0.2
    cfg.q_weight = 1.0
    cfg.forecast_horizons = (args.hold_bars,)
    cfg.use_delta_lags = True
    cfg.delta_lag_periods = (1,)
    cfg.n_epochs = 20
    cfg.early_stopping_patience = 5
    cfg.apply_timeframe_defaults()

    print(f"\nLoading H1 panel...")
    t0 = time.time()
    panel = build_panel_auto(cfg)
    print(f"Panel: {panel.shape}, loaded in {time.time()-t0:.1f}s")
    _log_device_info()

    closes_full = panel[cfg.ticker].values
    times_full = panel.index

    atr_full = compute_atr(closes_full, period=14)

    n_splits = args.n_splits
    wf_min = cfg.wf_min_train
    wf_step = cfg.wf_step
    splits = list(walk_forward_splits(
        panel, n_splits=n_splits, min_train=wf_min, step=wf_step,
    ))
    print(f"Walk-forward: {len(splits)} splits, min_train={wf_min}, step={wf_step}")

    # ------------------------------------------------------------------
    # --start_split: resume from split N, loading previously saved data
    # ------------------------------------------------------------------
    start_split = getattr(args, 'start_split', 1)
    all_trades: List[Trade] = []
    split_metrics = []

    if start_split > 1:
        prev_trades_path = os.path.join(OUT_DIR, f"strategy_{mode}_trades.csv")
        prev_splits_path = os.path.join(OUT_DIR, f"strategy_{mode}_splits.csv")
        if os.path.exists(prev_trades_path):
            prev_df = pd.read_csv(prev_trades_path)
            for _, row in prev_df.iterrows():
                all_trades.append(Trade(
                    entry_bar=0, exit_bar=0,
                    entry_time=str(row["entry_time"]),
                    exit_time=str(row["exit_time"]),
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    direction=str(row["direction"]),
                    pnl_pct=float(row["pnl_pct"]),
                    pnl_pts=float(row["pnl_pts"]),
                    exit_reason=str(row["exit_reason"]),
                    signal_strength=float(row.get("signal_strength", 0)),
                ))
            print(f"  Loaded {len(all_trades)} trades from previous run (splits 1-{start_split-1})")
        if os.path.exists(prev_splits_path):
            prev_sm = pd.read_csv(prev_splits_path).to_dict("records")
            split_metrics.extend(prev_sm)
            print(f"  Loaded {len(prev_sm)} split metrics from previous run")

    for i, (tr_idx, te_idx) in enumerate(splits, 1):
        if i < start_split:
            continue
        set_global_seed(SEED)
        tr_panel = panel.iloc[tr_idx]
        te_panel = panel.iloc[te_idx]
        te_closes = te_panel[cfg.ticker].values
        te_times = te_panel.index.astype(str).values
        te_atr = atr_full[te_idx]

        print(f"\n{'='*60}")
        print(f"  Split {i}/{len(splits)}: train={len(tr_panel)} test={len(te_panel)}")
        print(f"  Test: {te_times[0]} -> {te_times[-1]}")
        print(f"  Price range: {te_closes.min():.2f} - {te_closes.max():.2f}")
        print(f"{'='*60}")

        # Momentum signal (always available)
        sig_mom = generate_signals_momentum(
            te_closes, te_atr,
            horizon=args.hold_bars,
            atr_threshold=args.atr_threshold,
        )
        n_mom = (sig_mom != 0).sum()
        print(f"  Momentum signals: {n_mom}/{len(sig_mom)} "
              f"({n_mom/len(sig_mom)*100:.1f}%)")

        # TFT signal (if needed)
        sig_tft_full = np.zeros(len(te_closes))
        tft_dirac = float("nan")

        needs_tft = mode in ("tft_only", "combined", "weighted",
                          "long_only_tft", "long_only_combined")
        if needs_tft:
            t1 = time.time()
            feat_train, feat_test = build_features_safe(tr_panel, te_panel, cfg)
            feat_train, feat_test = fit_transforms(feat_train, feat_test, cfg)

            met, yte, y_pred, _, _, model = run_one_split(
                feat_train, feat_test, cfg, verbose=True)

            tft_time = time.time() - t1
            tft_dirac = met.get("DirAcc", float("nan"))
            print(f"  TFT DirAcc={tft_dirac:.3f} (trained in {tft_time:.1f}s)")

            sig_tft_full = align_tft_to_bars(
                y_pred, feat_test.index, te_panel.index,
                cfg.seq_len, max(cfg.forecast_horizons),
                len(te_closes),
            )
            n_tft = (sig_tft_full != 0).sum()
            print(f"  TFT signals: {n_tft}/{len(sig_tft_full)} "
                  f"({n_tft/len(sig_tft_full)*100:.1f}%)")

        # Combine signals
        final_signals = combine_signals(sig_tft_full, sig_mom, mode=mode)
        n_final = (final_signals != 0).sum()
        print(f"  Final signals ({mode}): {n_final}/{len(final_signals)} "
              f"({n_final/len(final_signals)*100:.1f}%)")

        # Simulate trades
        trades = simulate_trades(
            te_closes, te_times, final_signals, te_atr,
            hold_bars=args.hold_bars,
            commission_pct=args.commission,
            trailing_atr_mult=args.trailing_atr,
        )

        sm = compute_strategy_metrics(trades)
        sm["split"] = i
        sm["test_start"] = te_times[0]
        sm["test_end"] = te_times[-1]
        sm["bars"] = len(te_closes)
        sm["tft_dirac"] = tft_dirac
        sm["signal_coverage"] = round(n_final / len(final_signals), 4)

        print(f"\n  Split {i}: {sm['num_trades']} trades, "
              f"PnL={sm['total_pnl_pct']:.2f}%, WR={sm['win_rate']:.1%}, "
              f"Sharpe={sm['sharpe']:.2f}, MaxDD={sm['max_drawdown_pct']:.2f}%")
        print(f"    Long: {sm['long_trades']} ({sm['long_pnl_pct']:.2f}%), "
              f"Short: {sm['short_trades']} ({sm['short_pnl_pct']:.2f}%)")

        split_metrics.append(sm)
        all_trades.extend(trades)

    return _save_results(args, mode, all_trades, split_metrics)


# -----------------------------------------------------------------------
# Save all results
# -----------------------------------------------------------------------
def _save_results(args, mode, all_trades, split_metrics):
    os.makedirs(OUT_DIR, exist_ok=True)

    overall = compute_strategy_metrics(all_trades)
    yearly = yearly_breakdown(all_trades)

    print(f"\n{'='*70}")
    print(f"  STRATEGY RESULTS: {args.ticker} H1 h={args.hold_bars} | mode={mode}")
    print(f"{'='*70}")
    print(f"  Total trades:          {overall['num_trades']}")
    print(f"  Total PnL:             {overall['total_pnl_pct']:.2f}% "
          f"({overall['total_pnl_pts']:.2f} pts)")
    print(f"  Avg trade:             {overall['avg_trade_pct']:.4f}%")
    print(f"  Win rate:              {overall['win_rate']:.1%}")
    print(f"  Profit factor:         {overall['profit_factor']:.3f}")
    print(f"  Sharpe ratio:          {overall['sharpe']:.3f}")
    print(f"  Max drawdown:          {overall['max_drawdown_pct']:.2f}%")
    print(f"  Avg bars held:         {overall['avg_bars_held']}")
    print(f"  Max consec. losses:    {overall['max_consecutive_loss']}")
    print(f"  Long:  {overall['long_trades']} trades ({overall['long_pnl_pct']:.2f}%)")
    print(f"  Short: {overall['short_trades']} trades ({overall['short_pnl_pct']:.2f}%)")

    if not yearly.empty:
        print(f"\n  YEARLY BREAKDOWN:")
        print(f"  {'Year':<6} {'Trades':>7} {'PnL%':>9} {'PnL pts':>10} "
              f"{'AvgTrade%':>10} {'WR':>6} {'PF':>6}")
        print(f"  {'-'*55}")
        for _, row in yearly.iterrows():
            print(f"  {row['year']:<6} {row['trades']:>7} "
                  f"{row['total_pnl_pct']:>8.2f}% {row['total_pnl_pts']:>10.2f} "
                  f"{row['avg_trade_pct']:>9.4f}% "
                  f"{row['win_rate']:>5.1%} {row['profit_factor']:>6.2f}")

    # Save trades
    trades_path = os.path.join(OUT_DIR, f"strategy_{mode}_trades.csv")
    if all_trades:
        trades_df = pd.DataFrame([{
            "entry_time": t.entry_time, "exit_time": t.exit_time,
            "entry_price": t.entry_price, "exit_price": t.exit_price,
            "direction": t.direction, "pnl_pct": t.pnl_pct,
            "pnl_pts": t.pnl_pts, "exit_reason": t.exit_reason,
            "signal_strength": t.signal_strength,
        } for t in all_trades])
        trades_df.to_csv(trades_path, index=False)

    # Save equity curve
    equity_path = os.path.join(OUT_DIR, f"strategy_{mode}_equity.csv")
    if all_trades:
        pnls = np.array([t.pnl_pct for t in all_trades])
        eq = np.cumsum(pnls)
        eq_df = pd.DataFrame({
            "trade_num": range(1, len(eq) + 1),
            "pnl_pct": pnls,
            "equity_pct": eq,
            "exit_time": [t.exit_time for t in all_trades],
            "direction": [t.direction for t in all_trades],
        })
        eq_df.to_csv(equity_path, index=False)

    # Save split metrics
    splits_path = os.path.join(OUT_DIR, f"strategy_{mode}_splits.csv")
    pd.DataFrame(split_metrics).to_csv(splits_path, index=False)

    if not yearly.empty:
        yearly.to_csv(os.path.join(OUT_DIR, f"strategy_{mode}_yearly.csv"), index=False)

    # Equity chart
    _plot_equity(all_trades, args, mode, overall, OUT_DIR)

    print(f"\n  Files saved to {OUT_DIR}/")
    return overall, all_trades


def _plot_equity(all_trades, args, mode, overall, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if not all_trades:
            return

        pnls = np.array([t.pnl_pct for t in all_trades])
        eq = np.cumsum(pnls)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                 gridspec_kw={"height_ratios": [3, 1, 1]})

        # Equity curve
        axes[0].plot(eq, linewidth=1.2, color="#2196F3")
        axes[0].fill_between(range(len(eq)), eq, alpha=0.12, color="#2196F3")
        axes[0].set_title(
            f"Equity Curve: {args.ticker} H1 h={args.hold_bars} | {mode}\n"
            f"Trades={len(pnls)}, PnL={pnls.sum():.2f}%, "
            f"Sharpe={overall['sharpe']:.2f}, MaxDD={overall['max_drawdown_pct']:.2f}%",
            fontsize=12)
        axes[0].set_ylabel("Cumulative PnL (%)")
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Drawdown
        rmax = np.maximum.accumulate(eq)
        dd = rmax - eq
        axes[1].fill_between(range(len(dd)), -dd, alpha=0.5, color="#F44336")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True, alpha=0.3)

        # Trade PnL distribution
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        axes[2].bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=1.0)
        axes[2].set_ylabel("Trade PnL (%)")
        axes[2].set_xlabel("Trade #")
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(out_dir, f"strategy_{mode}_equity.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"  Equity chart: {chart_path}")
    except Exception as e:
        print(f"  (Chart skipped: {e})")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ML Trading Strategy")
    parser.add_argument("--ticker", default="SBER")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2021-12-31")
    parser.add_argument("--test_years", type=int, default=1)
    parser.add_argument("--data_path", default=r"G:\data2")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--hold_bars", type=int, default=4,
                        help="Hold period in bars (= forecast horizon)")
    parser.add_argument("--commission", type=float, default=0.0002,
                        help="Commission as fraction of price (default 0.02%%)")
    parser.add_argument("--atr_threshold", type=float, default=0.3,
                        help="Min |delta|/ATR to generate momentum signal (default 0.3)")
    parser.add_argument("--trailing_atr", type=float, default=0.0,
                        help="Trailing stop in ATR multiples (0 = disabled)")
    parser.add_argument("--mode", nargs="*",
                        default=["momentum_only", "tft_only", "combined"],
                        choices=["momentum_only", "tft_only", "combined", "weighted",
                                 "long_only_mom", "long_only_tft", "long_only_combined"],
                        help="Signal mode(s) to test")
    parser.add_argument("--start_split", type=int, default=1,
                        help="Resume from this split number (loads previous trades from CSV)")
    args = parser.parse_args()

    results_all = []
    for mode in args.mode:
        overall, trades = run_strategy(args, mode=mode)
        overall["mode"] = mode
        results_all.append(overall)

    if len(results_all) > 1:
        print(f"\n{'='*70}")
        print(f"  COMPARISON ACROSS MODES")
        print(f"{'='*70}")
        hdr = (f"  {'Mode':<18} {'Trades':>7} {'PnL%':>8} {'AvgTrd%':>9} "
               f"{'WR':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>8}")
        print(hdr)
        print(f"  {'-'*72}")
        for r in results_all:
            print(f"  {r['mode']:<18} {r['num_trades']:>7} "
                  f"{r['total_pnl_pct']:>7.2f}% {r['avg_trade_pct']:>8.4f}% "
                  f"{r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} "
                  f"{r['sharpe']:>7.2f} {r['max_drawdown_pct']:>7.2f}%")

    summary_path = os.path.join(OUT_DIR, "strategy_comparison.csv")
    pd.DataFrame(results_all).to_csv(summary_path, index=False)
    print(f"\n  Comparison saved: {summary_path}")


if __name__ == "__main__":
    main()
