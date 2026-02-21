"""Backtester for the momentum-trend strategy.

Simulates trades bar-by-bar, tracks positions, records round-trip
trades, and computes standard performance metrics.

Includes a **Numba-JIT compiled** inner loop
(:func:`_simulate_trades_numba`) for the fast optimisation path,
and the original pure-Python :func:`simulate_trades` that records
full :class:`Trade` objects for reporting.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# -----------------------------------------------------------------------
# Numba-JIT compiled inner loop (returns PnL array only)
# -----------------------------------------------------------------------

def _simulate_numba_inner(
    closes: np.ndarray,
    entry_flags: np.ndarray,
    allow: np.ndarray,
    in_window: np.ndarray,
    lot_sizes: np.ndarray,
    dates_ord: np.ndarray,
    exitday: int,
    sdel_day: int,
) -> np.ndarray:
    """Pure-numpy backtest loop (Numba-JIT compiled when available).

    Returns a 1-D float64 array of per-trade PnL values.
    """
    max_trades = len(closes)
    pnls = np.empty(max_trades, dtype=np.float64)
    n_trades = 0

    position = 0
    entry_price = 0.0
    one_entry_today = False
    current_date = -1

    for i in range(len(closes)):
        d = dates_ord[i]
        if d != current_date:
            current_date = d
            one_entry_today = False

        if position > 0 and not allow[i]:
            pnls[n_trades] = (closes[i] - entry_price) * position
            n_trades += 1
            position = 0
            continue

        if allow[i] and in_window[i]:
            if position == 0:
                if entry_flags[i]:
                    if sdel_day == 1 and one_entry_today:
                        continue
                    lots = lot_sizes[i]
                    position = lots if lots > 0 else 1
                    entry_price = closes[i]
                    one_entry_today = True
            else:
                if not entry_flags[i]:
                    pnls[n_trades] = (closes[i] - entry_price) * position
                    n_trades += 1
                    position = 0
        elif position > 0 and exitday == 1:
            pnls[n_trades] = (closes[i] - entry_price) * position
            n_trades += 1
            position = 0

    if position > 0:
        pnls[n_trades] = (closes[-1] - entry_price) * position
        n_trades += 1

    return pnls[:n_trades]


# Apply Numba JIT if available
if _HAS_NUMBA:
    _simulate_numba_inner = njit(cache=True)(_simulate_numba_inner)


def simulate_trades_fast(df: pd.DataFrame,
                         params) -> np.ndarray:
    """Numba-accelerated backtest that returns **only PnL array**.

    ~10-50x faster than :func:`simulate_trades` because it avoids
    Python object creation per trade.  Use for optimisation loops.
    """
    closes = df["close"].values.astype(np.float64)
    entry_flags = df["entry_long"].values.astype(np.bool_)
    allow = df["allow_trade"].values.astype(np.bool_)
    in_window = df["in_time_window"].values.astype(np.bool_)
    lot_sizes = df["contracts"].values.astype(np.int64)

    # Convert dates to ordinal integers for fast comparison
    dates_ord = np.array([d.toordinal() for d in df.index.date],
                         dtype=np.int64)

    return _simulate_numba_inner(
        closes, entry_flags, allow, in_window, lot_sizes,
        dates_ord, params.exitday, params.sdel_day,
    )


def compute_metrics_from_pnl(pnls: np.ndarray) -> Dict[str, float]:
    """Compute metrics from a raw PnL array (fast path)."""
    n = len(pnls)
    if n == 0:
        return {
            "avg_trade": 0.0, "num_trades": 0, "total_pnl": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "max_drawdown": 0.0, "sharpe": 0.0,
        }
    total = float(pnls.sum())
    avg = total / n
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    wr = len(wins) / n
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    cum = np.cumsum(pnls)
    rmax = np.maximum.accumulate(cum)
    mdd = float((rmax - cum).max()) if n else 0.0
    sharpe = 0.0
    if n > 1 and pnls.std() > 0:
        sharpe = float((pnls.mean() / pnls.std()) * np.sqrt(min(250, n)))
    return {
        "avg_trade": avg, "num_trades": n, "total_pnl": total,
        "win_rate": wr, "profit_factor": pf,
        "max_drawdown": mdd, "sharpe": sharpe,
    }


# -----------------------------------------------------------------------
# Trade record
# -----------------------------------------------------------------------

@dataclass
class Trade:
    """One completed round-trip (entry + exit)."""

    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    contracts: int
    pnl: float
    exit_reason: str  # "signal_reverse", "end_of_day", "expiration", "flip"
    direction: str = "long"  # "long" or "short"


# -----------------------------------------------------------------------
# Trade simulation (full, with Trade objects for reporting)
# -----------------------------------------------------------------------

def _check_stop_loss(sl_type: str, pnl_sign: float, close: float,
                     entry_price: float, best_price: float,
                     atr_at_entry: float, bars_in_trade: int,
                     sl_pts: float, sl_atr_mult: float,
                     trail_atr: float, trail_pts: float,
                     be_target: float, be_offset: float,
                     time_bars: int, time_target: float) -> bool:
    """Return True if stop-loss is triggered on this bar."""
    if sl_type == "none":
        return False

    unrealised = pnl_sign * (close - entry_price)

    if sl_type == "fixed":
        return unrealised < -sl_pts

    if sl_type == "atr":
        return unrealised < -sl_atr_mult * atr_at_entry

    if sl_type == "trail_atr":
        retrace = pnl_sign * (best_price - close)
        return retrace > trail_atr * atr_at_entry

    if sl_type == "trail_pts":
        retrace = pnl_sign * (best_price - close)
        return retrace > trail_pts

    if sl_type == "breakeven":
        if unrealised >= be_target:
            stop_level = entry_price + pnl_sign * be_offset
            return pnl_sign * (close - stop_level) < 0
        return False

    if sl_type == "time":
        if bars_in_trade >= time_bars and unrealised < time_target:
            return True
        return False

    return False


def simulate_trades(df: pd.DataFrame,
                    params) -> List[Trade]:
    """Walk through bars sequentially and track a directional position.

    Supports both long and short via ``params.direction``,
    and all stop-loss types via ``params.sl_type``.

    Returns:
        List of completed :class:`Trade` objects.
    """
    trades: List[Trade] = []
    position = 0
    entry_price = 0.0
    entry_time: datetime = datetime.min
    one_entry_today = False
    current_date = None
    is_short = getattr(params, "direction", "long") == "short"
    pnl_sign = -1.0 if is_short else 1.0

    best_price = 0.0
    atr_at_entry = 0.0
    bars_in_trade = 0

    sl_type = getattr(params, "sl_type", "none")
    sl_pts = getattr(params, "sl_pts", 0.0)
    sl_atr_mult = getattr(params, "sl_atr_mult", 0.0)
    trail_atr = getattr(params, "trail_atr", 0.0)
    trail_pts = getattr(params, "trail_pts", 0.0)
    be_target = getattr(params, "be_target", 0.0)
    be_offset = getattr(params, "be_offset", 0.0)
    time_bars = getattr(params, "time_bars", 0)
    time_target = getattr(params, "time_target", 0.0)

    idx = df.index
    opens = df["open"].values
    closes = df["close"].values
    entry_flags = df["entry_long"].values
    exit_flags = df["exit_long"].values
    allow = df["allow_trade"].values
    in_window = df["in_time_window"].values
    lot_sizes = df["contracts"].values
    atr_arr = df["atr"].values if "atr" in df.columns else np.zeros(len(df))
    is_week_end = (df["is_week_end"].values
                   if "is_week_end" in df.columns
                   else np.zeros(len(df), dtype=bool))

    exitday = params.exitday
    sdel_day = params.sdel_day
    exit_week = getattr(params, "exit_week", 0)

    for i in range(len(df)):
        bar_date = idx[i].date()

        if bar_date != current_date:
            current_date = bar_date
            one_entry_today = False

        # --- Forced exit: expiration week --------------------------------
        if position > 0 and not allow[i]:
            pnl = pnl_sign * (closes[i] - entry_price) * position
            trades.append(Trade(
                entry_time=entry_time,
                exit_time=idx[i].to_pydatetime(),
                entry_price=entry_price,
                exit_price=closes[i],
                contracts=position,
                pnl=pnl,
                exit_reason="expiration",
            ))
            position = 0
            continue

        # --- Stop-loss check (before signal/time exits) ------------------
        if position > 0:
            bars_in_trade += 1
            if is_short:
                best_price = min(best_price, closes[i])
            else:
                best_price = max(best_price, closes[i])

            if _check_stop_loss(
                sl_type, pnl_sign, closes[i], entry_price, best_price,
                atr_at_entry, bars_in_trade,
                sl_pts, sl_atr_mult, trail_atr, trail_pts,
                be_target, be_offset, time_bars, time_target,
            ):
                pnl = pnl_sign * (closes[i] - entry_price) * position
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=idx[i].to_pydatetime(),
                    entry_price=entry_price,
                    exit_price=closes[i],
                    contracts=position,
                    pnl=pnl,
                    exit_reason="stop_loss",
                ))
                position = 0
                continue

        # --- Inside trading window ---------------------------------------
        if allow[i] and in_window[i]:
            if position == 0:
                if entry_flags[i]:
                    if sdel_day == 1 and one_entry_today:
                        continue
                    position = max(int(lot_sizes[i]), 1)
                    entry_price = closes[i]
                    entry_time = idx[i].to_pydatetime()
                    one_entry_today = True
                    best_price = closes[i]
                    atr_at_entry = float(atr_arr[i]) if atr_arr[i] > 0 else 1.0
                    bars_in_trade = 0
            else:
                if not entry_flags[i]:
                    pnl = pnl_sign * (closes[i] - entry_price) * position
                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=idx[i].to_pydatetime(),
                        entry_price=entry_price,
                        exit_price=closes[i],
                        contracts=position,
                        pnl=pnl,
                        exit_reason="signal_reverse",
                    ))
                    position = 0

        # --- Outside window: end-of-day exit -----------------------------
        elif position > 0 and exitday == 1:
            pnl = pnl_sign * (closes[i] - entry_price) * position
            trades.append(Trade(
                entry_time=entry_time,
                exit_time=idx[i].to_pydatetime(),
                entry_price=entry_price,
                exit_price=closes[i],
                contracts=position,
                pnl=pnl,
                exit_reason="end_of_day",
            ))
            position = 0

        # --- End-of-week exit -------------------------------------------
        if position > 0 and exit_week == 1 and is_week_end[i]:
            pnl = pnl_sign * (closes[i] - entry_price) * position
            trades.append(Trade(
                entry_time=entry_time,
                exit_time=idx[i].to_pydatetime(),
                entry_price=entry_price,
                exit_price=closes[i],
                contracts=position,
                pnl=pnl,
                exit_reason="end_of_week",
            ))
            position = 0

    if position > 0:
        last = len(df) - 1
        pnl = pnl_sign * (closes[last] - entry_price) * position
        trades.append(Trade(
            entry_time=entry_time,
            exit_time=idx[last].to_pydatetime(),
            entry_price=entry_price,
            exit_price=closes[last],
            contracts=position,
            pnl=pnl,
            exit_reason="end_of_data",
        ))

    return trades


# -----------------------------------------------------------------------
# Combined LONG+SHORT simulation with flip logic
# -----------------------------------------------------------------------

def simulate_combined_trades(
    df_long: pd.DataFrame,
    df_short: pd.DataFrame,
    params_long,
    params_short,
    max_contracts: int = 0,
    max_contracts_long: int = 0,
    max_contracts_short: int = 0,
    flip_mode: str = "flip",
    leverage: float = 1.0,
) -> List[Trade]:
    """Run a unified LONG+SHORT simulation on merged bar timelines.

    Both DataFrames must already have signals computed (via
    ``generate_signals``).  The function builds a single event
    timeline from both grids, maintains **one position** at a time,
    and *flips* direction when an opposing entry fires while a
    position is open.

    Capital is shared: position sizing uses the ATR / contracts
    column of whichever direction is being entered.
    """

    # -- extract arrays from each DF --
    def _extract(df, direction: str, params):
        n = len(df)
        return {
            "ts": df.index.to_pydatetime(),
            "close": df["close"].values.astype(np.float64),
            "entry": df["entry_long"].values.astype(bool),
            "allow": df["allow_trade"].values.astype(bool),
            "in_win": df["in_time_window"].values.astype(bool),
            "lots": df["contracts"].values.astype(np.int64),
            "atr": (df["atr"].values.astype(np.float64)
                    if "atr" in df.columns
                    else np.zeros(n, dtype=np.float64)),
            "week_end": (df["is_week_end"].values.astype(bool)
                         if "is_week_end" in df.columns
                         else np.zeros(n, dtype=bool)),
            "direction": direction,
            "exitday": int(params.exitday),
            "sdel_day": int(params.sdel_day),
            "exit_week": int(getattr(params, "exit_week", 0)),
        }

    L = _extract(df_long, "long", params_long)
    S = _extract(df_short, "short", params_short)

    # -- build unified event list: (timestamp, source_idx, meta_ref) --
    events: List[Tuple[datetime, str, int]] = []
    for i in range(len(L["ts"])):
        events.append((L["ts"][i], "long", i))
    for i in range(len(S["ts"])):
        events.append((S["ts"][i], "short", i))
    events.sort(key=lambda e: e[0])

    # -- simulation state --
    trades: List[Trade] = []
    pos_dir: str = ""          # "" = flat, "long", "short"
    pos_size: int = 0
    entry_price: float = 0.0
    entry_time: datetime = datetime.min
    current_date = None
    one_entry_today_L = False
    one_entry_today_S = False

    def _close_position(exit_ts, exit_price, reason):
        nonlocal pos_dir, pos_size, entry_price, entry_time
        sign = -1.0 if pos_dir == "short" else 1.0
        pnl = sign * (exit_price - entry_price) * pos_size
        trades.append(Trade(
            entry_time=entry_time, exit_time=exit_ts,
            entry_price=entry_price, exit_price=exit_price,
            contracts=pos_size, pnl=pnl, exit_reason=reason,
            direction=pos_dir,
        ))
        pos_dir = ""
        pos_size = 0

    def _open_position(direction, ts, price, lots, atr_val):
        nonlocal pos_dir, pos_size, entry_price, entry_time
        pos_dir = direction
        pos_size = max(int(lots * leverage), 1)
        cap = (max_contracts_long if direction == "long" else max_contracts_short)
        if cap <= 0:
            cap = max_contracts
        if cap > 0:
            pos_size = min(pos_size, cap)
        entry_price = price
        entry_time = ts

    # -- main loop --
    for ts, src, idx in events:
        bar_date = ts.date() if hasattr(ts, 'date') else ts
        if bar_date != current_date:
            current_date = bar_date
            one_entry_today_L = False
            one_entry_today_S = False

        M = L if src == "long" else S
        close_val = float(M["close"][idx])
        entry_flag = bool(M["entry"][idx])
        allow_val = bool(M["allow"][idx])
        in_win_val = bool(M["in_win"][idx])
        lots_val = int(M["lots"][idx])
        exitday = M["exitday"]
        sdel_day = M["sdel_day"]
        exit_week = M["exit_week"]
        is_wk_end = bool(M["week_end"][idx])

        # --- forced exit: expiration (either direction's allow can close) ---
        if pos_size > 0 and src == pos_dir and not allow_val:
            _close_position(ts, close_val, "expiration")
            continue

        # --- inside trading window for this source ---
        if allow_val and in_win_val:
            if pos_size == 0:
                # flat -> open if entry signal
                if entry_flag:
                    if sdel_day == 1:
                        if src == "long" and one_entry_today_L:
                            continue
                        if src == "short" and one_entry_today_S:
                            continue
                    _open_position(src, ts, close_val, lots_val, 0.0)
                    if src == "long":
                        one_entry_today_L = True
                    else:
                        one_entry_today_S = True

            elif src == pos_dir:
                # same direction bar -> check signal_reverse
                if not entry_flag:
                    _close_position(ts, close_val, "signal_reverse")

            else:
                # opposing direction bar -> handle based on flip_mode
                if entry_flag and flip_mode != "ignore":
                    if sdel_day == 1:
                        if src == "long" and one_entry_today_L:
                            continue
                        if src == "short" and one_entry_today_S:
                            continue

                    pnl_sign = -1.0 if pos_dir == "short" else 1.0
                    unrealised = pnl_sign * (close_val - entry_price)

                    do_close = False
                    do_reopen = False

                    if flip_mode == "flip":
                        do_close = True
                        do_reopen = True
                    elif flip_mode == "close_only":
                        do_close = True
                    elif flip_mode == "flip_profit":
                        if unrealised > 0:
                            do_close = True
                            do_reopen = True
                    elif flip_mode == "close_loss":
                        if unrealised <= 0:
                            do_close = True
                    elif flip_mode == "flip_long_close_short":
                        if pos_dir == "long":
                            do_close = True
                            do_reopen = True
                        else:
                            do_close = True

                    if do_close:
                        reason = "flip" if do_reopen else "close_opposing"
                        _close_position(ts, close_val, reason)
                        if do_reopen:
                            _open_position(src, ts, close_val, lots_val, 0.0)
                        if src == "long":
                            one_entry_today_L = True
                        else:
                            one_entry_today_S = True

        # --- outside window: end-of-day exit for current direction ---
        elif pos_size > 0 and src == pos_dir and exitday == 1:
            _close_position(ts, close_val, "end_of_day")

        # --- end-of-week exit ---
        if pos_size > 0 and src == pos_dir and exit_week == 1 and is_wk_end:
            _close_position(ts, close_val, "end_of_week")

    # -- close any open position at end of data --
    if pos_size > 0:
        M = L if pos_dir == "long" else S
        last_close = float(M["close"][-1])
        last_ts = M["ts"][-1]
        _close_position(last_ts, last_close, "end_of_data")

    return trades


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def compute_metrics(trades: List[Trade]) -> Dict[str, float]:
    """Compute standard performance metrics from a list of trades.

    Returns dict with keys:
        ``avg_trade``, ``num_trades``, ``total_pnl``, ``win_rate``,
        ``profit_factor``, ``max_drawdown``, ``sharpe``,
        ``avg_bars_in_trade`` (NaN if not computable).
    """
    n = len(trades)
    if n == 0:
        return {
            "avg_trade": 0.0,
            "num_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }

    pnls = np.array([t.pnl for t in trades], dtype=np.float64)
    total_pnl = float(pnls.sum())
    avg_trade = total_pnl / n

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = len(wins) / n if n > 0 else 0.0

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown from cumulative PnL curve
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    # Annualised Sharpe (assume ~250 trading days, ~1 trade per day average)
    if n > 1 and pnls.std() > 0:
        sharpe = float((pnls.mean() / pnls.std()) * np.sqrt(min(250, n)))
    else:
        sharpe = 0.0

    return {
        "avg_trade": avg_trade,
        "num_trades": n,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """Convert a list of trades to a DataFrame for analysis / export."""
    if not trades:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "entry_price", "exit_price",
            "contracts", "pnl", "exit_reason", "direction",
        ])
    return pd.DataFrame([
        {
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "contracts": t.contracts,
            "pnl": t.pnl,
            "exit_reason": t.exit_reason,
            "direction": getattr(t, "direction", "long"),
        }
        for t in trades
    ])
