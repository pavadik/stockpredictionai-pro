"""Momentum-trend strategy: multi-timeframe scoring for MOEX futures.

Translated from the MultiCharts EasyLanguage strategy that traded RTSF
(RTS index futures) using two analysis timeframes (e.g. 210-min and
90-min) with momentum scoring, moving-average filters, intraday time
windows, and expiration-week exclusions.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------

@dataclass
class StrategyParams:
    """All tuneable parameters for the momentum-trend strategy."""

    # --- Timeframe 1 (trend, e.g. 210-min bars) -------------------------
    lookback: int = 165
    """Momentum lookback on TF1 bars.  TrendScore compares current close
    to closes at shifts ``lookback+1 .. lookback+10``."""

    length: int = 180
    """Rolling window (in *trading-timeframe* bars) for TSA = SMA(TrendScore)
    and SMA = SMA(Close_TF1)."""

    # --- Timeframe 2 (trading, e.g. 90-min bars) ------------------------
    lookback2: int = 30
    """Momentum lookback on TF2 bars."""

    length2: int = 103
    """Rolling window for TSA2 and SMA2 (TF2)."""

    # --- Intraday time filter (minutes from first bar of each day) -------
    min_s: int = 290
    """Start of the trading window (minutes after the first M1 bar of the
    day).  Only bars inside [min_s, max_s) generate / hold signals."""

    max_s: int = 480
    """End of the trading window (minutes after the first M1 bar)."""

    # --- Signal thresholds -----------------------------------------------
    koeff1: float = 1.0
    """Multiplier for the TF1 moving-average thresholds."""

    koeff2: float = 1.005
    """Multiplier for the TF2 moving-average thresholds."""

    # --- Position sizing -------------------------------------------------
    mmcoff: int = 17
    """ATR period used for volatility-inverse position sizing."""

    capital: float = 5_000_000
    """Base capital for position-size calculation:
    ``contracts = floor(capital / (point_value_mult * ATR(mmcoff)))``."""

    point_value_mult: float = 300.0
    """Volatility divisor in the sizing formula (MC: ``capital / (300 * ATR)``)."""

    # --- Trade management ------------------------------------------------
    exitday: int = 1
    """If 1, force-close any open position when the bar leaves the
    intraday time window."""

    sdel_day: int = 0
    """If 1, allow only one entry per calendar day."""

    exit_week: int = 0
    """If 1, force-close any open position on the last bar of each
    trading week (Friday close)."""

    direction: str = "long"
    """Trade direction: ``"long"`` for buy signals, ``"short"`` for sell."""

    # --- Stop-loss --------------------------------------------------------
    sl_type: str = "none"
    """Stop-loss type: ``"none"``, ``"fixed"``, ``"atr"``,
    ``"trail_atr"``, ``"trail_pts"``, ``"breakeven"``, ``"time"``."""

    sl_pts: float = 0.0
    """Fixed stop-loss distance in points (used when sl_type="fixed")."""

    sl_atr_mult: float = 0.0
    """ATR multiplier for stop distance (used when sl_type="atr")."""

    trail_atr: float = 0.0
    """Trailing stop retracement threshold in ATR units (sl_type="trail_atr")."""

    trail_pts: float = 0.0
    """Trailing stop retracement threshold in points (sl_type="trail_pts")."""

    be_target: float = 0.0
    """Breakeven activation: move stop to entry once profit reaches this (pts)."""

    be_offset: float = 0.0
    """Breakeven offset: after activation, stop is entry_price +/- offset (pts)."""

    time_bars: int = 0
    """Time stop: exit if trade hasn't reached time_target after N bars."""

    time_target: float = 0.0
    """Time stop: minimum profit (pts) required within time_bars."""

    # --- Timeframe sizes (minutes) ---------------------------------------
    tf1_minutes: int = 210
    """Trend-analysis timeframe bar size (minutes)."""

    tf2_minutes: int = 90
    """Trading-timeframe bar size (minutes)."""

    # --- Misc ------------------------------------------------------------
    use_legacy_bug: bool = False
    """Replicate the original EasyLanguage bug where the second TrendScore
    loop used the outer-loop index (``Value1``) instead of ``Value2``.
    When *False* (default) the correct indexing is used."""


# -----------------------------------------------------------------------
# Expiration filter
# -----------------------------------------------------------------------

def apply_expiration_filter(index: pd.DatetimeIndex) -> pd.Series:
    """Return boolean Series -- *True* where trading is allowed.

    Futures expiration weeks (days 8-17 of Mar / Jun / Sep / Dec) are
    blocked.
    """
    day = index.day
    month = index.month
    in_expiry = (day > 7) & (day < 18) & (month.isin([3, 6, 9, 12]))
    return pd.Series(~in_expiry, index=index)


# -----------------------------------------------------------------------
# Intraday time filter
# -----------------------------------------------------------------------

def _minutes_from_day_start(index: pd.DatetimeIndex) -> pd.Series:
    """For each bar compute minutes elapsed since the *first bar* of
    that trading day.

    This matches the MultiCharts behaviour where ``startdatetimes`` is
    captured on the first bar after a date change, and MinS / MaxS are
    offsets from that reference.
    """
    sr = pd.Series(index.hour * 60 + index.minute, index=index)
    dates = index.date
    first_minute = sr.groupby(dates).transform("first")
    return sr - first_minute


def apply_time_filter(index: pd.DatetimeIndex,
                      min_s: int,
                      max_s: int) -> pd.Series:
    """Return boolean Series -- *True* for bars inside the time window.

    The window is ``[min_s, max_s)`` minutes from the first bar of each
    trading day.
    """
    elapsed = _minutes_from_day_start(index)
    return (elapsed >= min_s) & (elapsed < max_s)


# -----------------------------------------------------------------------
# TrendScore computation
# -----------------------------------------------------------------------

_MOMENTUM_OFFSETS = 10  # compare current close to 10 shifted values


def compute_trend_score(close: pd.Series,
                        lookback: int) -> pd.Series:
    """Vectorised TrendScore.

    For each bar *t*, score = sum over *i* in 1..10 of
    ``sign(close[t] - close[t - (lookback + i)])``.

    Returns a Series with integer values in [-10, +10].
    """
    score = pd.Series(np.zeros(len(close), dtype=np.int32),
                      index=close.index)
    for i in range(1, _MOMENTUM_OFFSETS + 1):
        diff = close.values - close.shift(lookback + i).values
        score += np.where(np.isnan(diff), 0, np.sign(diff)).astype(np.int32)
    return score


# -----------------------------------------------------------------------
# Multi-timeframe alignment
# -----------------------------------------------------------------------

def align_timeframes(df_trade: pd.DataFrame,
                     df_trend: pd.DataFrame) -> pd.DataFrame:
    """Merge the trend-timeframe (TF1) OHLCV onto the trading-timeframe
    (TF2) bars using an *as-of* join.

    For each 90-min bar we attach the most recent completed 210-min
    bar's *close* and full OHLCV (for ATR computation).

    Columns added to *df_trade*:
        ``close_tf1``, ``high_tf1``, ``low_tf1``

    Returns a copy -- the original DataFrames are not mutated.
    """
    trend_cols = df_trend[["close", "high", "low"]].rename(
        columns={"close": "close_tf1", "high": "high_tf1", "low": "low_tf1"}
    )
    merged = pd.merge_asof(
        df_trade,
        trend_cols,
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return merged


# -----------------------------------------------------------------------
# High-level data preparation
# -----------------------------------------------------------------------

def prepare_strategy_data(df_m1: pd.DataFrame,
                          params: StrategyParams) -> pd.DataFrame:
    """From raw M1 bars build the aligned multi-TF DataFrame ready for
    signal generation.

    Steps:
        1. Session-aware aggregation to TF1 and TF2.
        2. As-of alignment (TF1 onto TF2 timeline).
        3. Expiration filter column.
        4. Time-window filter column.
        5. ATR on trading timeframe (for position sizing).

    Returns:
        DataFrame indexed on TF2 timestamps with all OHLCV columns for
        both timeframes plus ``allow_trade``, ``in_time_window``, and
        ``atr`` columns.
    """
    from ..data_local import aggregate_intraday_custom

    df_trade = aggregate_intraday_custom(df_m1, params.tf2_minutes)
    df_trend = aggregate_intraday_custom(df_m1, params.tf1_minutes)

    df = align_timeframes(df_trade, df_trend)

    df["allow_trade"] = apply_expiration_filter(df.index)
    df["in_time_window"] = apply_time_filter(df.index,
                                             params.min_s,
                                             params.max_s)

    # ATR on the trading timeframe (for position sizing)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(params.mmcoff, min_periods=1).mean()

    # Week-end flag for exit_week logic
    iso = df.index.isocalendar()
    wid = np.array(iso.year * 100 + iso.week, dtype=np.int32)
    df["is_week_end"] = np.append(wid[:-1] != wid[1:], [True])

    return df


# -----------------------------------------------------------------------
# Pre-aggregated data container (for fast optimisation loops)
# -----------------------------------------------------------------------

@dataclass
class PreAggregatedData:
    """Holds pre-aggregated and aligned multi-TF data so that the
    expensive aggregation + alignment step runs only once.

    Fields that depend on ``min_s / max_s / mmcoff`` are recomputed per
    trial via :func:`apply_params_fast`.
    """
    df_aligned: pd.DataFrame
    """Aligned TF2 bars with TF1 close/high/low, expiration filter,
    and ``elapsed_minutes`` (minutes from day start) pre-computed."""

    elapsed_minutes: pd.Series
    """Minutes from the first bar of each trading day (for time filter)."""

    true_range: pd.Series
    """Pre-computed True Range (for ATR rolling with variable mmcoff)."""

    week_id: np.ndarray
    """ISO week identifier per bar (year*100 + week_number).
    Used to detect end-of-week boundaries."""


def pre_aggregate(df_m1: pd.DataFrame,
                  params: StrategyParams) -> PreAggregatedData:
    """Run the heavy aggregation + alignment once.

    Returns a :class:`PreAggregatedData` that can be reused across many
    parameter trials.
    """
    from ..data_local import aggregate_intraday_custom

    df_trade = aggregate_intraday_custom(df_m1, params.tf2_minutes)
    df_trend = aggregate_intraday_custom(df_m1, params.tf1_minutes)
    df = align_timeframes(df_trade, df_trend)
    df["allow_trade"] = apply_expiration_filter(df.index)

    elapsed = _minutes_from_day_start(df.index)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    iso = df.index.isocalendar()
    week_id = np.array(iso.year * 100 + iso.week, dtype=np.int32)

    return PreAggregatedData(df_aligned=df, elapsed_minutes=elapsed,
                             true_range=tr, week_id=week_id)


def apply_params_fast(pre: PreAggregatedData,
                      params: StrategyParams) -> pd.DataFrame:
    """Cheaply apply trial-specific parameters to pre-aggregated data.

    Recomputes only: time filter, ATR (mmcoff), and signal indicators.
    Returns a DataFrame ready for :func:`simulate_trades`.
    """
    df = pre.df_aligned.copy()

    df["in_time_window"] = ((pre.elapsed_minutes >= params.min_s)
                            & (pre.elapsed_minutes < params.max_s))
    df["atr"] = pre.true_range.rolling(params.mmcoff, min_periods=1).mean()

    df = generate_signals(df, params)
    return df


# -----------------------------------------------------------------------
# Signal generation
# -----------------------------------------------------------------------

def generate_signals(df: pd.DataFrame,
                     params: StrategyParams) -> pd.DataFrame:
    """Compute TrendScores, smoothed averages, and entry / exit flags.

    Expects *df* to be the output of :func:`prepare_strategy_data`.

    New columns added in-place:
        ``trend_score``, ``tsa``, ``sma_tf1``,
        ``trend_score2``, ``tsa2``, ``sma_tf2``,
        ``entry_long``, ``exit_long``, ``contracts``

    Returns the same DataFrame (mutated).
    """
    # --- TF1 signals (trend timeframe) ----------------------------------
    df["trend_score"] = compute_trend_score(df["close_tf1"],
                                            params.lookback)
    df["tsa"] = df["trend_score"].rolling(params.length,
                                          min_periods=1).mean()
    df["sma_tf1"] = df["close_tf1"].rolling(params.length,
                                            min_periods=1).mean()

    # --- TF2 signals (trading timeframe) --------------------------------
    if params.use_legacy_bug:
        # Replicate the MC bug: second loop always uses lookback + 10
        df["trend_score2"] = compute_trend_score(df["close"],
                                                 params.lookback + 10 - 1)
    else:
        df["trend_score2"] = compute_trend_score(df["close"],
                                                 params.lookback2)

    df["tsa2"] = df["trend_score2"].rolling(params.length2,
                                            min_periods=1).mean()
    df["sma_tf2"] = df["close"].rolling(params.length2,
                                        min_periods=1).mean()

    # --- Composite conditions -------------------------------------------
    cond_tf1 = (
        (df["trend_score"] > df["tsa"] * params.koeff1)
        & (df["close_tf1"] > df["sma_tf1"] * params.koeff1)
    )
    cond_tf2 = (
        (df["trend_score2"] > df["tsa2"] * params.koeff2)
        & (df["close"] > df["sma_tf2"] * params.koeff2)
    )

    active = df["allow_trade"] & df["in_time_window"]

    # Short conditions are the mirror of long
    cond_tf1_short = (
        (df["trend_score"] < df["tsa"] * (2 - params.koeff1))
        & (df["close_tf1"] < df["sma_tf1"] * (2 - params.koeff1))
    )
    cond_tf2_short = (
        (df["trend_score2"] < df["tsa2"] * (2 - params.koeff2))
        & (df["close"] < df["sma_tf2"] * (2 - params.koeff2))
    )

    if params.direction == "short":
        df["entry_long"] = active & cond_tf1_short & cond_tf2_short
        df["exit_long"] = active & cond_tf1_short & ~cond_tf2_short
    else:
        df["entry_long"] = active & cond_tf1 & cond_tf2
        df["exit_long"] = active & cond_tf1 & ~cond_tf2

    # Position sizing: contracts = floor(capital / (mult * ATR))
    safe_atr = df["atr"].replace(0, np.nan)
    raw_contracts = params.capital / (params.point_value_mult * safe_atr)
    df["contracts"] = raw_contracts.fillna(0).astype(int).clip(lower=0)

    return df
