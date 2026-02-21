"""Local MOEX data loader for date-based hierarchy (G:\\data2).

Loads M1 (1-minute) bars and tick data from the directory structure:
    YEAR/MONTH/DAY/TICKER/M1/{data.txt | data.csv}
    YEAR/MONTH/DAY/TICKER/ticks/data.txt

Supports aggregation to arbitrary timeframes:
    M1, M3, M5, M7, M14, M30, H1, H4, D1
building OHLCV bars from raw tick data, and session-aware custom
intraday aggregation (e.g. 90-min, 210-min) that avoids bars
spanning market breaks.
"""
import os
import warnings
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    Config, MOEX_CORRELATED_DEFAULT,
    parse_timeframe, timeframe_mult,
)
from .utils import indicators as ind

M1_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]
TICK_COLUMNS = ["date", "time", "price", "volume"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_data_file(directory: str) -> Optional[str]:
    """Find the data file in a directory (data.txt preferred, fallback data.csv)."""
    for name in ("data.txt", "data.csv"):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def _iter_date_dirs(data_path: str, start_dt: datetime, end_dt: datetime):
    """Yield (datetime, day_path) for each date directory in range."""
    for year_dir in sorted(os.listdir(data_path)):
        year_path = os.path.join(data_path, year_dir)
        if not os.path.isdir(year_path) or not year_dir.isdigit():
            continue
        year = int(year_dir)
        if year < start_dt.year or year > end_dt.year:
            continue

        for month_dir in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month_dir)
            if not os.path.isdir(month_path):
                continue

            for day_dir in sorted(os.listdir(month_path)):
                day_path = os.path.join(month_path, day_dir)
                if not os.path.isdir(day_path):
                    continue
                try:
                    dt = datetime(year, int(month_dir), int(day_dir))
                except ValueError:
                    continue
                if dt < start_dt or dt > end_dt:
                    continue
                yield dt, day_path


def _read_ohlcv_file(filepath: str) -> pd.DataFrame:
    """Read an M1 OHLCV file (no header, 7 columns)."""
    df = pd.read_csv(
        filepath, header=None, names=M1_COLUMNS,
        dtype={"date": str, "time": str,
               "open": np.float32, "high": np.float32,
               "low": np.float32, "close": np.float32,
               "volume": np.int64},
    )
    if df.empty:
        return df
    timestamps = pd.to_datetime(
        df["date"] + " " + df["time"].astype(str).str.zfill(6),
        format="%m/%d/%y %H%M%S",
    )
    df.index = timestamps
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def _read_tick_file(filepath: str) -> pd.DataFrame:
    """Read a tick data file (no header, 4 columns)."""
    df = pd.read_csv(
        filepath, header=None, names=TICK_COLUMNS,
        dtype={"date": str, "time": str,
               "price": np.float64, "volume": np.int64},
    )
    if df.empty:
        return df
    timestamps = pd.to_datetime(
        df["date"] + " " + df["time"].astype(str).str.zfill(6),
        format="%m/%d/%y %H%M%S",
    )
    df.index = timestamps
    df = df[["price", "volume"]]
    return df


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_available_tickers(data_path: str,
                           start: str = None,
                           end: str = None) -> dict:
    """Scan the data hierarchy and return available tickers with date ranges.

    Returns:
        dict: ticker -> {"first_date": str, "last_date": str, "days": int,
                         "has_m1": bool, "has_ticks": bool}
    """
    from collections import defaultdict
    tickers = defaultdict(lambda: {"dates": [], "has_m1": False, "has_ticks": False})

    start_dt = datetime.strptime(start, "%Y-%m-%d") if start else datetime(1900, 1, 1)
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime(2100, 1, 1)

    for dt, day_path in _iter_date_dirs(data_path, start_dt, end_dt):
        for ticker_dir in os.listdir(day_path):
            ticker_path = os.path.join(day_path, ticker_dir)
            if not os.path.isdir(ticker_path):
                continue
            info = tickers[ticker_dir]
            info["dates"].append(dt)
            m1_dir = os.path.join(ticker_path, "M1")
            tick_dir = os.path.join(ticker_path, "ticks")
            if os.path.isdir(m1_dir) and _find_data_file(m1_dir):
                info["has_m1"] = True
            if os.path.isdir(tick_dir) and _find_data_file(tick_dir):
                info["has_ticks"] = True

    result = {}
    for tk, info in tickers.items():
        dates = info["dates"]
        if dates:
            result[tk] = {
                "first_date": min(dates).strftime("%Y-%m-%d"),
                "last_date": max(dates).strftime("%Y-%m-%d"),
                "days": len(dates),
                "has_m1": info["has_m1"],
                "has_ticks": info["has_ticks"],
            }
    return result


# ---------------------------------------------------------------------------
# M1 loader
# ---------------------------------------------------------------------------

def load_m1_bars(data_path: str, ticker: str,
                 start: str, end: str) -> pd.DataFrame:
    """Load M1 (1-minute) bars for a single ticker from the date hierarchy.

    Looks for data.txt or data.csv inside each TICKER/M1/ directory.
    Reads only files within [start, end] range for memory efficiency.

    Returns:
        DataFrame with columns [open, high, low, close, volume] and DatetimeIndex.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    frames = []

    for dt, day_path in _iter_date_dirs(data_path, start_dt, end_dt):
        m1_dir = os.path.join(day_path, ticker, "M1")
        if not os.path.isdir(m1_dir):
            continue
        filepath = _find_data_file(m1_dir)
        if filepath is None:
            continue
        try:
            df = _read_ohlcv_file(filepath)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        raise RuntimeError(
            f"No M1 data found for ticker '{ticker}' in [{start}, {end}] "
            f"at {data_path}"
        )

    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    return result


# ---------------------------------------------------------------------------
# Tick loader
# ---------------------------------------------------------------------------

def load_ticks(data_path: str, ticker: str,
               start: str, end: str) -> pd.DataFrame:
    """Load tick data for a single ticker from the date hierarchy.

    Looks for data.txt inside each TICKER/ticks/ directory.

    Returns:
        DataFrame with columns [price, volume] and DatetimeIndex.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    frames = []

    for dt, day_path in _iter_date_dirs(data_path, start_dt, end_dt):
        tick_dir = os.path.join(day_path, ticker, "ticks")
        if not os.path.isdir(tick_dir):
            continue
        filepath = _find_data_file(tick_dir)
        if filepath is None:
            continue
        try:
            df = _read_tick_file(filepath)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        raise RuntimeError(
            f"No tick data found for ticker '{ticker}' in [{start}, {end}] "
            f"at {data_path}"
        )

    result = pd.concat(frames).sort_index()
    return result


# ---------------------------------------------------------------------------
# Aggregation: M1 -> any timeframe
# ---------------------------------------------------------------------------

_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def aggregate_timeframe(df_m1: pd.DataFrame,
                        timeframe: str) -> pd.DataFrame:
    """Aggregate M1 OHLCV bars to an arbitrary timeframe.

    Args:
        df_m1: DataFrame with M1 OHLCV data and DatetimeIndex.
        timeframe: Any supported string: M1, M3, M5, M7, M14, M30, H1, H4, D1.

    Returns:
        Aggregated OHLCV DataFrame.
    """
    info = parse_timeframe(timeframe)
    if info["unit"] == "M" and info["n"] == 1:
        return df_m1
    if info["unit"] == "tick":
        raise ValueError("Cannot aggregate OHLCV bars to tick timeframe. "
                         "Use ticks_to_bars() instead.")

    rule = info["resample_rule"]
    agg = df_m1.resample(rule).agg(_OHLCV_AGG).dropna(subset=["close"])
    return agg


def aggregate_intraday_custom(df_m1: pd.DataFrame,
                              n_minutes: int) -> pd.DataFrame:
    """Session-aware aggregation of M1 bars to custom intraday timeframes.

    Unlike :func:`aggregate_timeframe` (which uses calendar-aligned
    ``resample``), this function groups bars **within each trading day**
    so that resulting bars never span overnight or lunch breaks.

    Args:
        df_m1: DataFrame with M1 OHLCV data and DatetimeIndex.
        n_minutes: Number of minutes per output bar (e.g. 90, 210).

    Returns:
        Aggregated OHLCV DataFrame whose index is the timestamp of the
        last M1 bar in each group.
    """
    if n_minutes <= 0:
        raise ValueError(f"n_minutes must be positive, got {n_minutes}")
    if n_minutes == 1:
        return df_m1.copy()

    frames: list[pd.DataFrame] = []
    for _date, day in df_m1.groupby(df_m1.index.date):
        n = len(day)
        if n == 0:
            continue
        group_ids = np.arange(n) // n_minutes
        grouped = day.groupby(group_ids)
        agg = grouped.agg(_OHLCV_AGG)
        agg.index = grouped.apply(lambda g: g.index[-1])
        frames.append(agg)

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    result = pd.concat(frames)
    for col in ("open", "high", "low", "close"):
        result[col] = result[col].astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Aggregation: ticks -> OHLCV bars
# ---------------------------------------------------------------------------

def ticks_to_bars(df_ticks: pd.DataFrame,
                  timeframe: str) -> pd.DataFrame:
    """Build OHLCV bars from raw tick data at any resolution.

    Args:
        df_ticks: DataFrame with columns [price, volume] and DatetimeIndex.
        timeframe: Target resolution: M1, M3, M5, M7, M14, M30, H1, H4, D1.

    Returns:
        OHLCV DataFrame with columns [open, high, low, close, volume].
    """
    info = parse_timeframe(timeframe)
    if info["unit"] == "tick":
        raise ValueError("Cannot aggregate ticks to 'tick' timeframe. "
                         "Use load_ticks() directly.")

    rule = info["resample_rule"]
    grouped = df_ticks.resample(rule)
    bars = pd.DataFrame({
        "open": grouped["price"].first(),
        "high": grouped["price"].max(),
        "low": grouped["price"].min(),
        "close": grouped["price"].last(),
        "volume": grouped["volume"].sum(),
    })
    bars = bars.dropna(subset=["close"])

    # Ensure float32 for memory efficiency
    for col in ["open", "high", "low", "close"]:
        bars[col] = bars[col].astype(np.float32)

    return bars


# ---------------------------------------------------------------------------
# Load + aggregate (unified entry point)
# ---------------------------------------------------------------------------

def load_bars(data_path: str, ticker: str,
              start: str, end: str,
              timeframe: str = "M1",
              raw_source: str = "m1") -> pd.DataFrame:
    """Load OHLCV bars for a ticker at any timeframe.

    This is the unified entry point that handles both raw data sources:
    - raw_source="m1": load M1 bars, then aggregate to target timeframe
    - raw_source="ticks": load tick data, then build bars at target timeframe

    Returns:
        OHLCV DataFrame with columns [open, high, low, close, volume].
    """
    if raw_source == "ticks":
        ticks = load_ticks(data_path, ticker, start, end)
        if timeframe.upper() == "TICK":
            # Return tick data as pseudo-OHLCV (price for all OHLC)
            return pd.DataFrame({
                "open": ticks["price"].astype(np.float32),
                "high": ticks["price"].astype(np.float32),
                "low": ticks["price"].astype(np.float32),
                "close": ticks["price"].astype(np.float32),
                "volume": ticks["volume"],
            })
        return ticks_to_bars(ticks, timeframe)
    else:
        m1 = load_m1_bars(data_path, ticker, start, end)
        return aggregate_timeframe(m1, timeframe)


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------

def _get_correlated_tickers(cfg: Config,
                            available: dict) -> List[str]:
    """Determine which correlated tickers to use for the panel."""
    if cfg.local_correlated:
        candidates = list(cfg.local_correlated)
    else:
        candidates = list(MOEX_CORRELATED_DEFAULT)

    result = [t for t in candidates
              if t != cfg.ticker and t in available]
    return result


def build_local_panel(cfg: Config) -> pd.DataFrame:
    """Build a feature panel from local MOEX data.

    Loads M1 bars (or ticks), aggregates to the target timeframe, adds
    correlated tickers and technical indicators. Produces a panel compatible
    with the existing pipeline (same structure as build_panel from data.py).

    Supports arbitrary timeframes: M1, M3, M5, M7, M14, M30, H1, H4, D1.
    Supports tick-based construction via cfg.local_raw_source="ticks".
    """
    data_path = cfg.data_path
    if not data_path or not os.path.isdir(data_path):
        raise RuntimeError(f"data_path is not a valid directory: {data_path}")

    # Load target ticker at target timeframe
    target_df = load_bars(data_path, cfg.ticker, cfg.start, cfg.end,
                          timeframe=cfg.timeframe,
                          raw_source=cfg.local_raw_source)

    # Use Close price as the main column (matches yfinance panel convention)
    df = pd.DataFrame({cfg.ticker: target_df["close"]})

    # Load correlated tickers
    available = scan_available_tickers(data_path, cfg.start, cfg.end)
    corr_tickers = _get_correlated_tickers(cfg, available)

    for ct in corr_tickers:
        try:
            ct_df = load_bars(data_path, ct, cfg.start, cfg.end,
                              timeframe=cfg.timeframe,
                              raw_source=cfg.local_raw_source)
            df[ct] = ct_df["close"]
        except Exception:
            continue

    # Forward-fill then back-fill small gaps
    df = df.ffill().bfill()

    # Add volume features from the target ticker if enabled
    if cfg.use_volume_features:
        df[f"{cfg.ticker}_volume"] = target_df["volume"]
        df[f"{cfg.ticker}_range"] = target_df["high"] - target_df["low"]

        # Rolling VWAP (20-bar window instead of cumulative to avoid drift)
        typical = (target_df["high"] + target_df["low"] + target_df["close"]) / 3.0
        tv = typical * target_df["volume"]
        vwap_window = 20
        df[f"{cfg.ticker}_vwap"] = (
            tv.rolling(vwap_window, min_periods=1).sum()
            / target_df["volume"].rolling(vwap_window, min_periods=1).sum().replace(0, np.nan)
        )

        df[f"{cfg.ticker}_gap"] = target_df["open"] - target_df["close"].shift(1)

        # ATR (Average True Range) -- needed for D2 volatility-adjusted target
        prev_close = target_df["close"].shift(1)
        tr1 = target_df["high"] - target_df["low"]
        tr2 = (target_df["high"] - prev_close).abs()
        tr3 = (target_df["low"] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_period = getattr(cfg, "atr_period", 14)
        df[f"{cfg.ticker}_atr"] = true_range.ewm(span=atr_period, adjust=False).mean()

        # OBV (On-Balance Volume)
        close_diff = target_df["close"].diff()
        obv_sign = np.sign(close_diff).fillna(0)
        df[f"{cfg.ticker}_obv"] = (obv_sign * target_df["volume"]).cumsum()

    # Technical indicators (optional -- ablation showed neutral effect)
    if cfg.use_indicators:
        mult = cfg.timeframe_mult
        max_safe_mult = max(1, len(df) // 60)
        mult = min(mult, max_safe_mult)
        s = df[cfg.ticker]

        df["sma7"] = ind.sma(s, 7 * mult)
        df["sma21"] = ind.sma(s, 21 * mult)
        df["ema21"] = ind.ema(s, 21 * mult)

        macd_fast = 12 * mult
        macd_slow = 26 * mult
        macd_signal = 9 * mult
        macd_line, signal_line, hist = ind.macd(s, fast=macd_fast,
                                                slow=macd_slow,
                                                signal=macd_signal)
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = hist

        bbu, bbm, bbl = ind.bollinger(s, window=20 * mult)
        df["bb_upper"] = bbu
        df["bb_mid"] = bbm
        df["bb_lower"] = bbl

        df["rsi14"] = ind.rsi(s, 14 * mult)
        df["mom10"] = ind.momentum(s, 10 * mult)
        df["log_mom10"] = ind.log_momentum(s, 10 * mult)

    df = df.dropna().copy()
    if df.empty:
        raise RuntimeError(
            "Local panel is empty after dropna(). "
            "Check data availability and date range."
        )

    # Ensure float32 for memory efficiency
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    return df
