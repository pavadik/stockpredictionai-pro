import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, window=20, n_std=2):
    ma = sma(series, window)
    std = series.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return upper, ma, lower


def adx(df: pd.DataFrame, window=14) -> pd.Series:
    """Average Directional Index (ADX). Expects df with high, low, close."""
    alpha = 1.0 / window
    
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    # Wilder's Smoothing is basically EMA with alpha=1/window
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    return adx


def momentum(series: pd.Series, window=10):
    return series / series.shift(window) - 1.0


def log_momentum(series: pd.Series, window=10):
    """Natural-log momentum: ln(price / price_shifted)."""
    ratio = series / series.shift(window)
    return np.log(ratio.clip(lower=1e-9))
