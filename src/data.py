import numpy as np
import pandas as pd
import yfinance as yf
from .utils import indicators as ind

DEFAULT_CORRELATED = [
    "SPY","XLF","JPM","BAC","MS","^VIX","^TNX","DX-Y.NYB","GC=F","EURUSD=X"
]

def download_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    else:
        prices = data.copy()
    prices = prices.ffill().bfill()
    return prices

def build_panel(ticker: str, start: str, end: str) -> pd.DataFrame:
    tickers = [ticker] + DEFAULT_CORRELATED
    prices = download_prices(tickers, start, end)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(ticker)
    df = prices.copy()
    # технические индикаторы по целевой акции
    s = df[ticker]
    df["sma7"] = ind.sma(s, 7)
    df["sma21"] = ind.sma(s, 21)
    df["ema21"] = ind.ema(s, 21)
    macd_line, signal_line, hist = ind.macd(s)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    bbu, bbm, bbl = ind.bollinger(s)
    df["bb_upper"] = bbu
    df["bb_mid"] = bbm
    df["bb_lower"] = bbl
    df["rsi14"] = ind.rsi(s, 14)
    df["mom10"] = ind.momentum(s, 10)

    df = df.dropna().copy()
    return df
