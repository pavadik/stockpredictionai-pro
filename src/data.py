import numpy as np
import pandas as pd
import yfinance as yf
from .config import Config
from .utils import indicators as ind

DEFAULT_CORRELATED = [
    # --- Competitors / Financials ---
    "JPM", "BAC", "MS", "C", "WFC", "BLK", "SCHW", "AXP",
    # --- Sector ETFs ---
    "SPY", "XLF", "XLK", "XLE", "XLI",
    # --- US Indices ---
    "^GSPC", "^DJI", "^IXIC", "^RUT",
    # --- Global Indices ---
    "^FTSE", "^N225", "^HSI", "^BSESN", "^GDAXI", "^FCHI",
    # --- Volatility ---
    "^VIX",
    # --- Fixed Income / Rates ---
    "^TNX", "^TYX", "^FVX", "TLT", "SHY", "IEF",
    # --- Currencies ---
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X",
    # --- Commodities ---
    "GC=F", "SI=F", "CL=F", "NG=F",
    # --- Dollar index ---
    "DX-Y.NYB",
]

def download_prices(tickers, start, end):
    """Download Close prices for a list of tickers via yfinance.

    Handles both legacy (flat columns) and modern (MultiIndex) yfinance formats.
    Tickers that fail to download are silently dropped.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True,
                       progress=False, threads=False)

    if data.empty:
        raise RuntimeError(f"yfinance returned empty DataFrame for tickers={tickers}")

    # yfinance >= 0.2.31 returns MultiIndex columns (Price, Ticker)
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0).unique()
        if "Close" in level0:
            prices = data["Close"].copy()
        elif "Adj Close" in level0:
            prices = data["Adj Close"].copy()
        else:
            # Fallback: take the first price level
            prices = data[level0[0]].copy()
    elif "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    elif "Close" in data.columns:
        prices = data["Close"].copy()
    else:
        prices = data.copy()

    # Drop tickers that are entirely NaN (failed downloads)
    if isinstance(prices, pd.DataFrame):
        prices = prices.dropna(axis=1, how="all")

    prices = prices.ffill().bfill()
    return prices

def build_panel(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Build a feature panel from yfinance data (original, daily only)."""
    tickers = [ticker] + DEFAULT_CORRELATED
    prices = download_prices(tickers, start, end)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(ticker)
    df = prices.copy()

    if ticker not in df.columns:
        raise RuntimeError(f"Target ticker '{ticker}' not found in downloaded data. "
                           f"Available: {list(df.columns)}")

    # Technical indicators are always added for yfinance (D1 daily data)
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
    df["log_mom10"] = ind.log_momentum(s, 10)

    df = df.dropna().copy()
    if df.empty:
        raise RuntimeError("Panel is empty after dropna(). Check ticker data availability.")
    return df


def build_panel_auto(cfg: Config) -> pd.DataFrame:
    """Universal panel builder: dispatches to yfinance or local loader.

    - data_source="yfinance": uses build_panel() (daily US stocks)
    - data_source="local": uses build_local_panel() from data_local module
    """
    if cfg.data_source == "local":
        from .data_local import build_local_panel
        return build_local_panel(cfg)
    else:
        return build_panel(cfg.ticker, cfg.start, cfg.end)
