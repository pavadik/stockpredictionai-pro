"""Sentiment feature extraction using FinBERT.

Provides two entry points:
  - simple_finbert_sentiment(news_df) -- score a pre-supplied news DataFrame
  - get_ticker_sentiment(ticker, start, end) -- download news via yfinance and score
"""
from typing import Optional
import pandas as pd

try:
    from transformers import pipeline as _hf_pipeline
except Exception:
    _hf_pipeline = None


def _build_clf():
    """Lazily build the FinBERT classifier pipeline."""
    if _hf_pipeline is None:
        return None
    try:
        return _hf_pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
    except Exception:
        return None


def simple_finbert_sentiment(news_df: pd.DataFrame) -> Optional[pd.Series]:
    """Score a news DataFrame with columns ['published', 'title'].

    Returns daily average positive-sentiment score [0..1], or None if unavailable.
    """
    clf = _build_clf()
    if clf is None:
        return None
    scores = []
    for _, row in news_df.iterrows():
        txt = str(row.get("title", ""))[:256]
        if not txt.strip():
            continue
        out = clf(txt)[0]
        pos = next((o['score'] for o in out if o['label'].lower() == 'positive'), 0.0)
        scores.append((row['published'].date(), float(pos)))
    if not scores:
        return None
    s = pd.DataFrame(scores, columns=["date", "pos"]).groupby("date").pos.mean()
    s.name = "news_sent_pos"
    s.index = pd.to_datetime(s.index)
    return s


def get_ticker_sentiment(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    """Download yfinance news for *ticker* and return daily FinBERT sentiment.

    Returns None gracefully if yfinance news API is unavailable, there is no news,
    or the FinBERT model cannot be loaded.
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        news = tk.news
        if not news:
            return None
        rows = []
        for item in news:
            title = item.get("title", "")
            pub = item.get("providerPublishTime")
            if pub and title:
                rows.append({"published": pd.Timestamp(pub, unit="s"), "title": title})
        if not rows:
            return None
        return simple_finbert_sentiment(pd.DataFrame(rows))
    except Exception:
        return None
