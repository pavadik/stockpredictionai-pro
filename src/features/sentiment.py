from typing import Optional
import pandas as pd
from datetime import datetime
try:
    from transformers import pipeline
except Exception:
    pipeline = None

def simple_finbert_sentiment(news_df: pd.DataFrame) -> Optional[pd.Series]:
    """news_df: columns ['published', 'title'].
    Возвращает дневной средний сентимент [0..1] (положительный).
    Требует интернет/модель. Если недоступно — вернёт None.
    """
    if pipeline is None:
        return None
    try:
        clf = pipeline("text-classification", model="ProsusAI/finbert" , top_k=None)
    except Exception:
        return None
    scores = []
    for _, row in news_df.iterrows():
        txt = str(row.get("title", ""))[:256]
        if not txt.strip():
            continue
        out = clf(txt)[0]
        # convert {'label': 'positive'|'negative'|'neutral', 'score': p}
        pos = next((o['score'] for o in out if o['label'].lower()=='positive'), 0.0)
        scores.append((row['published'].date(), float(pos)))
    if not scores:
        return None
    s = pd.DataFrame(scores, columns=["date","pos"]).groupby("date").pos.mean()
    s.name = "news_sent_pos"
    return s
