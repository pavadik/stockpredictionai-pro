import numpy as np
import pandas as pd
from typing import Tuple

def train_test_split_by_years(df: pd.DataFrame, test_years: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df.index.max().year - test_years + 1
    train = df[df.index.year < cutoff].copy()
    test = df[df.index.year >= cutoff].copy()
    return train, test

def make_sequences(df: pd.DataFrame, target_col: str, seq_len: int = 17):
    X, y = [], []
    vals = df.values.astype('float32')
    t = df[target_col].values.astype('float32')
    for i in range(len(df) - seq_len - 1):
        X.append(vals[i:i+seq_len])
        # predict next-bar price change (delta)
        y.append(t[i+seq_len+0] - t[i+seq_len-1])
    return np.array(X), np.array(y)


def make_sequences_multi(df: pd.DataFrame, target_col: str, seq_len: int = 12,
                         horizons: tuple = (1, 2, 4)):
    """Multi-horizon sequences: predict deltas at multiple horizons ahead.

    Returns X [N, seq_len, F] and y [N, H] where H = len(horizons).
    y[i][j] = close[t + horizons[j]] - close[t], where t = i + seq_len - 1.
    """
    X, y = [], []
    vals = df.values.astype('float32')
    t = df[target_col].values.astype('float32')
    max_h = max(horizons)
    for i in range(len(df) - seq_len - max_h):
        X.append(vals[i:i + seq_len])
        base = t[i + seq_len - 1]
        y.append([t[i + seq_len - 1 + h] - base for h in horizons])
    return np.array(X), np.array(y)


def make_sequences_atr(df: pd.DataFrame, target_col: str, atr_col: str,
                       seq_len: int = 12):
    """ATR-normalized sequences: target = delta / ATR.

    Returns X [N, seq_len, F], y_norm [N], atr_vals [N].
    atr_vals can be used to inverse-transform predictions back to raw deltas.
    All outputs are float32.
    """
    X, y_norm, atr_vals = [], [], []
    vals = df.values.astype('float32')
    t = df[target_col].values.astype('float32')
    atr = df[atr_col].values.astype('float32')
    eps = np.float32(1e-8)
    for i in range(len(df) - seq_len - 1):
        X.append(vals[i:i + seq_len])
        delta = t[i + seq_len] - t[i + seq_len - 1]
        atr_val = max(atr[i + seq_len - 1], eps)
        y_norm.append(np.float32(delta / atr_val))
        atr_vals.append(np.float32(atr_val))
    return np.array(X, dtype='float32'), np.array(y_norm, dtype='float32'), np.array(atr_vals, dtype='float32')


def walk_forward_splits(df, n_splits=5, min_train=500, step=60):
    import numpy as np
    idx = np.arange(len(df))
    for i in range(n_splits):
        train_end = min_train + i*step
        test_end = train_end + step
        if test_end >= len(df):
            break
        tr_idx = idx[:train_end]
        te_idx = idx[train_end:test_end]
        yield tr_idx, te_idx
