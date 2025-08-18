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
        # предсказываем дневное изменение цены
        y.append(t[i+seq_len+0] - t[i+seq_len-1])
    return np.array(X), np.array(y)


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
