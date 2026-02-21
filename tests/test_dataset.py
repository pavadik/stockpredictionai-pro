"""Tests for dataset splitting and sequence creation."""
import numpy as np
import pandas as pd
from src.dataset import train_test_split_by_years, make_sequences, walk_forward_splits


def test_train_test_split_no_overlap(sample_panel):
    train, test = train_test_split_by_years(sample_panel, test_years=1)
    assert len(train) > 0
    assert len(test) > 0
    assert train.index.max() < test.index.min(), "Train and test should not overlap"


def test_make_sequences_shapes(sample_panel):
    seq_len = 10
    X, y = make_sequences(sample_panel, target_col="GS", seq_len=seq_len)
    assert X.ndim == 3
    assert X.shape[1] == seq_len
    assert X.shape[2] == sample_panel.shape[1]
    assert len(y) == X.shape[0]
    assert X.shape[0] == len(sample_panel) - seq_len - 1


def test_walk_forward_splits_count(sample_panel):
    splits = list(walk_forward_splits(sample_panel, n_splits=3, min_train=100, step=50))
    assert len(splits) <= 3
    assert len(splits) >= 1
    for tr_idx, te_idx in splits:
        assert len(tr_idx) > 0
        assert len(te_idx) > 0
        # test indices come after train indices
        assert tr_idx[-1] < te_idx[0]


def test_make_sequences_target_is_delta(sample_panel):
    """Target should be next-day price change, not raw price."""
    X, y = make_sequences(sample_panel, target_col="GS", seq_len=10)
    # Deltas should be centered around 0
    assert abs(y.mean()) < y.std() * 3, "Targets should be price deltas, not raw prices"
    assert not np.isnan(y).any()
