"""Shared pytest fixtures for StockPredictionAI Pro test suite."""
import os
import pytest
import pandas as pd
import numpy as np

# Ensure src is importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
PANEL_CACHE = os.path.join(FIXTURES_DIR, "panel_cache.csv")


@pytest.fixture(scope="session")
def sample_config():
    """Config with reduced epochs for fast tests."""
    return Config(
        ticker="GS", start="2018-01-01", end="2020-12-31",
        test_years=1, n_epochs=2, ae_epochs=2, batch_size=32,
        seq_len=10, hidden_size=32, num_layers=1,
        d_model=32, nhead=2, num_layers_tst=1,
    )


@pytest.fixture(scope="session")
def sample_panel(sample_config):
    """Download panel once per session, cache to disk."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    if os.path.exists(PANEL_CACHE):
        panel = pd.read_csv(PANEL_CACHE, index_col=0, parse_dates=True)
        if len(panel) > 100:
            return panel
    from src.data import build_panel
    panel = build_panel(sample_config.ticker, sample_config.start, sample_config.end)
    panel.to_csv(PANEL_CACHE)
    return panel


@pytest.fixture
def random_sequences():
    """Generate small random X, y for model tests."""
    B, T, F = 16, 10, 20
    X = np.random.randn(B, T, F).astype("float32")
    y = np.random.randn(B).astype("float32")
    return X, y
