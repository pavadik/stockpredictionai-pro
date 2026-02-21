"""Tests for data loading and panel construction."""
import pandas as pd


def test_download_prices_returns_dataframe(sample_panel):
    assert isinstance(sample_panel, pd.DataFrame)
    assert sample_panel.shape[0] > 100


def test_build_panel_has_indicators(sample_panel):
    expected = ["sma7", "sma21", "ema21", "macd", "macd_signal",
                "macd_hist", "bb_upper", "bb_mid", "bb_lower",
                "rsi14", "mom10", "log_mom10"]
    for col in expected:
        assert col in sample_panel.columns, f"Missing indicator column: {col}"


def test_build_panel_no_nans(sample_panel):
    assert sample_panel.isna().sum().sum() == 0, "Panel should have no NaN after dropna"


def test_build_panel_has_ticker(sample_panel):
    assert "GS" in sample_panel.columns
