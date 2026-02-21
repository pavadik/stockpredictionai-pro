"""Tests for all technical indicator functions in src/utils/indicators.py."""
import numpy as np
import pandas as pd
import pytest

from src.utils.indicators import sma, ema, rsi, macd, bollinger, momentum, log_momentum


@pytest.fixture
def price_series():
    """Simple ascending price series for predictable indicator values."""
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    prices = pd.Series(np.linspace(100, 150, 50), index=idx, name="price")
    return prices


@pytest.fixture
def constant_series():
    """Constant price series for edge-case testing."""
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    return pd.Series(100.0, index=idx, name="flat")


# --- SMA ---

class TestSMA:
    def test_basic_shape(self, price_series):
        result = sma(price_series, 7)
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_series)

    def test_first_values_are_nan(self, price_series):
        result = sma(price_series, 7)
        assert result.iloc[:6].isna().all()
        assert result.iloc[6:].notna().all()

    def test_known_value(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_window_1_equals_original(self, price_series):
        result = sma(price_series, 1)
        np.testing.assert_array_almost_equal(result.values, price_series.values)

    def test_constant_series(self, constant_series):
        result = sma(constant_series, 10)
        assert result.iloc[9:].eq(100.0).all()


# --- EMA ---

class TestEMA:
    def test_basic_shape(self, price_series):
        result = ema(price_series, 21)
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_series)

    def test_no_nans(self, price_series):
        result = ema(price_series, 21)
        assert result.notna().all()

    def test_ema_follows_trend(self, price_series):
        """EMA on ascending series should be below actual for most of it."""
        result = ema(price_series, 10)
        # After initial ramp-up, EMA should lag below ascending price
        assert (result.iloc[15:] < price_series.iloc[15:]).all()

    def test_constant_series(self, constant_series):
        result = ema(constant_series, 5)
        np.testing.assert_array_almost_equal(result.values, 100.0)


# --- RSI ---

class TestRSI:
    def test_basic_shape(self, price_series):
        result = rsi(price_series, 14)
        assert len(result) == len(price_series)

    def test_rsi_bounded(self, price_series):
        result = rsi(price_series, 14).dropna()
        assert (result >= 0).all()
        assert (result <= 100).all()

    def test_ascending_series_high_rsi(self, price_series):
        """Monotonically ascending series should have very high RSI."""
        result = rsi(price_series, 14).dropna()
        assert result.iloc[-1] > 90.0

    def test_descending_series_low_rsi(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        desc = pd.Series(np.linspace(150, 100, 50), index=idx)
        result = rsi(desc, 14).dropna()
        assert result.iloc[-1] < 10.0

    def test_constant_series(self, constant_series):
        """RSI of constant series: no up/down movement."""
        result = rsi(constant_series, 14).dropna()
        # All deltas are 0, so RSI should be ~50 or NaN-ish
        assert np.isfinite(result.values).all()


# --- MACD ---

class TestMACD:
    def test_returns_three_series(self, price_series):
        macd_line, signal_line, hist = macd(price_series)
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd_line) == len(price_series)

    def test_histogram_is_macd_minus_signal(self, price_series):
        macd_line, signal_line, hist = macd(price_series)
        expected = macd_line - signal_line
        np.testing.assert_array_almost_equal(hist.values, expected.values)

    def test_ascending_series_positive_macd(self, price_series):
        macd_line, _, _ = macd(price_series)
        # For monotonically ascending, fast EMA > slow EMA => MACD > 0
        assert macd_line.iloc[-1] > 0

    def test_no_nans_after_warmup(self, price_series):
        macd_line, signal_line, hist = macd(price_series)
        assert macd_line.notna().all()
        assert signal_line.notna().all()


# --- Bollinger Bands ---

class TestBollinger:
    def test_returns_three_series(self, price_series):
        upper, mid, lower = bollinger(price_series, window=20, n_std=2)
        assert len(upper) == len(price_series)
        assert len(mid) == len(price_series)
        assert len(lower) == len(price_series)

    def test_upper_above_lower(self, price_series):
        upper, mid, lower = bollinger(price_series)
        valid = upper.dropna().index
        assert (upper.loc[valid] >= lower.loc[valid]).all()

    def test_mid_is_sma(self, price_series):
        upper, mid, lower = bollinger(price_series, window=20)
        expected_sma = sma(price_series, 20)
        np.testing.assert_array_almost_equal(
            mid.dropna().values, expected_sma.dropna().values
        )

    def test_constant_series_bands_collapse(self, constant_series):
        upper, mid, lower = bollinger(constant_series, window=10)
        valid = upper.dropna().index
        # Zero std => bands collapse to SMA
        np.testing.assert_array_almost_equal(
            upper.loc[valid].values, lower.loc[valid].values
        )


# --- Momentum ---

class TestMomentum:
    def test_basic_shape(self, price_series):
        result = momentum(price_series, 10)
        assert len(result) == len(price_series)

    def test_first_values_nan(self, price_series):
        result = momentum(price_series, 10)
        assert result.iloc[:10].isna().all()

    def test_ascending_positive(self, price_series):
        result = momentum(price_series, 10).dropna()
        assert (result > 0).all()

    def test_known_value(self):
        s = pd.Series([100.0, 110.0, 120.0])
        result = momentum(s, 1)
        assert result.iloc[1] == pytest.approx(0.1)
        assert result.iloc[2] == pytest.approx(120.0 / 110.0 - 1.0)


# --- Log Momentum ---

class TestLogMomentum:
    def test_basic_shape(self, price_series):
        result = log_momentum(price_series, 10)
        assert len(result) == len(price_series)

    def test_first_values_nan(self, price_series):
        result = log_momentum(price_series, 10)
        assert result.iloc[:10].isna().all()

    def test_finite_values(self, price_series):
        result = log_momentum(price_series, 10).dropna()
        assert np.isfinite(result.values).all()

    def test_ascending_positive(self, price_series):
        result = log_momentum(price_series, 10).dropna()
        assert (result > 0).all()

    def test_log_of_one_is_zero(self):
        s = pd.Series([100.0, 100.0, 100.0])
        result = log_momentum(s, 1)
        assert result.iloc[1] == pytest.approx(0.0, abs=1e-6)
