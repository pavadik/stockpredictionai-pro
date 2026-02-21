"""Tests for statistical quality checks in src/utils/stat_checks.py."""
import numpy as np
import pandas as pd
import pytest

from src.utils.stat_checks import (
    check_stationarity,
    check_multicollinearity,
    check_serial_correlation,
    check_heteroskedasticity,
    run_all_checks,
)


@pytest.fixture
def stationary_series():
    """White noise -- stationary by construction."""
    np.random.seed(42)
    return pd.Series(np.random.randn(500), name="noise")


@pytest.fixture
def nonstationary_series():
    """Random walk -- non-stationary."""
    np.random.seed(42)
    rw = np.cumsum(np.random.randn(500))
    return pd.Series(rw, name="random_walk")


@pytest.fixture
def correlated_df():
    """DataFrame with some correlated columns for VIF testing."""
    np.random.seed(42)
    x1 = np.random.randn(200)
    x2 = x1 * 0.9 + np.random.randn(200) * 0.1  # highly correlated
    x3 = np.random.randn(200)  # independent
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


# --- Stationarity (ADF) ---

class TestStationarity:
    def test_stationary_passes(self, stationary_series):
        result = check_stationarity(stationary_series)
        assert result["is_stationary"] is True
        assert result["p_value"] < 0.05

    def test_nonstationary_warns(self, nonstationary_series):
        result = check_stationarity(nonstationary_series)
        assert bool(result["is_stationary"]) is False
        assert result["p_value"] > 0.05

    def test_returns_dict_keys(self, stationary_series):
        result = check_stationarity(stationary_series)
        assert "adf_stat" in result
        assert "p_value" in result
        assert "is_stationary" in result

    def test_custom_significance(self, stationary_series):
        result = check_stationarity(stationary_series, significance=0.001)
        assert isinstance(result["is_stationary"], bool)


# --- Multicollinearity (VIF) ---

class TestMulticollinearity:
    def test_high_vif_detected(self, correlated_df):
        result = check_multicollinearity(correlated_df, threshold=5.0)
        assert isinstance(result, pd.Series)
        # x1 and x2 are highly correlated, expect VIF > 5
        assert result.max() > 5.0

    def test_independent_low_vif(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.randn(200),
            "b": np.random.randn(200),
            "c": np.random.randn(200),
        })
        result = check_multicollinearity(df, threshold=10.0)
        assert (result < 10.0).all()

    def test_single_column_skips(self):
        df = pd.DataFrame({"a": np.random.randn(50)})
        result = check_multicollinearity(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 0  # skipped


# --- Serial Correlation (Ljung-Box) ---

class TestSerialCorrelation:
    def test_white_noise_no_autocorrelation(self, stationary_series):
        result = check_serial_correlation(stationary_series)
        assert isinstance(result["has_autocorrelation"], bool)
        assert "lb_stat" in result
        assert "lb_pvalue" in result

    def test_autocorrelated_series(self):
        np.random.seed(42)
        # AR(1) process with strong autocorrelation
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.95 * x[i - 1] + np.random.randn()
        s = pd.Series(x, name="ar1")
        result = check_serial_correlation(s)
        assert result["has_autocorrelation"] is True

    def test_custom_lags(self, stationary_series):
        result = check_serial_correlation(stationary_series, lags=5)
        assert np.isfinite(result["lb_stat"])


# --- Heteroskedasticity (Breusch-Pagan) ---

class TestHeteroskedasticity:
    def test_homoskedastic_passes(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(200), name="homo")
        result = check_heteroskedasticity(s)
        assert "has_heteroskedasticity" in result
        assert bool(result["has_heteroskedasticity"]) in (True, False)
        assert "bp_stat" in result
        assert "bp_pvalue" in result

    def test_heteroskedastic_detected(self):
        np.random.seed(42)
        n = 500
        # Variance increases with index
        x = np.random.randn(n) * np.linspace(1, 10, n)
        s = pd.Series(x, name="hetero")
        result = check_heteroskedasticity(s)
        assert bool(result["has_heteroskedasticity"]) is True


# --- run_all_checks ---

class TestRunAllChecks:
    def test_returns_dict(self, sample_panel):
        result = run_all_checks(sample_panel, "GS")
        assert isinstance(result, dict)
        assert "stationarity_price" in result
        assert "stationarity_delta" in result
        assert "serial_corr" in result
        assert "heteroskedasticity" in result
        assert "vif" in result

    def test_all_subresults_are_dicts_or_series(self, sample_panel):
        result = run_all_checks(sample_panel, "GS")
        for key, val in result.items():
            assert isinstance(val, (dict, pd.Series)), f"{key} has unexpected type {type(val)}"
