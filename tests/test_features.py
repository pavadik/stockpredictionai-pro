"""Tests for feature engineering modules."""
import pandas as pd
import numpy as np


def test_fourier_approx_shape(sample_panel):
    from src.features.fourier import fourier_approx
    result = fourier_approx(sample_panel["GS"], k=10)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)
    assert not np.isnan(result).all()


def test_fourier_multi_shape(sample_panel):
    from src.features.fourier import fourier_multi
    result = fourier_multi(sample_panel["GS"], components=(3, 6, 9))
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (len(sample_panel), 3)
    assert "GS_fft3" in result.columns
    assert "GS_fft6" in result.columns
    assert "GS_fft9" in result.columns


def test_fourier_fit_transform_no_leakage(sample_panel):
    """Leakage-safe Fourier: fit on train, transform on train and test separately."""
    from src.features.fourier import fit_fourier_multi, transform_fourier_multi
    n = len(sample_panel)
    split = n // 2
    train_s = sample_panel["GS"].iloc[:split]
    test_s = sample_panel["GS"].iloc[split:]

    states = fit_fourier_multi(train_s, components=(3, 6))
    assert len(states) == 2

    tr_result = transform_fourier_multi(train_s.index, states)
    te_result = transform_fourier_multi(test_s.index, states)
    assert tr_result.shape == (split, 2)
    assert te_result.shape == (n - split, 2)
    assert "GS_fft3" in tr_result.columns
    assert "GS_fft6" in te_result.columns
    assert not np.isnan(tr_result.values).any()
    assert not np.isnan(te_result.values).any()


def test_arima_in_sample_shape(sample_panel):
    from src.features.arima_feat import arima_in_sample
    result = arima_in_sample(sample_panel["GS"], order=(5, 1, 0))
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)


def test_arima_fit_transform_no_leakage(sample_panel):
    """Leakage-safe ARIMA: fit on train, forecast on test."""
    from src.features.arima_feat import fit_arima, transform_arima
    n = len(sample_panel)
    split = n // 2
    train_s = sample_panel["GS"].iloc[:split]
    test_s = sample_panel["GS"].iloc[split:split + 20]  # small test for speed

    state = fit_arima(train_s, order=(5, 1, 0))
    train_feat, test_feat = transform_arima(train_s, test_s, state)

    assert isinstance(train_feat, pd.Series)
    assert isinstance(test_feat, pd.Series)
    assert len(train_feat) == len(train_s)
    assert len(test_feat) == len(test_s)
    assert not np.isnan(train_feat.values).all()
    assert not np.isnan(test_feat.values).all()


def test_arima_different_order(sample_panel):
    from src.features.arima_feat import arima_in_sample
    result = arima_in_sample(sample_panel["GS"].iloc[:200], order=(2, 1, 1))
    assert isinstance(result, pd.Series)
    assert len(result) == 200


def test_autoencoder_fit_transform_shapes(sample_panel):
    from src.features.autoencoder import fit_autoencoder, transform_autoencoder
    df = sample_panel.iloc[:200]
    ae_df, model = fit_autoencoder(df, hidden=32, bottleneck=16, epochs=2)
    assert ae_df.shape[0] == df.shape[0]
    assert ae_df.shape[1] == 16  # bottleneck size
    assert all(c.startswith("ae_") for c in ae_df.columns)
    # Transform
    ae_test = transform_autoencoder(df.iloc[:50], model)
    assert ae_test.shape == (50, 16)


def test_pca_fit_transform_shapes(sample_panel):
    from src.features.pca_eigen import fit_eigen, transform_eigen
    df = sample_panel.iloc[:200]
    pca_df, pca_model = fit_eigen(df, n_components=5)
    assert pca_df.shape == (200, 5)
    assert all(c.startswith("eig_") for c in pca_df.columns)
    # Transform
    pca_test = transform_eigen(df.iloc[:50], pca_model)
    assert pca_test.shape == (50, 5)


def test_sentiment_returns_none_without_model():
    """Sentiment should return None gracefully when model is unavailable."""
    from src.features.sentiment import get_ticker_sentiment
    result = get_ticker_sentiment("ZZZZZZZ_INVALID", "2020-01-01", "2020-06-30")
    assert result is None or isinstance(result, pd.Series)


def test_log_momentum(sample_panel):
    from src.utils.indicators import log_momentum
    result = log_momentum(sample_panel["GS"], window=10)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_panel)
    # First 10 values should be NaN due to shift
    assert result.iloc[:10].isna().all()
    # Rest should be finite
    assert np.isfinite(result.iloc[10:]).all()


# --- PCA / Eigen Portfolio ---

def test_eigen_portfolio_shape(sample_panel):
    from src.features.pca_eigen import eigen_portfolio
    df = sample_panel.select_dtypes(include=[np.number]).iloc[:200]
    result = eigen_portfolio(df, n_components=5)
    assert result.shape == (200, 5)
    assert all(c.startswith("eig_") for c in result.columns)


def test_eigen_portfolio_components_capped(sample_panel):
    """n_components cannot exceed number of features."""
    from src.features.pca_eigen import eigen_portfolio
    df = sample_panel.select_dtypes(include=[np.number]).iloc[:100, :3]
    result = eigen_portfolio(df, n_components=3)
    assert result.shape == (100, 3)


def test_pca_fit_transform_consistency(sample_panel):
    """PCA transform on train data should match fit_transform result."""
    from src.features.pca_eigen import fit_eigen, transform_eigen
    df = sample_panel.select_dtypes(include=[np.number]).iloc[:200]
    pca_df, pca_model = fit_eigen(df, n_components=5)
    pca_again = transform_eigen(df, pca_model)
    np.testing.assert_array_almost_equal(pca_df.values, pca_again.values)


# --- Autoencoder edge cases ---

def test_autoencoder_small_data():
    from src.features.autoencoder import fit_autoencoder, transform_autoencoder
    df = pd.DataFrame(np.random.randn(20, 5).astype("float32"),
                      columns=[f"f{i}" for i in range(5)])
    ae_df, model = fit_autoencoder(df, hidden=8, bottleneck=3, epochs=2, batch_size=8)
    assert ae_df.shape == (20, 3)
    ae_test = transform_autoencoder(df.iloc[:5], model, batch_size=8)
    assert ae_test.shape == (5, 3)


# --- Fourier edge cases ---

def test_fourier_large_k():
    """k larger than half series length should still work (wraps around)."""
    from src.features.fourier import fourier_approx
    s = pd.Series(np.random.randn(50), name="test")
    result = fourier_approx(s, k=40)
    assert len(result) == 50
    assert not np.isnan(result).all()


def test_fourier_fit_transform_extrapolation():
    """Transform on longer index should extrapolate without error."""
    from src.features.fourier import fit_fourier, transform_fourier
    train = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 100)), name="sin")
    state = fit_fourier(train, k=5)
    # Extrapolate to 150 points (50 beyond training)
    longer_index = pd.RangeIndex(150)
    result = transform_fourier(longer_index, state)
    assert len(result) == 150
    assert np.isfinite(result.values).all()
