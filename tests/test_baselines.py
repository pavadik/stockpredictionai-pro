"""Tests for baseline benchmark strategies in scripts/baselines.py."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.baselines import (
    naive_persistence,
    random_walk,
    mean_reversion,
    momentum_baseline,
    sma_baseline,
    evaluate_baseline,
    run_all_baselines,
)


@pytest.fixture
def sample_y():
    np.random.seed(42)
    return np.random.randn(200).astype("float32")


# --- Individual baselines ---

class TestNaivePersistence:
    def test_all_zeros(self, sample_y):
        pred = naive_persistence(sample_y)
        assert pred.shape == sample_y.shape
        assert (pred == 0).all()


class TestRandomWalk:
    def test_shape(self, sample_y):
        pred = random_walk(sample_y, seed=42)
        assert pred.shape == sample_y.shape

    def test_deterministic_with_seed(self, sample_y):
        p1 = random_walk(sample_y, seed=123)
        p2 = random_walk(sample_y, seed=123)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self, sample_y):
        p1 = random_walk(sample_y, seed=1)
        p2 = random_walk(sample_y, seed=2)
        assert not np.allclose(p1, p2)

    def test_std_matches(self, sample_y):
        pred = random_walk(sample_y, seed=42)
        # Std of predictions should be roughly same as std of true
        assert abs(np.std(pred) - np.std(sample_y)) / np.std(sample_y) < 0.3


class TestMeanReversion:
    def test_shape(self, sample_y):
        pred = mean_reversion(sample_y)
        assert pred.shape == sample_y.shape

    def test_first_is_zero(self, sample_y):
        pred = mean_reversion(sample_y)
        assert pred[0] == 0.0

    def test_contrarian(self, sample_y):
        pred = mean_reversion(sample_y)
        # pred[i] == -y[i-1] for i > 0
        np.testing.assert_array_almost_equal(pred[1:], -sample_y[:-1])


class TestMomentumBaseline:
    def test_shape(self, sample_y):
        pred = momentum_baseline(sample_y)
        assert pred.shape == sample_y.shape

    def test_first_is_zero(self, sample_y):
        pred = momentum_baseline(sample_y)
        assert pred[0] == 0.0

    def test_trend_following(self, sample_y):
        pred = momentum_baseline(sample_y)
        # pred[i] == y[i-1]
        np.testing.assert_array_almost_equal(pred[1:], sample_y[:-1])


class TestSMABaseline:
    def test_shape(self, sample_y):
        pred = sma_baseline(sample_y, window=5)
        assert pred.shape == sample_y.shape

    def test_first_window_is_zero(self, sample_y):
        pred = sma_baseline(sample_y, window=10)
        assert (pred[:10] == 0.0).all()

    def test_known_value(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pred = sma_baseline(y, window=3)
        # pred[3] = mean(y[0:3]) = mean(1,2,3) = 2.0
        assert pred[3] == pytest.approx(2.0)
        # pred[5] = mean(y[2:5]) = mean(3,4,5) = 4.0
        assert pred[5] == pytest.approx(4.0)

    def test_window_larger_than_data(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = sma_baseline(y, window=10)
        assert (pred == 0.0).all()


# --- evaluate_baseline ---

class TestEvaluateBaseline:
    def test_returns_all_keys(self, sample_y):
        pred = naive_persistence(sample_y)
        result = evaluate_baseline("test", sample_y, pred)
        assert "Model" in result
        assert "MAE" in result
        assert "MAPE" in result
        assert "sMAPE" in result
        assert "DirAcc" in result
        assert "PinballLoss" in result

    def test_perfect_prediction(self):
        y = np.array([1.0, -1.0, 2.0, -2.0])
        result = evaluate_baseline("perfect", y, y)
        assert result["MAE"] == pytest.approx(0.0)

    def test_naive_dir_acc(self, sample_y):
        pred = naive_persistence(sample_y)
        result = evaluate_baseline("naive", sample_y, pred)
        # sign(0) == 0 which mismatches sign(nonzero), so DirAcc is low
        assert 0.0 <= result["DirAcc"] <= 1.0


# --- run_all_baselines ---

class TestRunAllBaselines:
    def test_returns_dataframe(self, sample_y):
        table = run_all_baselines(sample_y)
        assert hasattr(table, "columns")
        assert "Model" in table.columns
        assert len(table) >= 6  # at least 6 baseline strategies

    def test_with_gan_predictions(self, sample_y):
        gan_pred = sample_y * 0.9
        table = run_all_baselines(sample_y, gan_pred=gan_pred)
        assert "GAN" in table["Model"].values
        assert len(table) >= 7

    def test_with_gan_quantiles(self, sample_y):
        gan_pred = sample_y * 0.9
        gan_q = np.column_stack([sample_y * 0.8, sample_y * 0.9, sample_y * 1.0])
        table = run_all_baselines(sample_y, gan_pred=gan_pred, gan_q=gan_q)
        gan_row = table[table["Model"] == "GAN"]
        assert not gan_row.empty
        assert np.isfinite(gan_row["PinballLoss"].iloc[0])

    def test_sorted_by_mae(self, sample_y):
        table = run_all_baselines(sample_y)
        maes = table["MAE"].values
        assert all(maes[i] <= maes[i + 1] for i in range(len(maes) - 1))

    def test_all_maes_finite(self, sample_y):
        table = run_all_baselines(sample_y)
        assert table["MAE"].apply(np.isfinite).all()
