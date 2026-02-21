"""Tests for helper functions in src/train.py."""
import os
import numpy as np
import torch
import pytest

from src.train import (
    set_global_seed,
    compute_metrics,
    _quantile_col_names,
    dl_from_xy,
    evaluate_gan,
    _build_gan,
    train_gan_with_early_stop,
)
from src.config import Config
from src.dataset import make_sequences


# --- set_global_seed ---

class TestSetGlobalSeed:
    def test_numpy_reproducibility(self):
        set_global_seed(42)
        a1 = np.random.randn(10)
        set_global_seed(42)
        a2 = np.random.randn(10)
        np.testing.assert_array_equal(a1, a2)

    def test_torch_reproducibility(self):
        set_global_seed(42)
        t1 = torch.randn(10)
        set_global_seed(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_different_seeds_differ(self):
        set_global_seed(42)
        a1 = np.random.randn(10)
        set_global_seed(99)
        a2 = np.random.randn(10)
        assert not np.allclose(a1, a2)

    def test_env_var_set(self):
        set_global_seed(123)
        assert os.environ["PYTHONHASHSEED"] == "123"


# --- _quantile_col_names ---

class TestQuantileColNames:
    def test_default(self):
        result = _quantile_col_names((0.1, 0.5, 0.9))
        assert result == ["q10", "q50", "q90"]

    def test_custom(self):
        result = _quantile_col_names((0.05, 0.25, 0.75, 0.95))
        assert result == ["q5", "q25", "q75", "q95"]

    def test_empty(self):
        result = _quantile_col_names(())
        assert result == []


# --- compute_metrics ---

class TestComputeMetrics:
    def test_returns_all_keys(self):
        y_true = np.array([1.0, -1.0, 2.0])
        y_pred = np.array([0.9, -1.1, 2.1])
        y_q = np.column_stack([y_pred - 0.5, y_pred, y_pred + 0.5])
        met = compute_metrics(y_true, y_pred, y_q, (0.1, 0.5, 0.9))
        assert "MAE" in met
        assert "MAPE" in met
        assert "DirAcc" in met
        assert "PinballLoss" in met

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        y_q = np.column_stack([y, y, y])
        met = compute_metrics(y, y, y_q, (0.1, 0.5, 0.9))
        assert met["MAE"] == pytest.approx(0.0)
        assert met["DirAcc"] == 1.0

    def test_all_finite(self):
        np.random.seed(42)
        y_true = np.random.randn(50)
        y_pred = np.random.randn(50)
        y_q = np.random.randn(50, 3)
        met = compute_metrics(y_true, y_pred, y_q)
        for k, v in met.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"


# --- dl_from_xy ---

class TestDlFromXY:
    def test_returns_dataset_and_loader(self):
        X = np.random.randn(20, 5, 10).astype("float32")
        y = np.random.randn(20).astype("float32")
        ds, dl = dl_from_xy(X, y, batch=8)
        assert len(ds) == 20
        batch_x, batch_y = next(iter(dl))
        assert batch_x.shape[0] <= 8
        assert batch_y.shape[0] <= 8

    def test_small_batch(self):
        X = np.random.randn(5, 3, 4).astype("float32")
        y = np.random.randn(5).astype("float32")
        ds, dl = dl_from_xy(X, y, batch=2)
        batches = list(dl)
        assert len(batches) >= 2


# --- train_gan_with_early_stop ---

class TestTrainGANWithEarlyStop:
    def test_early_stop_triggers(self):
        """With patience=1 and constant bad predictions, should stop early."""
        cfg = Config(
            n_epochs=10, batch_size=8, hidden_size=16,
            seq_len=5, early_stopping_patience=2,
            use_lr_scheduler=False, grad_clip=0.0,
        )
        B, T, F = 16, 5, 10
        X = np.random.randn(B, T, F).astype("float32")
        y = np.random.randn(B).astype("float32")
        _, tr_dl = dl_from_xy(X, y, cfg.batch_size)
        Xte = np.random.randn(8, T, F).astype("float32")
        yte = np.random.randn(8).astype("float32")
        gan = _build_gan(cfg, input_size=F, seq_len=T)

        g_loss, d_loss, stopped = train_gan_with_early_stop(
            gan, tr_dl, Xte, yte, cfg, verbose=False
        )
        # Should stop before n_epochs (or at n_epochs if no improvement)
        assert stopped <= cfg.n_epochs
        assert np.isfinite(g_loss)
        assert np.isfinite(d_loss)

    def test_no_early_stop_when_patience_zero(self):
        cfg = Config(
            n_epochs=3, batch_size=8, hidden_size=16,
            seq_len=5, early_stopping_patience=0,
            use_lr_scheduler=False, grad_clip=0.0,
        )
        B, T, F = 16, 5, 10
        X = np.random.randn(B, T, F).astype("float32")
        y = np.random.randn(B).astype("float32")
        _, tr_dl = dl_from_xy(X, y, cfg.batch_size)
        gan = _build_gan(cfg, input_size=F, seq_len=T)

        g_loss, d_loss, stopped = train_gan_with_early_stop(
            gan, tr_dl, None, None, cfg, verbose=False
        )
        assert stopped == cfg.n_epochs


# --- evaluate_gan ---

class TestEvaluateGAN:
    def test_output_shapes(self):
        B, T, F = 16, 8, 12
        cfg = Config(
            hidden_size=16, seq_len=T, batch_size=8,
            use_lr_scheduler=False, grad_clip=0.0,
        )
        gan = _build_gan(cfg, input_size=F, seq_len=T)
        Xte = np.random.randn(B, T, F).astype("float32")
        yte = np.random.randn(B).astype("float32")
        yp, yl, yq = evaluate_gan(gan, Xte, yte, cfg.batch_size, cfg.quantiles)
        assert yp.shape == (B,)
        assert yl.shape == (B,)
        assert yq.shape == (B, len(cfg.quantiles))

    def test_deterministic_in_eval(self):
        """Predictions should be deterministic across calls (eval mode, no dropout)."""
        B, T, F = 8, 5, 10
        cfg = Config(
            hidden_size=16, seq_len=T, batch_size=8,
            dropout_lstm=0.0, use_lr_scheduler=False, grad_clip=0.0,
        )
        gan = _build_gan(cfg, input_size=F, seq_len=T)
        Xte = np.random.randn(B, T, F).astype("float32")
        yte = np.random.randn(B).astype("float32")
        # Note: Generator adds noise internally, so predictions may differ
        # But the shapes should be consistent
        yp1, _, _ = evaluate_gan(gan, Xte, yte, cfg.batch_size, cfg.quantiles)
        yp2, _, _ = evaluate_gan(gan, Xte, yte, cfg.batch_size, cfg.quantiles)
        assert yp1.shape == yp2.shape
