"""Integration tests for the full training pipeline."""
import os
import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.train import build_features_safe, fit_transforms, run_one_split, evaluate_gan, _build_gan
from src.dataset import train_test_split_by_years, make_sequences, walk_forward_splits


@pytest.fixture
def pipeline_data(sample_panel, sample_config):
    """Prepare features and splits for integration tests (leakage-safe)."""
    train_panel, test_panel = train_test_split_by_years(sample_panel, sample_config.test_years)
    tr_feat, te_feat = build_features_safe(train_panel, test_panel, sample_config)
    tr_all, te_all = fit_transforms(tr_feat, te_feat, sample_config)
    return tr_all, te_all, sample_config


class TestFullPipeline:
    def test_single_split_lstm(self, pipeline_data):
        tr_all, te_all, cfg = pipeline_data
        cfg_lstm = Config(**{**vars(cfg), "generator": "lstm", "n_epochs": 2})
        met, yte, ypred, ylogit, yq, _ = run_one_split(tr_all, te_all, cfg_lstm)
        assert "MAE" in met
        assert "MAPE" in met
        assert "DirAcc" in met
        assert "PinballLoss" in met
        assert np.isfinite(met["MAE"])
        assert np.isfinite(met["PinballLoss"])
        assert len(ypred) == len(yte)
        assert yq.shape[1] == len(cfg_lstm.quantiles)

    def test_single_split_tst(self, pipeline_data):
        tr_all, te_all, cfg = pipeline_data
        cfg_tst = Config(**{**vars(cfg), "generator": "tst", "n_epochs": 2,
                            "d_model": 32, "nhead": 2, "num_layers_tst": 1})
        met, yte, ypred, ylogit, yq, _ = run_one_split(tr_all, te_all, cfg_tst)
        assert np.isfinite(met["MAE"])
        assert len(ypred) == len(yte)

    def test_walk_forward(self, sample_panel, sample_config):
        splits = list(walk_forward_splits(sample_panel, n_splits=2,
                                          min_train=100, step=50))
        assert len(splits) >= 1
        tr_panel = sample_panel.iloc[splits[0][0]]
        te_panel = sample_panel.iloc[splits[0][1]]
        tr_feat, te_feat = build_features_safe(tr_panel, te_panel, sample_config)
        tr_all, te_all = fit_transforms(tr_feat, te_feat, sample_config)
        cfg_wf = Config(**{**vars(sample_config), "n_epochs": 2})
        met, yte, ypred, ylogit, yq, _ = run_one_split(tr_all, te_all, cfg_wf)
        assert np.isfinite(met["MAE"])

    def test_output_csv_columns(self, pipeline_data):
        tr_all, te_all, cfg = pipeline_data
        cfg2 = Config(**{**vars(cfg), "n_epochs": 2})
        met, yte, ypred, ylogit, yq, _ = run_one_split(tr_all, te_all, cfg2)
        q_names = [f"q{int(q * 100)}" for q in cfg2.quantiles]
        out_data = {"y_true": yte, "y_pred": ypred, "y_logit": ylogit}
        for j, qn in enumerate(q_names):
            out_data[qn] = yq[:, j]
        df = pd.DataFrame(out_data)
        assert "y_true" in df.columns
        assert "y_pred" in df.columns
        assert "q10" in df.columns
        assert "q50" in df.columns
        assert "q90" in df.columns

    def test_evaluate_gan_shapes(self, pipeline_data):
        tr_all, te_all, cfg = pipeline_data
        Xtr, ytr = make_sequences(tr_all, target_col=cfg.ticker, seq_len=cfg.seq_len)
        Xte, yte = make_sequences(te_all, target_col=cfg.ticker, seq_len=cfg.seq_len)
        gan = _build_gan(cfg, input_size=Xtr.shape[-1], seq_len=Xtr.shape[1])
        yp, yl, yq = evaluate_gan(gan, Xte, yte, cfg.batch_size, cfg.quantiles)
        assert yp.shape == yte.shape
        assert yl.shape == yte.shape
        assert yq.shape == (len(yte), len(cfg.quantiles))

    def test_baselines_run(self):
        """Verify baseline benchmarks produce valid results."""
        from scripts.baselines import run_all_baselines
        y_true = np.random.randn(100).astype("float32")
        gan_pred = y_true + np.random.randn(100).astype("float32") * 0.1
        table = run_all_baselines(y_true, gan_pred=gan_pred)
        assert "Model" in table.columns
        assert "MAE" in table.columns
        assert "DirAcc" in table.columns
        assert len(table) >= 6  # at least 6 baselines + GAN
