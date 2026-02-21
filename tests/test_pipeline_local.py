"""Integration tests for the full training pipeline with local MOEX data.

Uses synthetic fixture data in tests/fixtures/moex_sample/.
These tests verify the end-to-end flow: local data -> features -> GAN -> metrics.
"""
import os
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.data import build_panel_auto
from src.train import build_features_safe, fit_transforms, run_one_split
from src.dataset import train_test_split_by_years, make_sequences

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "moex_sample")


def _local_config(**overrides):
    """Create a minimal Config for local data pipeline tests."""
    defaults = dict(
        ticker="SBRF",
        start="2020-03-16",
        end="2020-03-18",
        data_source="local",
        data_path=FIXTURES_DIR,
        timeframe="M1",
        local_correlated=("GMKN",),
        test_years=1,
        n_epochs=2,
        ae_epochs=2,
        batch_size=16,
        seq_len=5,
        hidden_size=16,
        num_layers=1,
        d_model=16,
        nhead=2,
        num_layers_tst=1,
        pca_components=4,
        ae_hidden=16,
        ae_bottleneck=8,
        use_arima=False,
        use_sentiment=False,
        early_stopping_patience=0,
        num_workers=0,
    )
    defaults.update(overrides)
    return Config(**defaults)


class TestBuildPanelAuto:
    def test_local_dispatch(self):
        cfg = _local_config()
        panel = build_panel_auto(cfg)
        assert isinstance(panel, pd.DataFrame)
        assert "SBRF" in panel.columns
        assert len(panel) > 50

    def test_local_panel_has_indicators(self):
        cfg = _local_config()
        panel = build_panel_auto(cfg)
        for col in ["sma7", "rsi14", "macd", "bb_upper"]:
            assert col in panel.columns


def _split_panel(panel, test_frac=0.3):
    """Split panel by percentage (not by year), for small fixture data."""
    n = len(panel)
    cut = int(n * (1 - test_frac))
    return panel.iloc[:cut].copy(), panel.iloc[cut:].copy()


class TestLocalFeatures:
    @pytest.fixture
    def local_panel(self):
        cfg = _local_config()
        return build_panel_auto(cfg), cfg

    def test_build_features_no_arima(self, local_panel):
        panel, cfg = local_panel
        train_panel, test_panel = _split_panel(panel)
        tr_feat, te_feat = build_features_safe(train_panel, test_panel, cfg)
        assert len(tr_feat) > 0
        assert len(te_feat) > 0
        arima_cols = [c for c in tr_feat.columns if "arima" in c.lower()]
        assert len(arima_cols) == 0, "ARIMA should be disabled for M1"

    def test_fit_transforms(self, local_panel):
        panel, cfg = local_panel
        train_panel, test_panel = _split_panel(panel)
        tr_feat, te_feat = build_features_safe(train_panel, test_panel, cfg)
        tr_all, te_all = fit_transforms(tr_feat, te_feat, cfg)
        assert len(tr_all) > 0
        assert len(te_all) > 0
        assert not tr_all.isna().any().any()


class TestLocalPipeline:
    @pytest.fixture
    def prepared_data(self):
        cfg = _local_config()
        panel = build_panel_auto(cfg)
        train_panel, test_panel = _split_panel(panel)
        tr_feat, te_feat = build_features_safe(train_panel, test_panel, cfg)
        tr_all, te_all = fit_transforms(tr_feat, te_feat, cfg)
        if len(te_all) < cfg.seq_len + 2:
            pytest.skip("Not enough test data after features")
        return tr_all, te_all, cfg

    def test_single_split_lstm_local(self, prepared_data):
        tr_all, te_all, cfg = prepared_data
        cfg_lstm = Config(**{**vars(cfg), "generator": "lstm"})
        met, yte, ypred, ylogit, yq, _ = run_one_split(tr_all, te_all, cfg_lstm)
        assert "MAE" in met
        assert "DirAcc" in met
        assert np.isfinite(met["MAE"])
        assert len(ypred) == len(yte)
        assert yq.shape[1] == len(cfg_lstm.quantiles)

    def test_single_split_tst_local(self, prepared_data):
        tr_all, te_all, cfg = prepared_data
        cfg_tst = Config(**{**vars(cfg), "generator": "tst"})
        met, yte, ypred, ylogit, yq, _ = run_one_split(tr_all, te_all, cfg_tst)
        assert np.isfinite(met["MAE"])
        assert len(ypred) == len(yte)

    def test_make_sequences_with_local_data(self, prepared_data):
        tr_all, te_all, cfg = prepared_data
        Xtr, ytr = make_sequences(tr_all, target_col=cfg.ticker, seq_len=cfg.seq_len)
        assert Xtr.shape[0] > 0
        assert Xtr.shape[1] == cfg.seq_len
        assert Xtr.shape[2] == tr_all.shape[1]
        assert len(ytr) == len(Xtr)

    def test_output_metrics_keys(self, prepared_data):
        tr_all, te_all, cfg = prepared_data
        met, yte, ypred, ylogit, yq, _ = run_one_split(tr_all, te_all, cfg)
        expected_keys = {"MAE", "MAPE", "DirAcc", "PinballLoss"}
        assert set(met.keys()) == expected_keys
        for k, v in met.items():
            assert np.isfinite(v), f"Metric {k} is not finite: {v}"
