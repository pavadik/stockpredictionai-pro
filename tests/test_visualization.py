"""Tests for visualization functions in src/utils/visualization.py.

These tests verify that plot functions create output files without errors.
They do NOT verify visual correctness (that requires manual inspection).
"""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.utils.visualization import (
    plot_predictions,
    plot_training_curves,
    plot_technical_indicators,
    plot_fourier_components,
    _OUT,
)


@pytest.fixture(autouse=True)
def use_temp_output(tmp_path, monkeypatch):
    """Redirect plot outputs to a temp dir so tests don't pollute real outputs/."""
    monkeypatch.setattr("src.utils.visualization._OUT", str(tmp_path))
    return tmp_path


class TestPlotPredictions:
    def test_creates_file(self, use_temp_output):
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        plot_predictions(y_true, y_pred, filename="test_pred.png")
        assert os.path.exists(os.path.join(str(use_temp_output), "test_pred.png"))

    def test_with_quantiles(self, use_temp_output):
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        y_q = np.random.randn(100, 3)
        plot_predictions(y_true, y_pred, y_q=y_q,
                         quantiles=(0.1, 0.5, 0.9), filename="test_q.png")
        assert os.path.exists(os.path.join(str(use_temp_output), "test_q.png"))

    def test_without_quantiles(self, use_temp_output):
        y_true = np.random.randn(50)
        y_pred = np.random.randn(50)
        plot_predictions(y_true, y_pred, y_q=None, filename="test_noq.png")
        assert os.path.exists(os.path.join(str(use_temp_output), "test_noq.png"))

    def test_single_sample(self, use_temp_output):
        plot_predictions(np.array([1.0]), np.array([1.1]), filename="test_single.png")
        assert os.path.exists(os.path.join(str(use_temp_output), "test_single.png"))


class TestPlotTrainingCurves:
    def test_creates_file(self, use_temp_output):
        g = [1.0, 0.8, 0.6, 0.5]
        d = [0.5, 0.4, 0.3, 0.25]
        plot_training_curves(g, d, filename="test_curves.png")
        assert os.path.exists(os.path.join(str(use_temp_output), "test_curves.png"))

    def test_with_val_maes(self, use_temp_output):
        g = [1.0, 0.8, 0.6]
        d = [0.5, 0.4, 0.3]
        v = [2.0, 1.5, 1.2]
        plot_training_curves(g, d, val_maes=v, filename="test_curves_val.png")
        assert os.path.exists(os.path.join(str(use_temp_output), "test_curves_val.png"))

    def test_single_epoch(self, use_temp_output):
        plot_training_curves([0.5], [0.3], filename="test_single_epoch.png")
        assert os.path.exists(os.path.join(str(use_temp_output), "test_single_epoch.png"))


class TestPlotTechnicalIndicators:
    def test_creates_file(self, sample_panel, use_temp_output):
        plot_technical_indicators(sample_panel, "GS", last_days=50,
                                  filename="test_tech.png")
        assert os.path.exists(os.path.join(str(use_temp_output), "test_tech.png"))

    def test_fewer_days_than_requested(self, sample_panel, use_temp_output):
        """Should not crash if last_days > panel length."""
        plot_technical_indicators(sample_panel, "GS", last_days=99999,
                                  filename="test_tech_large.png")
        assert os.path.exists(os.path.join(str(use_temp_output), "test_tech_large.png"))


class TestPlotFourierComponents:
    def test_creates_file(self, sample_panel, use_temp_output):
        plot_fourier_components(sample_panel["GS"], components=(3, 6),
                                filename="test_fft.png")
        assert os.path.exists(os.path.join(str(use_temp_output), "test_fft.png"))

    def test_single_component(self, sample_panel, use_temp_output):
        plot_fourier_components(sample_panel["GS"], components=(3,),
                                filename="test_fft1.png")
        assert os.path.exists(os.path.join(str(use_temp_output), "test_fft1.png"))
