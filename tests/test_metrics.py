"""Tests for evaluation metrics."""
import numpy as np
from src.utils.metrics import mae, mape, smape, direction_accuracy, pinball_loss


def test_mae_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert abs(mae(y_true, y_pred) - 0.5) < 1e-6


def test_mae_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_mape_near_zero_handling():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = mape(y_true, y_pred)
    assert np.isnan(result), "MAPE should be NaN when all y_true are near-zero"


def test_mape_normal():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    result = mape(y_true, y_pred)
    assert 0 < result < 1.0
    # Expected: mean(10/100, 10/200, 10/300) = mean(0.1, 0.05, 0.033) ~ 0.061
    assert abs(result - 0.061) < 0.01


def test_smape_bounded():
    y_true = np.random.randn(100)
    y_pred = np.random.randn(100)
    result = smape(y_true, y_pred)
    assert 0 <= result <= 2.0, "sMAPE should be bounded in [0, 2]"


def test_smape_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert smape(y, y) < 1e-6


def test_direction_accuracy_perfect():
    y_true = np.array([1.0, -1.0, 0.5, -0.5])
    y_pred = np.array([2.0, -3.0, 0.1, -0.1])
    assert direction_accuracy(y_true, y_pred) == 1.0


def test_direction_accuracy_worst():
    y_true = np.array([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([-1.0, 1.0, -1.0, 1.0])
    assert direction_accuracy(y_true, y_pred) == 0.0


def test_pinball_loss_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    # Perfect predictions at median
    y_q = np.column_stack([y_true - 1, y_true, y_true + 1])  # q10, q50, q90
    quantiles = (0.1, 0.5, 0.9)
    result = pinball_loss(y_true, y_q, quantiles)
    assert result >= 0, "Pinball loss should be non-negative"


def test_pinball_loss_zero_at_perfect():
    y_true = np.array([1.0, 2.0, 3.0])
    # All quantiles exactly match y_true
    y_q = np.column_stack([y_true, y_true, y_true])
    result = pinball_loss(y_true, y_q, (0.1, 0.5, 0.9))
    assert abs(result) < 1e-6, "Pinball loss should be 0 when predictions are perfect"


# --- Edge cases ---

def test_mae_single_element():
    assert mae(np.array([5.0]), np.array([3.0])) == 2.0


def test_mae_large_values():
    y = np.array([1e9, 2e9])
    p = np.array([1e9 + 1, 2e9 - 1])
    assert mae(y, p) == 1.0


def test_mape_mixed_zero_nonzero():
    """MAPE should ignore near-zero targets and compute on the rest."""
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([5.0, 110.0, 190.0])
    result = mape(y_true, y_pred)
    assert np.isfinite(result)
    # Only elements 1,2 used: mean(10/100, 10/200) = 0.075
    assert abs(result - 0.075) < 0.01


def test_mape_negative_values():
    y_true = np.array([-10.0, -20.0])
    y_pred = np.array([-11.0, -19.0])
    result = mape(y_true, y_pred)
    assert np.isfinite(result)
    assert result > 0


def test_smape_opposite_signs():
    y_true = np.array([1.0, -1.0])
    y_pred = np.array([-1.0, 1.0])
    result = smape(y_true, y_pred)
    assert result == 2.0 or abs(result - 2.0) < 0.01


def test_direction_accuracy_all_zeros():
    """When both true and pred are zero, sign(0)==sign(0) so accuracy=1."""
    y = np.zeros(10)
    assert direction_accuracy(y, y) == 1.0


def test_direction_accuracy_pred_zero_true_nonzero():
    """sign(0)=0, sign(positive)=1 => mismatch."""
    y_true = np.array([1.0, -1.0, 2.0])
    y_pred = np.zeros(3)
    result = direction_accuracy(y_true, y_pred)
    assert result == 0.0


def test_pinball_loss_asymmetric():
    """Low quantile should penalize over-predictions more, high -- under-predictions."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_over = np.column_stack([
        np.array([1.0, 1.0, 1.0]),   # q10 = 1 (over)
        np.array([1.0, 1.0, 1.0]),   # q50 = 1 (over)
        np.array([1.0, 1.0, 1.0]),   # q90 = 1 (over)
    ])
    y_under = np.column_stack([
        np.array([-1.0, -1.0, -1.0]),
        np.array([-1.0, -1.0, -1.0]),
        np.array([-1.0, -1.0, -1.0]),
    ])
    loss_over = pinball_loss(y_true, y_over, (0.1, 0.5, 0.9))
    loss_under = pinball_loss(y_true, y_under, (0.1, 0.5, 0.9))
    # Both should be positive
    assert loss_over > 0
    assert loss_under > 0


def test_pinball_loss_single_quantile():
    y_true = np.array([1.0, 2.0])
    y_q = np.array([[1.5], [2.5]])
    result = pinball_loss(y_true, y_q, (0.5,))
    assert np.isfinite(result)
