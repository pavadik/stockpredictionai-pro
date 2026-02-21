import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error.

    Uses |y_true| in denominator to avoid sign-flip explosion,
    and filters out near-zero targets where MAPE is undefined.
    """
    mask = np.abs(y_true) > 1e-6
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))))


def smape(y_true, y_pred):
    """Symmetric MAPE -- bounded in [0, 2], more robust than MAPE for near-zero values."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-9
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def direction_accuracy(y_true, y_pred):
    """Fraction of samples where predicted direction (sign) matches true direction."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def multi_horizon_metrics(y_true_multi, y_pred_multi, horizons):
    """Compute per-horizon MAE and DirAcc for multi-horizon predictions.

    Args:
        y_true_multi: [N, H] true deltas at each horizon.
        y_pred_multi: [N, H] predicted deltas.
        horizons: tuple of horizon values, e.g. (1, 2, 4).
    Returns:
        dict with per-horizon metrics, e.g. {"MAE_h1": ..., "DirAcc_h1": ...}
    """
    result = {}
    for i, h in enumerate(horizons):
        yt = y_true_multi[:, i]
        yp = y_pred_multi[:, i]
        result[f"MAE_h{h}"] = mae(yt, yp)
        result[f"DirAcc_h{h}"] = direction_accuracy(yt, yp)
    result["MAE_avg"] = float(np.mean([result[f"MAE_h{h}"] for h in horizons]))
    result["DirAcc_avg"] = float(np.mean([result[f"DirAcc_h{h}"] for h in horizons]))
    return result


def pinball_loss(y_true, y_quantiles, quantiles=(0.1, 0.5, 0.9)):
    """Pinball (quantile) loss for evaluation.

    Args:
        y_true: 1-D array of true values, shape [N].
        y_quantiles: 2-D array of quantile predictions, shape [N, Q].
        quantiles: tuple of Q quantile levels.
    Returns:
        Average pinball loss across all quantiles.
    """
    total = 0.0
    for i, q in enumerate(quantiles):
        err = y_true - y_quantiles[:, i]
        total += float(np.mean(np.maximum(q * err, (q - 1) * err)))
    return total / len(quantiles)
