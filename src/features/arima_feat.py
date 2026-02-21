import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple


def arima_in_sample(series: pd.Series, order=(5, 1, 0)) -> pd.Series:
    """In-sample ARIMA approximation (uses full series -- look-ahead bias).

    WARNING: kept for quick experiments / visualization only.
    For train/test safe version use ``fit_arima`` + ``transform_arima``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(series, order=order)
        res = model.fit()
        pred = res.predict(start=series.index[0], end=series.index[-1])
    pred.name = f"{series.name}_arima_{order[0]}{order[1]}{order[2]}"
    return pred


# ---------------------------------------------------------------------------
# Leakage-safe ARIMA: fit on train, forecast on test
# ---------------------------------------------------------------------------

def fit_arima(train_series: pd.Series, order=(5, 1, 0)):
    """Fit ARIMA on *train_series* and return the fitted result object.

    Returns (result, order, series_name) tuple for use with ``transform_arima``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(train_series, order=order)
        res = model.fit()
    return res, order, train_series.name


def transform_arima(train_series: pd.Series, test_series: pd.Series,
                    arima_state) -> Tuple[pd.Series, pd.Series]:
    """Produce ARIMA features for both train and test without data leakage.

    - Train part: in-sample fitted values from the model.
    - Test part: rolling one-step-ahead forecasts (the model is re-estimated
      incrementally with ``append`` to avoid look-ahead bias while still
      adapting to recent data).

    Args:
        train_series: target price series (train portion).
        test_series: target price series (test portion).
        arima_state: tuple returned by ``fit_arima``.

    Returns:
        (train_feature, test_feature) -- two pd.Series aligned with inputs.
    """
    res, order, name = arima_state
    col_name = f"{name}_arima_{order[0]}{order[1]}{order[2]}"

    # Train: in-sample predictions (no leakage -- model was fit on same data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_pred = res.predict(start=train_series.index[0],
                                 end=train_series.index[-1])
    train_feat = pd.Series(train_pred.values, index=train_series.index,
                           name=col_name)

    # Test: rolling one-step-ahead forecasts
    test_preds = []
    current_res = res
    for i in range(len(test_series)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = current_res.forecast(steps=1)
            test_preds.append(float(fc.iloc[0]))
            # Append the actual observation so next forecast uses real data
            try:
                current_res = current_res.append(
                    [test_series.iloc[i]], refit=False
                )
            except Exception:
                # Fallback: if append fails, keep using last model
                pass

    test_feat = pd.Series(test_preds, index=test_series.index, name=col_name)
    return train_feat, test_feat
