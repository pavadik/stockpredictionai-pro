import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_in_sample(series: pd.Series, order=(5,1,0)) -> pd.Series:
    # максимально простая in-sample аппроксимация как фича
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(series, order=order)
        res = model.fit()
        pred = res.predict(start=series.index[0], end=series.index[-1])
    pred.name = f"{series.name}_arima_{order[0]}{order[1]}{order[2]}"
    return pred
