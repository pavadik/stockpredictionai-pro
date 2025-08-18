import numpy as np

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))))

def direction_accuracy(y_true, y_pred):
    # учитываем знак изменения (вверх/вниз)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))
