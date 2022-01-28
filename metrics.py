import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error

def RMSE(y_test,y_pred):
    rmse = mean_squared_error(y_true=y_test,y_pred=y_pred)
    return np.sqrt(rmse)

def MAE(y_test,y_pred):
    mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
    return mae
