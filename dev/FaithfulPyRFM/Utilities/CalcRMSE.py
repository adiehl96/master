import numpy as np


def calc_rmse(predictions, truth):
    if truth.size != 0:
        rmse = np.sqrt(np.mean(np.square(predictions - truth)))
        return rmse
    else:
        return -1
