import numpy as np


def m_t(a):
    if len(a.shape) == 1:
        return np.expand_dims(a, 1)
    return a.T
