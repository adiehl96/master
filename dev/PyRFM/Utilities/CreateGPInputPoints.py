import numpy as np


def create_gp_input_points(ii, jj, u, v=None):
    if v is None:
        return np.column_stack((u[ii], u[jj]))
    else:
        return [u[ii], v[jj]]
