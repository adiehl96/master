import numpy as np


def gppu_elliptical(xx, chol_sigma, log_like_fn, rng, angle_range=0):
    cur_log_like = log_like_fn(xx)

    dimension = len(xx)
    if chol_sigma.shape != (dimension, dimension):
        raise Exception("chol_sigma has the wrong dimension")

    nu = (chol_sigma.T @ rng.standard_normal((dimension))).reshape(xx.shape)
    hh = np.log(rng.uniform()) + cur_log_like

    if angle_range <= 0:
        phi = rng.uniform() * 2 * np.pi
        phi_min = phi - 2 * np.pi
        phi_max = phi
    else:
        phi_min = -angle_range * rng.uniform()
        phi_max = phi_min + angle_range
        phi = rng.uniform() * (phi_max - phi_min) + phi_min

    while True:
        xx_prop = xx * np.cos(phi) + nu * np.sin(phi)
        cur_log_like = log_like_fn(xx_prop)
        if cur_log_like > hh:
            return xx_prop
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise Exception(
                "BUG DETECTED: Shrunk to current position and still not acceptable."
            )
        phi = rng.uniform() * (phi_max - phi_min) + phi_min
