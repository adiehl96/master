import numpy as np


def new_permutation(length, rng):
    perm = np.arange(length)
    rng.shuffle(perm)
    iperm = np.argsort(perm)

    return perm, iperm
