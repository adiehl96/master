import numpy as np
from scipy.special import expit


def bernoulli_log_likelihood(prob, data):
    """
    Source: Cond_llh_2array.m
    """
    return np.sum(data * np.log(prob) + (1 - data) * np.log(1 - prob))


def conditional_log_likelihood(W, X):
    """
    This function reflects the conditional log likelihood calculation present
    in the code accompanying the paper by Lloyd et al. (2012).
    Source: Cond_llh_2array.m
    Parameters:
    """
    p = expit(W)
    return bernoulli_log_likelihood(p, X)
