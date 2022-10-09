import numpy as np
from Enumerations import ObservationModels
from scipy.special import expit

def bernoulli_llh(prob, data):
    """
    from sklearn.metrics import log_loss
    results = -log_loss(X, p, normalize=False, labels=[0, 1])
    """
    eps = np.finfo(np.float64).eps
    prob[prob < eps] = eps
    prob[prob > (1 - eps)] = 1 - eps
    results = np.sum(data @ np.log(prob) + (1 - data) @ np.log(1 - prob))
    return results


def cond_llh_2array(W, X, ObsModel):
    if ObsModel == ObservationModels.Logit:
        p = expit(W)
        return bernoulli_llh(p, X)
    if ObsModel == ObservationModels.Gaussian:
        raise Exception("Only Logit Observation Model Implemented")
    if ObsModel == ObservationModels.Poisson:
        raise Exception("Only Logit Observation Model Implemented")
