from Enumerations import ObservationModels
from scipy.special import expit
from utils.bernoulli_llh import bernoulli_llh


def cond_llh_2array(W, X, ObsModel):
    if ObsModel == ObservationModels.Logit:
        p = expit(W)
        return bernoulli_llh(p, X)
    if ObsModel == ObservationModels.Gaussian:
        raise Exception("Only Logit Observation Model Implemented")
    if ObsModel == ObservationModels.Poisson:
        raise Exception("Only Logit Observation Model Implemented")
