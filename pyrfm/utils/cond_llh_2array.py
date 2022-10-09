from Enumerations import ObservationModels
from scipy.special import expit
from sklearn.metrics import log_loss


def cond_llh_2array(W, X, ObsModel):
    if ObsModel == ObservationModels.Logit:
        p = expit(W)
        results = -log_loss(X, p, normalize=False, labels=[0, 1])
        return results
    if ObsModel == ObservationModels.Gaussian:
        raise Exception("Only Logit Observation Model Implemented")
    if ObsModel == ObservationModels.Poisson:
        raise Exception("Only Logit Observation Model Implemented")
