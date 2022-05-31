from enum import Enum


class KernelPriors(Enum):
    LogNormals = 0
    InverseGammas = 1


class InitialisationMethods(Enum):
    NONE = 0
    ResamplePseudo = 1
    MAPU = 2
    Both = 3
    PCA = 4


class ObservationModels(Enum):
    Gaussian = 0
    Logit = 1
    Poisson = 2
