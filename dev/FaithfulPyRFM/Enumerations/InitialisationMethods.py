from enum import Enum


class InitialisationMethods(Enum):
    NONE = 0
    ResamplePseudo = 1
    MAPU = 2
    Both = 3
    PCA = 4
