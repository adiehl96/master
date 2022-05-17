import numpy as np


class RBFKernel:
    def __init__(self, lengthscale, signal_variance):
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance

    def __call__(self, xi1, xi2):
        if len(xi1.shape) == 2:
            padded_xi1 = np.expand_dims(xi1, 1)
        else:
            padded_xi1 = xi1
        return np.square(self.signal_variance) * np.exp(
            -0.5
            * np.square(np.linalg.norm(padded_xi1 - xi2, axis=-1) / self.lengthscale)
        )


class LloydKernel:
    def __init__(self, lengthscale, signal_variance):
        self.lengthscale = lengthscale
        self.signal_variance = signal_variance
        self.k_hat = RBFKernel(lengthscale, signal_variance)

    def __call__(self, xi1, xi2):
        flipped_xi2 = np.flip(xi2, axis=-1)
        return 0.5 * (self.k_hat(xi1, xi2) + self.k_hat(xi1, flipped_xi2))
