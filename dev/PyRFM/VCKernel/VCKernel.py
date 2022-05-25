import numpy as np
from Enumerations.KernelPriors import KernelPriors


class VCKernel:
    def __init__(self):
        self.name = "covSEiso"  # Name of GPML kernel function
        self.params = [np.log(1), np.log(2)]  # Parameters to be passed to GPML

        self.diag_noise = np.log(0.1)  # Diagonal noise - part of prior
        self.jitter = 10e-6  # Diagonal noise  - for numerical stability

        self.prior_type = KernelPriors.LogNormals  # Form of prior
        self.prior_params = [
            [np.log(1), 0.5],
            [np.log(2), 0.5],
        ]  # Form depends on prior type
        self.noise_params = [np.log(0.1), 0.5]  # Parameters for prior on noise

    def prior(self):  # Prior for parameters
        # print("self.noise_params")
        # print(self.noise_params)
        if self.prior_type == KernelPriors.LogNormals:
            llh = 0
            # print("llh inside")
            # print(llh)
            for idx, param in enumerate(self.params):
                llh = llh - 0.5 * np.square(
                    param - self.prior_params[idx, 0]
                ) / np.square(self.prior_params[idx, 1])
            # print("llh inside")
            # print(llh)
            llh = llh - 0.5 * np.square(
                self.diag_noise - self.noise_params[0]
            ) / np.square(self.noise_params[1])
            # print("llh inside")
            # print(llh)
            return llh

    def matrix(self, x, z=None):  # Calculate kernel matrix
        # print("self.params")
        # print(self.params)
        # print("self.params[0]", self.params[0])
        # print("self.params[1]", self.params[1])
        if z is None:
            matrix = lloyd_kernel_matrix(x, x, self.params[0], self.params[1])
            matrix = matrix + (
                self.jitter * np.max(matrix) + np.exp(2 * self.diag_noise)
            ) * np.eye(len(matrix))
            return matrix
        else:
            matrix = lloyd_kernel_matrix(x, z, self.params[0], self.params[1])
            return matrix


def rbf_kernel_matrix(xi1, xi2, lengthscale, signal_variance):
    if len(xi1.shape) == 2:
        padded_xi1 = np.expand_dims(xi1, 1)
    else:
        padded_xi1 = xi1
    return np.square(signal_variance) * np.exp(
        -0.5 * np.square(np.linalg.norm(padded_xi1 - xi2, axis=-1) / lengthscale)
    )


def lloyd_kernel_matrix(xi1, xi2, log_lengthscale, log_signal_variance):
    """
    This kernel function tries to replicate
    source: covSEiso_sym.m
    """
    lengthscale = np.exp(log_lengthscale)
    signal_variance = np.exp(log_signal_variance)
    flipped_xi2 = np.flip(xi2, axis=-1)
    return 0.5 * (
        rbf_kernel_matrix(xi1, xi2, lengthscale, signal_variance)
        + rbf_kernel_matrix(xi1, flipped_xi2, lengthscale, signal_variance)
    )
