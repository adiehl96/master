import numpy as np
from Enumerations import KernelPriors


def matrix(kernel_params, x, z=None):  # Calculate kernel matrix
    jitter = 10e-6
    if z is None:
        matrix = lloyd_kernel_matrix(x, x, kernel_params["lls"], kernel_params["lsv"])
        if isinstance(matrix, np.ndarray):
            matrix = matrix + (
                jitter * np.max(matrix) + np.exp(2 * kernel_params["ldn"])
            ) * np.eye(len(matrix))
        elif isinstance(matrix, np.float64):
            matrix = matrix + (jitter * matrix + np.exp(2 * kernel_params["ldn"]))
        else:
            raise Exception("Unknown kernel output type")
        return matrix
    else:
        matrix = lloyd_kernel_matrix(x, z, kernel_params["lls"], kernel_params["lsv"])
        return matrix


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


def rbf_kernel_matrix(xi1, xi2, lengthscale, signal_variance):
    if len(xi1.shape) == 2:
        padded_xi1 = np.expand_dims(xi1, 1)
    else:
        padded_xi1 = xi1
    return np.square(signal_variance) * np.exp(
        -0.5 * np.square(np.linalg.norm(padded_xi1 - xi2, axis=-1) / lengthscale)
    )


def kernel_prior_llh(prior_type, kernel_priors, kernel_params):
    if prior_type == KernelPriors.LogNormals:
        llh = 0
        for prior, param in zip(
            kernel_priors,
            [kernel_params["lls"], kernel_params["lsv"], kernel_params["ldn"]],
        ):
            llh = llh - 0.5 * np.square(param - prior[0]) / np.square(prior[1])
        return llh
