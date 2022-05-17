import numpy as np
from tqdm import tqdm


class EllipticalSliceSampler:
    """Elliptical Slice Sampler Class

    The elliptical slice sampling algorithm is a Markov chain Monte Carlo
    approach to sampling from posterior distributions that consist of an
    arbitrary likelihood times a multivariate normal prior. The elliptical
    slice sampling algorithm is advantageous because it is conceptually simple
    and easy to implement and because it has no free parameters.

    The algorithm operates by randomly selecting a candidate from an ellipse
    defined by two vectors, one of which is assumed to be drawn from the target
    posterior and another that is an auxiliary random sample of a zero-mean
    multivariate normal. The algorithm iteratively shrinks the range from which
    candidates can be drawn until a candidate proposal is accepted.
    """

    def __init__(self, covariance, log_likelihood_func, seed=42):
        """Initialize the parameters of the elliptical slice sampler object.

        Parameters:
        mean (numpy array): A mean vector of a multivariate Gaussian.
        covariance (numpy array): A two-dimensional positive-definite
            covariance matrix of a multivariate Gaussian.
        log_likelihood_func (function): A log-likelihood function that maps
        a given sample (as its exclusive input) to a real number
        reflecting the log-likelihood of the observational data under
        the input parameter.
        """
        np.random.seed(seed)
        self.covariance = covariance
        self.log_likelihood_func = log_likelihood_func

    def __sample(self, f):
        """Internal Function that draws an individual sample according to the
        elliptical slice sampling routine. The input is drawn from the target
        distribution and the output is as well.

        Parameters:
        f (numpy array): A vector representing a parameter state that has
        been sampled from the target posterior distribution. Note that
        a sufficently high 'burnin' parameter can be leveraged to
        achieve a good mixin for this purpose.
        """
        # Choose the ellipse for this sampling iteration.
        nu = np.random.multivariate_normal(
            np.zeros(self.covariance.shape[:1]), self.covariance
        )
        # print("nu", nu, nu.shape)
        # Set the candidate acceptance treshold.
        log_y = self.log_likelihood_func(f) + np.log(np.random.uniform())
        # Set the bracket for selecting candidates on the ellipse.
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        theta_min, theta_max = theta - 2.0 * np.pi, theta

        # Iterates until a candidate is selected.
        while True:
            # Generates a point on the ellipse defined by 'nu' and the input. We
            # also compute the log-likelihood of the candidate and compare to
            # our threshold.
            fp = (f) * np.cos(theta) + nu * np.sin(theta)
            log_fp = self.log_likelihood_func(fp)
            if log_fp > log_y:
                return fp
            else:
                # If the candidate is not selected, shrink the bracket and
                # generate a new 'theta', which will yield a new candidate
                # point on the ellipse.
                if theta < 0.0:
                    theta_min = theta
                else:
                    theta_max = theta
                if theta_max - theta_min == 0.0:
                    print("should stop")
                theta = np.random.uniform(theta_min, theta_max)

    def sample(self, n_samples, initial_state, burnin=1000):
        """This function is user-facing and is used to generate a specified
        number of samples from the target distribution using elliptical slice
        sampling. The 'burnin' parameter defines how many iterations should be
        performed (and excluded) to achieve convergence to the target
        distribution.

        Parameters:
            n_samples (int): The number of samples to produce from this sampling
            routine.
            burnin (int, optional): The number of burnin iterations to perform.
            This is necessary to achieve samples that are representative of
            the true posterior and correctly characterize uncertainty.
        """
        # Compute the total number of samples.
        total_samples = n_samples + burnin
        # Initialize a matrix to store the samples. The first sample is chosen
        # to be a draw from the multivariate normal prior.
        samples = np.zeros((total_samples, self.covariance.shape[0]))
        samples[0] = (
            np.zeros(self.covariance.shape[:1])
            if initial_state is None
            else initial_state
        )
        for i in tqdm(range(1, total_samples)):
            samples[i] = self.__sample(samples[i - 1])
        return samples[burnin:]
