import numpy as np
import scipy.stats as stats


class PerturbationKernel:
    """
    Abstract base class for ABC Perturbation Kernels
    """

    def perturb(self, theta):
        """
        Returns a point theta* from the proposal distribution K_t(theta*|theta)
        """
        return NotImplementedError

    def p(self, x, y):
        """
        Returns the probability of obtaining x, given that x~K_t(.|y)
        """
        return NotImplementedError


class SphericalGaussianKernel(PerturbationKernel):
    """
    A Naïve perturbation kernel which uses a Multivariate Gaussian with diagonal covariance matrix
    With all non-zero entries set to the single passed in value
    """

    def __init__(self, variance):
        self._variance = variance

    def perturb(self, theta):
        n_parameters = len(theta)
        ret = theta + np.random.multivariate_normal(np.zeros(n_parameters),
                                                     np.diag(np.full(n_parameters, self._variance)))
        return ret

    def p(self, x, y):
        if len(x) != len(y):
            raise ValueError("Target and given parameter vectors must be of the same length")
        x_star = x - y
        return stats.multivariate_normal.pdf([x_star], mean=np.zeros(len(x)),
                                             cov=np.diag(np.full(len(x), self._variance)))
