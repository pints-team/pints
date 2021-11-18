#
# ABC Rejection method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints


class RejectionABC(pints.ABCSampler):
    r"""
    Implements the rejection ABC algorithm as described in [1].

    Here is a high-level description of the algorithm:

    .. math::
        \begin{align}
        \theta^* &\sim p(\theta) \\
        x &\sim p(x|\theta^*) \\
        \textrm{if } s(x) < \textrm{threshold}, \textrm{then} \\
        \theta^* \textrm{ is added to list of samples} \\
        \end{align}

    In other words, the first two steps sample parameters
    from the prior distribution and then sample data from the
    sampling distribution (assuming the sampled parameters).
    In the end, if the summary statistics are within the threshold,
    we add the sampled parameters to the list of samples.

    References
    ----------
    .. [1] "Approximate Bayesian Computation (ABC) in practice". Katalin
           Csillery, Michael G.B. Blum, Oscar E. Gaggiotti, Olivier Francois
           (2010) Trends in Ecology & Evolution
           https://doi.org/10.1016/j.tree.2010.04.001

    """
    def __init__(self, log_prior):

        self._log_prior = log_prior
        self._threshold = 1
        self._xs = None
        self._ready_for_tell = False

    def name(self):
        """ See :meth:`pints.ABCSampler.name()`. """
        return 'Rejection ABC'

    def ask(self, n_samples):
        """ See :meth:`ABCSampler.ask()`. """
        if self._ready_for_tell:
            raise RuntimeError('Ask called before tell.')
        self._xs = self._log_prior.sample(n_samples)

        self._ready_for_tell = True
        return self._xs

    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before ask.')
        self._ready_for_tell = False

        fx = pints.vector(fx)
        return self._xs[fx < self._threshold]

    def threshold(self):
        """
        Returns threshold error distance that determines if a sample is
        accepted (is ``error < threshold``).
        """
        return self._threshold

    def set_threshold(self, threshold):
        """
        Sets threshold error distance that determines if a sample is accepted
        (if ``error < threshold``).
        """
        x = float(threshold)
        if x <= 0:
            raise ValueError('Threshold must be greater than zero.')
        self._threshold = threshold
