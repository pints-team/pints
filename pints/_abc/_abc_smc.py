#
# ABC SMC method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import numpy as np


class ABCSMC(pints.ABCSampler):
    r"""
    Implements the ABC-SMC algorithm as describd in [1].

    In each iteration of the algorithm, the following steps occur:
    .. math::
        \begin{align}
        & \theta^* \sim p_{t-1}(\theta) \textrm{, i.e. sample parameters from
        previous intermediate distribution} \\
        & \theta^{**} \sim K(\theta|\theta^{*}), \textrm{i.e. perturb }
        \theta^{*} \textrm{ to     obtain to new point } x \sim
        p(x|\theta^{**})\textrm{, i.e. sample data from sampling
        distribution} \\
        & \textrm{if } s(x) < \textrm{threshold}_(t), \theta^* \textrm{
        added to list of samples[t]}
        \end{align}

    After we have obtained nr_samples samples, t is advanced, and weights
    are calculated for samples[t-1]. At the last error threshold,
    samples are returned whenever they are accepted. This algorithm is
    also referred to as ABC Population Monte Carlo (ABC PMC) [2].

    References
    ----------
    .. [1] "Toni, Tina, et al. Approximate Bayesian computation scheme
            for parameter inference and model selection in dynamical systems.
            Journal of the Royal Society Interface, 6.31: 187-202, 2009.
            https://doi.org/10.1098/rsif.2008.0172


    .. [2] "Beaumont, Mark A., et al. Adaptive approximate Bayesian
            computation. Biometrika, 96.4: 983-990, 2009."
            https://doi.org/10.48550/arXiv.0805.2256

    Parameters
    ----------
    nr_samples
        The number of samples requested for intermediate distributions.
    error_schedule
        The schedule of error threshold distance for all distributions.
    """

    def __init__(self, log_prior, perturbation_kernel=None):
        # Log prior
        self._log_prior = log_prior

        # Default value for error threshold schedule
        self._e_schedule = [1]

        # Default value for current threshold
        self._threshold = 1

        # Size of intermediate distributions
        self._nr_samples = 100

        # Set up for first iteration
        self._samples = [[]]
        self._accepted_count = 0
        self._weights = []
        self._xs = None
        self._ready_for_tell = False
        self._t = 0
        dim = log_prior.n_parameters()

        # Setting the perturbation kernel
        if perturbation_kernel is None:
            self._perturbation_kernel = pints.MultivariateGaussianLogPrior(
                np.zeros(dim),
                0.001 * np.identity(dim))
        elif isinstance(perturbation_kernel, pints.LogPrior):
            self._perturbation_kernel = perturbation_kernel
        else:
            raise ValueError('Provided perturbation kernel must be an instance'
                  ' of pints.LogPrior')

    def name(self):
        """ See :meth:`pints.ABCSampler.name()`. """
        return 'ABC-SMC'

    def ask(self, n_samples):
        """ See :meth:`ABCSampler.ask()`. """
        if self._ready_for_tell:
            raise RuntimeError('ask called before tell.')
        if self._t == 0:
            self._xs = self._log_prior.sample(n_samples)
        else:
            self._xs = []

            while len(self._xs) < n_samples:
                theta_s_s = None  # to appease the linter
                while (theta_s_s is None or
                       self._log_prior(theta_s_s) == -np.inf):
                    indices = np.random.choice(
                        range(len(self._samples[self._t - 1])),
                        p=self._weights[self._t - 1])
                    theta_s = self._samples[self._t - 1][indices]
                    # perturb using K_t
                    theta_s_s = np.add(theta_s,
                                       self._perturbation_kernel.sample(1)[0])
                    # check if theta_s_s is possible under the prior
                    # sample again if not
                self._xs.append(theta_s_s)
        self._ready_for_tell = True
        return self._xs

    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('tell called before ask.')
        self._ready_for_tell = False
        if isinstance(fx, list):
            accepted = [a < self._threshold for a in fx]
            if sum(accepted) > 0:
                if self._t == len(self._e_schedule) - 1:
                    return [self._xs[c].tolist() for c, x in
                            enumerate(accepted) if x]

                self._accepted_count += sum(accepted)
                self._samples[self._t].extend(
                    [self._xs[c].tolist() for c, x in enumerate(accepted) if x]
                )

                if self._accepted_count >= self._nr_samples:
                    self._advance_time()
            return None
        else:
            if fx < self._threshold:
                self._accepted_count += 1
                if self._t == len(self._e_schedule) - 1:
                    return self._xs
                self._samples[self._t].append(self._xs)
                if self._accepted_count >= self._nr_samples:
                    self._advance_time()
            return None

    def _advance_time(self):
        t = self._t
        if t == 0:
            self._weights.append(
                np.full(self._accepted_count, 1 / self._accepted_count))
        else:
            unnorm_weights = self._calculate_weights(
                self._samples[t], self._samples[t - 1], self._weights[t - 1])
            # Normalise weights
            normal = sum(unnorm_weights)
            self._weights.append([w / normal for w in unnorm_weights])

        self._samples.append([])
        self._accepted_count = 0
        self._t += 1
        self._threshold = self._e_schedule[self._t]
        print(
            "Trying t=" + str(self._t)
            + ", threshold=" + str(self._threshold))

    def _calculate_weights(self, new_samples, old_samples, old_weights):
        new_weights = []
        for i in range(0, self._accepted_count):
            prior_prob = np.exp(self._log_prior(new_samples[i]))

            mw = [old_weights[j] * np.exp(self._perturbation_kernel(
                np.subtract(new_samples[i], old_samples[j])))
                for j in range(len(old_samples))]

            w = prior_prob / sum(mw)
            new_weights.append(w)
        return new_weights

    def set_threshold_schedule(self, schedule):
        """
        Sets a schedule for the threshold error distance that determines if a
        sample is accepted (if error < threshold). Schedule should be a list
        of float values.
        """
        e_schedule = np.array(schedule)
        if any(e_schedule <= 0):
            raise ValueError('All threshold values must be positive.')
        self._e_schedule = e_schedule
        self._threshold = self._e_schedule[self._t]

    def set_intermediate_size(self, n):
        """
        Sets the size of the intermediate distributions, after we find n
        acceptable samples then we will progress to the next threshold values
        in the schedule.
        """
        self._nr_samples = n
