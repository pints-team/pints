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
    Implements the ABC-SMC algorithm as described in [1]. This
    algorithm is also referred to as ABC Population Monte Carlo
    (ABC PMC) [2].

    In this algorithm there are going to be several rounds of ABC,
    each with a different proposal distribution. For each round, we
    try to obtain ``nr_samples`` samples at once. In order to
    transition between each set of ``nr_samples``, we have intermediate
    distributions :math:`p_t(\theta)` from which we sample the
    candidates for the new distribution. In particular,
    :math:`p_0(\theta)` is the prior and for :math:`t>0`
    :math:`p_t(\theta)` proposes sample :math:`\theta^{t-1}_i` with
    weight :math:`\w^{t-1}_i` (which is computed at the end of the
    iteration). In particular, at each iteration of the algorithm
    the following steps occur:

    .. math::
        \begin{align}
        & \theta^* \sim p_{t-1}(\theta) \textrm{, i.e. sample parameters from
        previous intermediate distribution} \\
        & \theta^{**} \sim K(\theta|\theta^{*}), \textrm{i.e. perturb }
        \theta^{*} \textrm{ to     obtain to new point } x \sim
        p(x|\theta^{**})\textrm{, i.e. sample data from sampling
        distribution} \\
        & \textrm{if } s(x) < \textrm{threshold}_(t), \theta^* \textrm{
        add to list of samples[t]}
        \end{align}

    After we have obtained ``nr_samples`` samples, ``t`` is advanced, and
    weights are calculated for ``samples[t-1]``. At the last error
    threshold, samples are returned whenever they are accepted.

    Parameters
    ----------
    log_prior
        A ``pints.LogPrior`` that specifies the logarithmic prior for
        the set of parameters that will be sampled.
    perturbation_kernel
        A ``pints.LogPrior`` used for perturbing the particles sampled
        from the previous generation. By default a multivariate
        Gaussian distribution is used with mean I and 0.001 * I
        as covariance.
    nr_samples
        The number of samples requested for intermediate distributions.
    error_schedule
        The schedule of error threshold distance for all distributions.

    References
    ----------
    .. [1] Toni, Welch, Strelkowa, Ipsem, Stumpf. Approximate Bayesian
           computation scheme for parameter inference and model selection in
           dynamical systems. Journal of the Royal Society Interface,
           6.31: 187-202, 2009.
           https://doi.org/10.1098/rsif.2008.0172

    .. [2] Beaumont, Cornuet, Marin, Robert. Adaptive approximate Bayesian
           computation. Biometrika, 96.4: 983-990, 2009.
           https://doi.org/10.48550/arXiv.0805.2256
    """

    def __init__(self, log_prior, perturbation_kernel=None,
                 nr_samples=100, error_schedule=[1]):
        # Log prior
        self._log_prior = log_prior

        # Default value for error threshold schedule
        self._e_schedule = error_schedule

        # Default value for current threshold
        self._threshold = error_schedule[0]

        # Size of intermediate distributions
        self._nr_samples = nr_samples

        # Set up for first iteration
        self._samples = [[], []]
        self._accepted_count = 0
        self._weights = []
        self._xs = None
        self._ready_for_tell = False
        self._t = 0
        self._to_print = True

        # Setting the perturbation kernel
        if perturbation_kernel is None:
            dim = log_prior.n_parameters()
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
            if self._to_print:
                self._to_print = False
                print(
                    "Starting t=" + str(self._t)
                    + ", with threshold=" + str(self._threshold))
            self._xs = self._log_prior.sample(n_samples)
        else:
            self._xs = [None] * n_samples
            i = 0
            while i < n_samples:
                theta_s_s = None
                while (theta_s_s is None or
                       self._log_prior(theta_s_s) == -np.inf):
                    indices = np.random.choice(
                        range(len(self._samples[(self._t - 1) % 2])),
                        p=self._weights[self._t - 1])
                    theta_s = self._samples[(self._t - 1) % 2][indices]
                    # perturb using K_t
                    theta_s_s = np.add(theta_s,
                                       self._perturbation_kernel.sample(1)[0])
                    # check if theta_s_s is possible under the prior
                    # sample again if not
                self._xs[i] = theta_s_s
                i = i + 1
        self._ready_for_tell = True
        intermediate = np.array(self._xs, copy=True)

        # Set as read-only so that it cannot be corrupted
        intermediate.setflags(write=False)

        return intermediate

    def tell(self, fx):
        """ See :meth:`ABCSampler.tell()`. """
        if not self._ready_for_tell:
            raise RuntimeError('tell called before ask.')
        self._ready_for_tell = False
        fx = np.asarray(fx)
        accepted = fx < self._threshold
        if np.any(accepted) > 0:
            if self._t < len(self._e_schedule) - 1:
                self._accepted_count += sum(accepted)
                self._samples[self._t % 2].extend(
                    [self._xs[c].tolist() for c, x in enumerate(accepted) if x]
                )

                if self._accepted_count >= self._nr_samples:
                    self._advance_time()
            return [self._xs[c].tolist() for c, x in
                    enumerate(accepted) if x]

        return None

    def _advance_time(self):
        t = self._t
        if t == 0:
            self._weights.append(
                np.full(self._accepted_count, 1 / self._accepted_count))
        else:
            unnorm_weights = self._calculate_weights(
                self._samples[self._t % 2], self._samples[(self._t - 1) % 2],
                self._weights[t - 1])
            # Normalise weights
            normal = sum(unnorm_weights)
            self._weights.append([w / normal for w in unnorm_weights])

        self._samples[(t + 1) % 2] = []
        self._accepted_count = 0
        self._t += 1
        self._threshold = self._e_schedule[self._t]
        print(
            "Starting t=" + str(self._t)
            + ", with threshold=" + str(self._threshold))

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

    def set_perturbation_kernel(self, perturbation_kernel):
        """
        Sets the perturbation kernel used for perturbing particles
        that are sampled from the previous generations. It must
        implement the ``pints.LogPrior`` class.
        """
        if not isinstance(perturbation_kernel, pints.LogPrior):
            raise ValueError('Perturbation kernel must implement ',
                             'pints.LogPrior.')
        self._perturbation_kernel = perturbation_kernel

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
