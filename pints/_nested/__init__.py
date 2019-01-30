#
# Sub-module containing nested samplers
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints
import numpy as np
from scipy.misc import logsumexp


class NestedSampler(object):
    """
    Abstract base class for nested samplers.
    """
    def __init__(self, log_prior):
        # Store function
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError('Given log_prior must extend pints.LogPrior')
        self._log_prior = log_prior
        self._running_log_likelihood = -float('Inf')
        self._proposed = None

        # Initialise active point containers
        self._n_active_points = 400
        self._dimension = self._log_prior.n_parameters()
        self._m_active = np.zeros((self._n_active_points, self._dimension + 1))

    def needs_sensitivities(self):
        """
        Determines whether sampler uses sensitivities of the solution
        """
        raise self._needs_sensitivities

    def name(self):
        """ Name of sampler """
        raise NotImplementedError

    def ask(self):
        """
        Proposes new point at which to evaluate log-likelihood
        """
        raise NotImplementedError

    def tell(self, fx):
        """
        Whether to accept point if its likelihood exceeds the current
        minimum threshold
        """
        if np.isnan(fx) or fx < self._running_log_likelihood:
            return None
        else:
            return self._proposed

    def in_initial_phase(self):
        """
        For methods that need an initial phase (see
        :meth:`needs_initial_phase()`), this method returns ``True`` if the
        method is currently configured to be in its initial phase. For other
        methods a ``NotImplementedError`` is returned.
        """
        raise NotImplementedError

    def needs_initial_phase(self):
        """
        Returns ``True`` if this method needs an initial phase, for example
        ellipsoidal nested sampling has a period of running rejection
        sampling before it starts to fit ellipsoids to points
        """
        return False

    def set_initial_phase(self, in_initial_phase):
        """
        For methods that need an initial phase (see
        :meth:`needs_initial_phase()`), this method toggles the initial phase
        algorithm. For other methods a ``NotImplementedError`` is returned.
        """
        raise NotImplementedError

    def _set_running_log_likelihood(self, running_log_likelihood):
        """
        Updates the current value of the threshold log-likelihood value
        """
        self._running_log_likelihood = running_log_likelihood

    def running_log_likelihood(self):
        """
        Returns current value of the threshold log-likelihood value
        """
        return self._running_log_likelihood

    def set_n_active_points(self, active_points):
        """
        Sets the number of active points for the next run.
        """
        active_points = int(active_points)
        if active_points <= 5:
            raise ValueError('Number of active points must be greater than 5.')
        self._n_active_points = active_points

    def n_active_points(self):
        """
        Returns the number of active points that will be used in next run.
        """
        return self._n_active_points

    def active_points(self):
        """
        Returns the active points from nested sampling run
        """
        return self._m_active


class NestedSampling(object):
    """
    Uses nested sampling to sample from a posterior distribution

    Arguments:

    ``log_likelihood``
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    ``log_prior``
        A :class:`LogPrior` function on the same parameter space.
    """
    def __init__(self, log_likelihood, log_prior, method=None):

        # Store log_likelihood and log_prior
        #if not isinstance(log_likelihood, pints.LogLikelihood):
        if not isinstance(log_likelihood, pints.LogPDF):
            raise ValueError(
                'Given log_likelihood must extend pints.LogLikelihood')
        self._log_likelihood = log_likelihood

        # Store function
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError('Given log_prior must extend pints.LogPrior')
        self._log_prior = log_prior

        # Get dimension
        self._dimension = self._log_likelihood.n_parameters()
        if self._dimension != self._log_prior.n_parameters():
            raise ValueError(
                'Given log_likelihood and log_prior must have same number of'
                ' parameters.')

        # Logging
        self._log_to_screen = True
        self._log_filename = None
        self._log_csv = False

        # By default do serial evaluation
        self._parallel = False
        self._n_workers = 1
        self.set_parallel()

        # Parameters common to all routines
        # Total number of iterations
        self._iterations = 1000

        # Total number of posterior samples
        self._posterior_samples = 1000

        # Total number of likelihood evaluations made
        self._n_evals = 0

        # Convergence criterion in log-evidence
        self._marginal_log_likelihood_threshold = 0.5

        # Initial marginal difference
        self._diff = np.float('-Inf')

        # By default use ellipsoidal sampling
        if method is None:
            method = pints.NestedEllipsoidSampler
        else:
            try:
                ok = issubclass(method, pints.NestedSampler)
            except TypeError:   # Not a class
                ok = False
            if not ok:
                raise ValueError(
                    'Given method must extend pints.NestedSampler.'
                )

        self._sampler = method(log_prior=self._log_prior)
        # Check if sensitivities are required
        self._needs_sensitivities = self._sampler.needs_sensitivities()

    def set_parallel(self, parallel=False):
        """
        Enables/disables parallel evaluation.

        If ``parallel=True``, the method will run using a number of worker
        processes equal to the detected cpu core count. The number of workers
        can be set explicitly by setting ``parallel`` to an integer greater
        than 0.
        Parallelisation can be disabled by setting ``parallel`` to ``0`` or
        ``False``.
        """
        if parallel is True:
            self._parallel = True
            self._n_workers = pints.ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1

    def parallel(self):
        """
        Returns the number of parallel worker processes this routine will be
        run on, or ``False`` if parallelisation is disabled.
        """
        return self._n_workers if self._parallel else False

    def set_marginal_log_likelihood_threshold(self, threshold):
        """
        Sets criterion for determining convergence in estimate of marginal
        log likelihood which leads to early termination of the algorithm
        """
        if threshold <= 0:
            raise ValueError('Convergence threshold must be positive.')
        self._marginal_log_likelihood_threshold = threshold

    def iterations(self):
        """
        Returns the total number of iterations that will be performed in the
        next run.
        """
        return self._iterations

    def n_posterior_samples(self):
        """
        Returns the number of posterior samples that will be returned (see
        :meth:`set_posterior_samples()`).
        """
        return self._posterior_samples

    def run(self):
        """
        Runs the nested sampling routine and returns a tuple of the
        posterior samples and an estimate of the marginal likelihood.
        """

        # Choose method to evaluate
        f = self._log_likelihood
        if self._needs_sensitivities:
            f = f.evaluateS1

        # Create evaluator object
        if self._parallel:
            # Use at most n_workers workers
            n_workers = min(self._n_workers, self._chains)
            evaluator = pints.ParallelEvaluator(
                f, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(f)

        # Check if settings are sensible
        n_active_points = self._sampler.n_active_points()
        max_post = 0.25 * (self._iterations + n_active_points)
        if self._posterior_samples > max_post:
            raise ValueError(
                'Number of posterior samples must not exceed 0.25 times (the'
                ' number of iterations + the number of active points).')

        # Set up progress reporting
        next_message = 0
        message_warm_up = 3
        message_interval = 20

        # Start logging
        logging = self._log_to_screen or self._log_filename
        if logging:
            # Create timer
            timer = pints.Timer()

            if self._log_to_screen:
                # Show current settings
                print('Running ' + self._sampler.name())
                print('Number of active points: ' + str(n_active_points))
                print('Total number of iterations: ' + str(self._iterations))
                print('Total number of posterior samples: ' + str(
                    self._posterior_samples))

            # Set up logger
            logger = pints.Logger()
            if not self._log_to_screen:
                logger.set_stream(None)
            if self._log_filename:
                logger.set_filename(self._log_filename, csv=self._log_csv)

            # Add fields to log
            logger.add_counter('Iter.', max_value=self._iterations)
            logger.add_counter('Eval.', max_value=self._iterations * 10)
            #TODO: Add other informative fields ?
            logger.add_time('Time m:s')
            logger.add_float('Delta_log(z)')

        d = self._dimension
        m_active = self._sampler.active_points()
        m_initial = self._log_prior.sample(n_active_points)
        for i in range(0, n_active_points):
            # Calculate likelihood
            m_active[i, d] = self._log_likelihood(m_initial[i, :])
            self._n_evals += 1

            # Show progress
            if logging and i >= next_message:
                # Log state
                logger.log(0, self._n_evals, timer.time(), self._diff)

                # Choose next logging point
                if i > message_warm_up:
                    next_message = message_interval * (
                        1 + i // message_interval)

        m_active[:, :-1] = m_initial

        # store all inactive points, along with their respective
        # log-likelihoods (hence, d+1)
        self._m_inactive = np.zeros((self._iterations, d + 1))

        # store weights
        self._w = np.zeros(n_active_points + self._iterations)

        # store X values (defined in [1])
        self._X = np.zeros(self._iterations + 1)
        self._X[0] = 1

        # log marginal likelihood holder
        self._v_log_Z = np.zeros(self._iterations + 1)

        # Run!
        self._X[0] = 1.0
        i_message = n_active_points - 1
        for i in range(0, self._iterations):
            self._i = i
            # Update threshold and various quantities
            self._sampler._set_running_log_likelihood(
                np.min(m_active[:, d])
            )
            a_min_index = np.argmin(m_active[:, d])
            self._X[i + 1] = np.exp(-(i + 1) / n_active_points)
            if i > 0:
                self._w[i] = 0.5 * (self._X[i - 1] - self._X[i + 1])
            else:
                self._w[i] = self._X[i] - self._X[i + 1]
            self._v_log_Z[i] = self._sampler.running_log_likelihood()
            self._m_inactive[i, :] = m_active[a_min_index, :]

            # Use some method to propose new samples
            proposed = self._sampler.ask()

            # Evaluate their fit
            log_likelihood = evaluator.evaluate([proposed])[0]

            # Until log-likelihood exceeds current threshold keep drawing
            # samples
            while self._sampler.tell(log_likelihood) is None:
                proposed = self._sampler.ask()
                log_likelihood = evaluator.evaluate([proposed])[0]
                self._n_evals += 1

            m_active[a_min_index, :] = np.concatenate(
                (proposed, np.array([log_likelihood])))

            # Check whether within convergence threshold
            if i > 2:
                v_temp = np.concatenate((self._v_log_Z[0:(i - 1)],
                                        [np.max(m_active[:, d])]))
                w_temp = np.concatenate((self._w[0:(i - 1)], [self._X[i]]))
                self._diff = (logsumexp(self._v_log_Z[0:(i - 1)],
                                        b=self._w[0:(i - 1)]) -
                              logsumexp(v_temp, b=w_temp))
                if (np.abs(self._diff) <
                   self._marginal_log_likelihood_threshold):
                    if self._log_to_screen:
                        print('Convergence obtained with Delta_z = ' +
                              str(self._diff))

                    # shorten arrays according to current iteration
                    self._iterations = i
                    self._v_log_Z = self._v_log_Z[0:(self._iterations + 1)]
                    self._w = self._w[0:(n_active_points + self._iterations)]
                    self._X = self._X[0:(self._iterations + 1)]
                    self._m_inactive = self._m_inactive[0:self._iterations, :]
                    break

            # Show progress
            if logging:
                i_message += 1
                if i_message >= next_message:
                    # Log state
                    logger.log(i_message, self._n_evals, timer.time(),
                               self._diff)

                    # Choose next logging point
                    if i_message > message_warm_up:
                        next_message = message_interval * (
                            1 + i_message // message_interval)

        # Calculate log_evidence and uncertainty
        self._log_Z = self.marginal_log_likelihood()
        self._log_Z_sd = self.marginal_log_likelihood_standard_deviation()

        # Draw samples from posterior
        n = self._posterior_samples
        self._m_posterior_samples = self.sample_from_posterior(n)

        return self._m_posterior_samples

    def sample_from_posterior(self, posterior_samples):
        """
        Draws posterior samples based on nested sampling run
        """
        if posterior_samples < 1:
            raise ValueError('Number of posterior samples must be positive.')

        # Calculate probabilities (can this be used to calculate effective
        # sample size as in importance sampling?) of each particle
        self._vP = np.exp(self._m_samples_all[:, self._dimension]
                          - self._log_Z) * self._w

        # Draw posterior samples
        m_theta = self._m_samples_all[:, :-1]
        vIndex = np.random.choice(
            range(0, self._iterations + self._sampler.n_active_points()),
            size=posterior_samples, p=self._vP)

        m_posterior_samples = m_theta[vIndex, :]
        return m_posterior_samples

    def posterior_samples(self):
        """
        Returns posterior samples generated during run of nested
        sampling object
        """
        return self._m_posterior_samples

    def prior_space(self):
        """
        Returns a vector of X samples which approximates the proportion
        of prior space compressed
        """
        return self._X

    def marginal_log_likelihood(self):
        """
        Calculates the marginal log likelihood of nested sampling run
        """
        # Include active particles in sample
        m_active = self._sampler.active_points()
        self._v_log_Z[self._iterations] = logsumexp(m_active[:,
                                                    self._dimension])
        self._w[self._iterations:] = float(self._X[self._iterations]) / float(
            self._sampler.n_active_points())
        self._m_samples_all = np.vstack((self._m_inactive, m_active))

        # Determine log evidence
        log_Z = logsumexp(self._v_log_Z,
                          b=self._w[0:(self._iterations + 1)])
        self._log_Z_called = True
        return log_Z

    def marginal_log_likelihood_standard_deviation(self):
        """
        Calculates standard deviation in marginal log likelihood as in [1]

        [1] "Multimodan nested sampling: an efficient and robust alternative
        to Markov chain Monte Carlo methods for astronomical data analyses",
        F. Feroz and M. P. Hobson, 2008, Mon. Not. R. Astron. Soc.
        """
        if not self._log_Z_called:
            self._marginal_log_likelihood()
        log_L_minus_Z = self._v_log_Z - self._log_Z
        log_Z_sd = logsumexp(log_L_minus_Z,
                             b=self._w[0:(self._iterations + 1)] *
                             log_L_minus_Z)
        log_Z_sd = np.sqrt(log_Z_sd / self._sampler.n_active_points())
        return log_Z_sd

    def log_likelihood_vector(self):
        """
        Returns vector of log likelihoods for each of the
        stacked [m_active, m_inactive] points
        """
        return self._m_samples_all[:, -1]

    def set_log_to_file(self, filename=None, csv=False):
        """
        Enables logging to file when a filename is passed in, disables it if
        ``filename`` is ``False`` or ``None``.

        The argument ``csv`` can be set to ``True`` to write the file in comma
        separated value (CSV) format. By default, the file contents will be
        similar to the output on screen.
        """
        if filename:
            self._log_filename = str(filename)
            self._log_csv = True if csv else False
        else:
            self._log_filename = None
            self._log_csv = False

    def set_log_to_screen(self, enabled):
        """
        Enables or disables logging to screen.
        """
        self._log_to_screen = True if enabled else False

    def set_iterations(self, iterations):
        """
        Sets the total number of iterations to be performed in the next run.
        """
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError('Number of iterations cannot be negative.')
        self._iterations = iterations

    def set_n_posterior_samples(self, posterior_samples):
        """
        Sets the number of posterior samples to generate from points proposed
        by the nested sampling algorithm.
        """
        posterior_samples = int(posterior_samples)
        if posterior_samples < 1:
            raise ValueError(
                'Number of posterior samples must be greater than zero.')
        self._posterior_samples = posterior_samples

    def effective_sample_size(self):
        """
        Calculates the effective sample size of posterior samples from a
        nested sampling run using the formula,
        ESS = exp(-sum_i=1^m p_i log p_i),
        in other words, the information.
        From here: https://www.cosmos.esa.int/documents/1371789/1544987/
        5-nested-sampling.pdf/c0280ec4-68e3-98ce-e5c4-1a554ed61242
        """
        self._log_vP = (self._m_samples_all[:, self._dimension]
                        - self._log_Z + np.log(self._w))
        return np.exp(-np.sum(self._vP * self._log_vP))

    def inactive_points(self):
        """
        Returns the inactive points from nested sampling
        """
        return self._m_inactive

    def active_points(self):
        """
        Returns the active points from nested sampling
        """
        return self._sampler.active_points()
