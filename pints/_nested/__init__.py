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

    Arguments:

    ``log_likelihood``
        A :class:`LogPDF` function that evaluates points in the parameter
        space.
    ``log_prior``
        A :class:`LogPrior` function on the same parameter space.

    """
    def __init__(self, log_likelihood, log_prior):

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

        # Parameters common to all routines
        # Target acceptance rate
        self._active_points = 1000

        # Total number of iterations
        self._iterations = 1000

        # Total number of posterior samples
        self._posterior_samples = 1000

        # Total number of likelihood evaluations made
        self._n_evals = 0

        # Convergence criterion in log-evidence
        self._diff_log_Z = 0.5

    def active_points(self):
        """
        Returns the number of active points that will be used in next run.
        """
        return self._active_points

    def iterations(self):
        """
        Returns the total number of iterations that will be performed in the
        next run.
        """
        return self._iterations

    def posterior_samples(self):
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

        # Create evaluator object
        if self._parallel:
            # Use at most n_workers workers
            n_workers = min(self._n_workers, self._chains)
            evaluator = pints.ParallelEvaluator(
                self._log_likelihood, n_workers=n_workers)
        else:
            evaluator = pints.SequentialEvaluator(self._log_likelihood)

        # Check if settings are sensible
        max_post = 0.25 * (self._iterations + self._active_points)
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
                print('Running nested rejection sampling')
                print('Number of active points: ' + str(self._active_points))
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

        # Problem dimension
        d = self._dimension

        # Generate initial random points by sampling from the prior
        self._m_active = np.zeros((self._active_points, d + 1))
        m_initial = self._log_prior.sample(self._active_points)
        for i in range(0, self._active_points):
            # Calculate likelihood
            self._m_active[i, d] = self._log_likelihood(m_initial[i, :])
            self._n_evals += 1

            # Show progress
            if logging and i >= next_message:
                # Log state
                logger.log(0, self._n_evals, timer.time())

                # Choose next logging point
                if i > message_warm_up:
                    next_message = message_interval * (
                        1 + i // message_interval)

        self._m_active[:, :-1] = m_initial

        # store all inactive points, along with their respective
        # log-likelihoods (hence, d+1)
        self._m_inactive = np.zeros((self._iterations, d + 1))

        # store weights
        self._w = np.zeros(self._active_points + self._iterations)

        # store X values (defined in [1])
        self._X = np.zeros(self._iterations + 1)
        self._X[0] = 1

        # log marginal likelihood holder
        self._v_log_Z = np.zeros(self._iterations + 1)

        # Run!
        i_message = self._active_points - 1
        for i in range(0, self._iterations):
            # Update threshold and various quantities
            self._running_log_likelihood = np.min(self._m_active[:, d])
            a_min_index = np.argmin(self._m_active[:, d])
            self._X[i + 1] = np.exp(-(i + 1) / self._active_points)
            self._w[i] = 0.5 * (self._X[i - 1] - self._X[i + 1])
            self._v_log_Z[i] = self._running_log_likelihood
            self._m_inactive[i, :] = self._m_active[a_min_index, :]

            # Use some method to propose new samples
            self._proposed = self._ask()
            log_likelihood = evaluator.evaluate(self._proposed)

            # Until log-likelihood exceeds current threshold keep drawing
            # samples
            while self._tell(log_likelihood) is None:
                self._proposed = self._ask()
                log_likelihood = evaluator.evaluate(self._proposed)
                self._n_evals += 1

            self._m_active[a_min_index, :] = np.concatenate(
                (self._proposed, np.array([log_likelihood])))

            # Show progress
            if logging:
                i_message += 1
                if i_message >= next_message:
                    # Log state
                    logger.log(i_message, self._n_evals, timer.time())

                    # Choose next logging point
                    if i_message > message_warm_up:
                        next_message = message_interval * (
                            1 + i_message // message_interval)

        # Include active particles in sample
        self._v_log_Z[self._iterations] = logsumexp(self._m_active[:, d])
        self._w[self._iterations:] = float(self._X[self._iterations]) / float(
            self._active_points)
        m_samples_all = np.vstack((self._m_inactive, self._m_active))

        # Determine log evidence
        self._log_Z = logsumexp(self._v_log_Z,
                                b=self._w[0:(self._iterations + 1)])

        # Calculate probabilities (can this be used to calculate effective
        # sample size as in importance sampling?) of each particle
        vP = np.exp(m_samples_all[:, d] - self._log_Z) * self._w

        # Draw posterior samples
        m_theta = m_samples_all[:, :-1]
        vIndex = np.random.choice(
            range(0, self._iterations + self._active_points),
            self._posterior_samples, p=vP)
        m_posterior_samples = m_theta[vIndex, :]

        return m_posterior_samples, self._log_Z

    def _ask(self):
        """
        Proposes new point at which to evaluate log-likelihood
        """
        raise NotImplementedError

    def _tell(self, fx):
        """
        Whether to accept point if its likelihood exceeds the current
        minimum threshold
        """
        self._n_evals += 1
        if fx < self._running_log_likelihood:
            return None
        else:
            return self._proposed

    def set_active_points(self, active_points):
        """
        Sets the number of active points for the next run.
        """
        active_points = int(active_points)
        if active_points <= 5:
            raise ValueError('Number of active points must be greater than 5.')
        self._active_points = active_points

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

    def set_posterior_samples(self, posterior_samples):
        """
        Sets the number of posterior samples to generate from points proposed
        by the nested sampling algorithm.
        """
        posterior_samples = int(posterior_samples)
        if posterior_samples < 1:
            raise ValueError(
                'Number of posterior samples must be greater than zero.')
        self._posterior_samples = posterior_samples
