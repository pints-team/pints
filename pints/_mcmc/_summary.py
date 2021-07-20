#
# MCMC summary method
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
#
import numpy as np
import pints
import warnings

from tabulate import tabulate


class MCMCSummary(object):
    """
    Calculates and prints key summaries of posterior samples and diagnostic
    quantities from MCMC chains.

    These include the posterior mean, standard deviation, quantiles, rhat,
    effective sample size and (if running time is supplied) effective samples
    per second.

    Parameters
    ----------
    chains
        An array or list of chains returned by an MCMC sampler.
    time : float
        The time taken for the run, in seconds (optional).
    parameter_names : sequence
        A list of parameter names (optional).

    References
    ----------
    .. [1] "Inference from iterative simulation using multiple
            sequences", A Gelman and D Rubin, 1992, Statistical Science.
    .. [2] "Bayesian data analysis", 3rd edition, CRC Press.,  A Gelman et al.,
           2014.
    """

    def __init__(self, chains, time=None, parameter_names=None):

        # Store unmodified chains
        # Note: For performance reasons we're not copying the chains, or
        # enforcing read-only or anything like that: so users will have the
        # ability to change the contents of ``chains``, and make it go out of
        # sync with the summary.
        self._chains = chains
        self._chains_unmodified = chains

        # Deal with special case where only one chain is provided
        if len(chains) == 1:
            warnings.warn(
                'Summaries calculated with one chain may be unreliable. It is'
                ' recommended that you rerun sampling with more than one'
                ' chain.')

            self._chains = chains

        # Get number of parameters
        self._n_parameters = chains[0].shape[1]

        # Check time, if supplied
        if time is not None and float(time) <= 0:
            raise ValueError('Elapsed time must be positive.')
        self._time = time

        # Check parameter names, if supplied
        if parameter_names is None:
            parameter_names = [
                'param ' + str(i + 1) for i in range(self._n_parameters)]
        elif self._n_parameters != len(parameter_names):
            raise ValueError(
                'Parameter names list must be same length as number of '
                'sampled parameters')
        self._parameter_names = parameter_names

        # Initialise
        self._ess = None
        self._ess_per_second = None
        self._mean = None
        self._quantiles = None
        self._rhat = None
        self._std = None
        self._summary_list = []
        self._summary_str = None

        # Create summary
        self._make_summary()

    def __str__(self):
        """
        Prints posterior summaries for all parameters to the console, including
        the parameter name, posterior mean, posterior std deviation, the
        2.5%, 25%, 50%, 75% and 97.5% posterior quantiles, rhat, effective
        sample size (ess) and ess per second of run time.
        """
        if self._summary_str is None:
            headers = [
                'param', 'mean', 'std.',
                '2.5%', '25%', '50%', '75%', '97.5%',
                'rhat', 'ess']
            if self._time is not None:
                headers.append('ess per sec.')

            self._summary_str = tabulate(
                self._summary_list,
                headers=headers,
                numalign='left',
                floatfmt='.2f',
            )

        return self._summary_str

    def chains(self):
        """
        Returns posterior samples from all chains separately.
        """
        return self._chains_unmodified

    def ess(self):
        """
        Return the effective sample size for each parameter as defined in [2]_.
        """
        return self._ess

    def ess_per_second(self):
        """
        Return the effective sample size (as defined in [2]_) per second of run
        time for each parameter.

        This is only defined if a run time was passed in at construction time,
        if no run time is known ``None`` is returned.
        """
        return self._ess_per_second

    def _make_summary(self):
        """
        Calculates posterior summaries for all parameters.
        """
        stacked = np.vstack(self._chains)

        # Mean, std and quantiles
        self._mean = np.mean(stacked, axis=0)
        self._std = np.std(stacked, axis=0)
        self._quantiles = np.percentile(
            stacked, [2.5, 25, 50, 75, 97.5], axis=0)

        # Rhat
        self._rhat = pints.rhat(self._chains)

        # Effective sample size
        self._ess = np.zeros(self._n_parameters)
        for i, chain in enumerate(self._chains):
            self._ess += pints.effective_sample_size(chain)

        if self._time is not None:
            self._ess_per_second = np.array(self._ess) / self._time

        # Create
        for i in range(0, self._n_parameters):
            row = [
                self._parameter_names[i],
                self._mean[i],
                self._std[i],
                self._quantiles[0, i],
                self._quantiles[1, i],
                self._quantiles[2, i],
                self._quantiles[3, i],
                self._quantiles[4, i],
                self._rhat[i],
                self._ess[i],
            ]
            if self._time is not None:
                row.append(self._ess_per_second[i])

            self._summary_list.append(row)

    def mean(self):
        """
        Return the posterior means of all parameters.
        """
        return self._mean

    def quantiles(self):
        """
        Return the 2.5%, 25%, 50%, 75% and 97.5% posterior quantiles.
        """
        return self._quantiles

    def rhat(self):
        """
        Return Gelman and Rubin's rhat value as defined in [1]_. If a single
        chain is used, the chain is split into two halves and rhat is
        calculated using these two parts.
        """
        return self._rhat

    def std(self):
        """
        Return the posterior standard deviation of all parameters.
        """
        return self._std

    def summary(self):
        """
        Return a list of the parameter name, posterior mean, posterior std
        deviation, the 2.5%, 25%, 50%, 75% and 97.5% posterior quantiles,
        rhat, effective sample size (ess) and ess per second of run time.
        """
        return list(self._summary_list)

    def time(self):
        """
        Return the run time taken for sampling.
        """
        return self._time
