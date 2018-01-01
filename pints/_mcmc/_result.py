#
# MCMC result object
#
# This file is part of PINTS.
#  Copyright (c) 2017, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
# Some code in this file was adapted from Myokit (see http://myokit.org)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pints._diagnostics as diagnostics
import numpy as np
from tabulate import tabulate


class McmcResultObject(object):
    """
    Class that represents the result of running MCMC and
    calculates key parameter summaries which are printed to the
    console, including the posterior mean, standard deviation,
    quantiles, rhat and effective sample size
    """

    def __init__(self, chains, time):
        self._chains = chains
        stacked = np.vstack(chains)
        self._time = time
        self._mean = np.mean(stacked, axis=0)
        self._std = np.std(stacked, axis=0)
        self._quantiles = np.percentile(stacked, [2.5, 25, 50,
                                                        75, 97.5], axis=0)
        self._ess = diagnostics.effective_sample_size(stacked)
        self._ess_per_second = self._ess / time
        self._num_params = stacked.shape[1]
        self._num_chains = len(chains)

        # If there is more than 1 chain calculate rhat
        # otherwise return NA
        if self._num_chains > 1:
            self._rhat = diagnostics.rhat_all_params(chains)
        else:
            self._rhat = np.repeat("NA", self._num_params)

        self._summary_list = None
        self.make_summary()
        
    def chains(self):
        """
        Returns the posterior samples from all chains
        separately
        """
        return self._chains

    def mean(self):
        """
        Return the posterior means of all parameters
        """
        return self._mean

    def std(self):
        """
        Return the posterior standard deviation of all parameters
        """
        return self._std

    def quantiles(self):
        """
        Return the 2.5%, 25%, 50%, 75% and 97.5% posterior quantiles
        """
        return self._quantiles

    def rhat(self):
        """
        Return Gelman and Rubin's [1] rhat value where values rhat > 1.1
        indicate lack of posterior convergence

        [1] "Inference from iterative simulation using multiple
        sequences", 1992, Gelman and Rubin, Statistical Science.
        """
        return self._rhat

    def ess(self):
        """
        Return the effective sample size [1] for each parameter

        [1] "Bayesian data analysis", 3rd edition, 2014, Gelman et al.,
        CRC Press.
        """
        return self._ess

    def ess_per_second(self):
        """
        Return the effective sample size [1] per second of
        run time for each parameter

        [1] "Bayesian data analysis", 3rd edition, 2014, Gelman et al.,
        CRC Press.
        """
        return self._ess_per_second

    def make_summary(self):
        """
        Calculates posterior summaries for all parameters and puts them in a
        list
        """
        self._summary_list = []
        for i in range(0, self._num_params):
            self._summary_list.append(["param " + str(i + 1), self._mean[i],
                                       self._std[i], self._quantiles[0, i],
                                       self._quantiles[1, i],
                                       self._quantiles[2, i],
                                       self._quantiles[3, i],
                                       self._quantiles[4, i],
                                       self._rhat[i], self._ess[i],
                                       self._ess_per_second[i]])

    def summary(self):
        """
        Return a list of the parameter name, posterior mean, posterior std
        deviation, the 2.5%, 25%, 50%, 75% and 97.5% posterior quantiles,
        rhat, effective sample size (ess) and ess per second of run time
        """
        return self._summary_list

    def print_summary(self):
        """
        Prints posterior summaries for all parameters to the console, including
        the parameter name, posterior mean, posterior std deviation, the
        2.5%, 25%, 50%, 75% and 97.5% posterior quantiles, rhat, effective
        sample size (ess) and ess per second of run time
        """
        print(tabulate(self._summary_list, headers=["param", "mean", "std.",
                                                    "2.5%", "25%", "50%",
                                                    "75%", "97.5%", "rhat",
                                                    "ess", "ess per sec."],
                       numalign="left", floatfmt=".2f"))

    def extract(self, param_number):
        """
        Extracts posterior samples for a given parameter number
        """
        stacked = np.vstack(self._chains)
        return stacked[:, param_number]
        
    def extract_all(self):
        """
        Return the posterior samples for all parameters
        """
        return np.vstack(self._chains)
    
    def time(self):
        """
        Return the run time taken for sampling
        """
        return self._time
