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
import pints
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
    
    def __init__(self, chains):
        self._stacked = np.vstack(chains)
        self._mean = np.mean(self._stacked, axis=0)
        self._std = np.std(self._stacked, axis=0)
        self._quantiles = np.percentile(self._stacked,[2.5, 25, 50,
                                                        75, 97.5], axis=0)
        self._ess = diagnostics.effective_sample_size(self._stacked)
        self._num_params = self._stacked.shape[1]
        self._num_chains = len(chains)
        
        # If there is more than 1 chain calculate rhat
        # otherwise return NA
        if self._num_chains > 1:
            self._rhat = diagnostics.rhat_all_params(chains)
        else:
            self._rhat = np.repeat("NA", self._num_params)
        
        self._summary_list = None
        self.make_summary()
    
    def stacked(self):
        """
        Return the posterior samples from all chains
        vertically stacked
        """
        return self._stacked
    
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

    def make_summary(self):
        """
        Calculates posterior summaries for all parameters and puts them in a
        list
        """
        self._summary_list = []
        for i in range(0, self._num_params):
            self._summary_list.append(["param " + str(i+1), self._mean[i], self._std[i],
                                      self._quantiles[0, i], self._quantiles[1, i],
                                      self._quantiles[2, i], self._quantiles[3, i],
                                      self._quantiles[4, i], self._rhat[i], self._ess[i]])

    def summary(self):
        """
        Return a list of the parameter name, posterior mean, posterior std
        deviation, the 2.5%, 25%, 50%, 75% and 97.5% posterior quantiles,
        rhat and effective sample size
        """
        return self._summary_list

    def print_summary(self):
        """
        Prints posterior summaries for all parameters to the console, including
        the parameter name, posterior mean, posterior std deviation, the
        2.5%, 25%, 50%, 75% and 97.5% posterior quantiles, rhat and effective
        sample size
        """
        print(tabulate(self._summary_list, headers = ["param", "mean", "std.",
                "2.5%","25%", "50%", "75%", "97.5%", "rhat", "ess"],
                numalign="left", floatfmt=".2f"))

    def extract(self, param_number):
        """
        Extracts posterior samples for a given parameter number
        """
        return self._stacked[:, param_number]
