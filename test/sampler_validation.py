#!/usr/bin/env python3
#
# Tests samplers against some basic targets
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy as toy
import unittest
import numpy as np

debug = False


class SamplerValidation(unittest.TestCase):
    """
    Tests a sampler against a number of targets.
    """
    def __init__(self, method, num_chains, num_iterations):
        if not issubclass(method, pints.MCMCSampler):
            raise ValueError('Given method must extend pints.MCMCSampler.')
        self._method = method
        self._num_chains = num_chains
        self._num_iterations = num_iterations
        self._extremum = 8

        # Thresholds are maxima from 200 runs of Random Walk Metropolis
        self._threshold_mean_3_low = 1.88
        self._threshold_cov_3_low = 3.98
        self._threshold_mean_3_high = 1.86
        self._threshold_cov_3_high = 4.16
        self._threshold_mean_6_low = 3.98
        self._threshold_cov_6_low = 14.4
        self._threshold_mean_6_high = 5.5
        self._threshold_cov_6_high = 23.1

    def normal_nd_correlation(self, dimension, correlation,
                              mean_threshold, cov_threshold):
        # Create target density
        log_pdf = toy.HighDimensionalNormalLogPDF(dimension=dimension,
                                                  correlation=correlation)

        # Create random starting locations
        x0 = np.random.uniform(np.full(dimension, -1 * self._extremum),
                               np.full(dimension, self._extremum),
                               size=(self._num_chains, dimension))

        # Create sampler
        mcmc = pints.MCMCSampling(log_pdf, self._num_chains, x0,
                                  method=self._method)

        # Set maximum number of iterations
        mcmc.set_max_iterations(self._num_iterations)

        # Disable logging
        mcmc.set_log_to_screen(False)

        # Run sampler!
        chains = mcmc.run()

        # Discard warmup
        chains = chains[:, int(self._num_iterations / 2.0):, :]

        # Stack
        chains = np.vstack(chains)

        # Compare means and covariance matrices
        mean_est = np.mean(chains, axis=0)
        cov_est = np.cov(chains.T)
        cov_true = log_pdf.covariance_matrix()
        self.assertTrue(np.linalg.norm(mean_est) < mean_threshold)
        self.assertTrue(np.linalg.norm(cov_true - cov_est) < cov_threshold)

    def normal_nd_correlation_run_all(self):

        # 3d normal low correlation
        self.normal_nd_correlation(3, 0.5,
                                   self._threshold_mean_3_low,
                                   self._threshold_cov_3_low)

        # 3d normal high correlation
        self.normal_nd_correlation(3, 0.95,
                                   self._threshold_mean_3_high,
                                   self._threshold_cov_3_high)

        # 6d normal low correlation
        self.normal_nd_correlation(6, 0.5,
                                   self._threshold_mean_6_low,
                                   self._threshold_cov_6_low)

        # 6d normal low correlation
        self.normal_nd_correlation(6, 0.95,
                                   self._threshold_mean_6_high,
                                   self._threshold_cov_6_high)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
