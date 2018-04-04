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
    def __init__(self, method):
        if not issubclass(method, pints.MCMCSampler):
            raise ValueError('Given method must extend pints.MCMCSampler.')
        self._method = method
        self._threshold_mean_1 = 0.5
        self._threshold_sd_1 = 0.5

    def normal_low_correlation(self, num_chains):
      
        # Create target density
        log_pdf = toy.HighDimensionalNormalLogPDF(dimension=3, correlation=0.5)
        
        # Create sampler
        self._num_chains = num_chains
        x0 = np.random.uniform([2, 2, 2], [8, 8, 8],size=(self._num_chains, 3))
        mcmc = pints.MCMCSampling(log_pdf, self._num_chains, x0, method=self._method)

        # Set maximum number of iterations
        mcmc.set_max_iterations(4000)

        # Disable logging
        mcmc.set_log_to_screen(False)
        
        # Run sampler!
        chains = mcmc.run()
        
        # Discard warmup
        chains = chains[:, 2000:, :]
        
        # Stack
        chains = np.vstack(chains)
        
        # Compare means and standard deviations
        mean_est = np.mean(chains, axis=0)
        sd_est = np.std(chains, axis=0)
        sd_true = np.sqrt([1, 2, 3])
        self.assertTrue(np.linalg.norm(mean_est) < self._threshold_mean_1)
        self.assertTrue(np.linalg.norm(sd_est - sd_true) < self._threshold_sd_1)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
