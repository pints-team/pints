#!/usr/bin/env python3
#
# Tests the Pints plot methods.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.toy as toy
import pints.plot
import unittest
import numpy as np

debug = False


class TestPlot(unittest.TestCase):
    """
    Tests Pints plot methods.
    """
    def __init__(self):
        # Create toy model
        self.model = toy.LogisticModel()
        self.real_parameters = [0.015, 500]
        self.times = np.linspace(0, 1000, 100)  # small problem
        self.values = self.model.simulate(self.real_parameters, self.times)

        # Add noise
        self.noise = 10
        self.values += np.random.normal(0, self.noise, self.values.shape)
        self.real_parameters.append(self.noise)
        self.real_parameters = np.array(self.real_parameters)

        # Create an object with links to the model and time series
        self.problem = pints.SingleSeriesProblem(
            self.model, self.times, self.values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        self.log_prior = pints.UniformLogPrior(
            [0.01, 400, self.noise * 0.1],
            [0.02, 600, self.noise * 100]
        )

        # Create a log likelihood
        self.log_likelihood = pints.UnknownNoiseLogLikelihood(self.problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        self.log_posterior = pints.LogPosterior(
            self.log_likelihood, self.log_prior)
        
        # Run MCMC
        self.x0 = [
            self.real_parameters * 1.1,
            self.real_parameters * 0.9,
            self.real_parameters * 1.05
        ]
        mcmc = pints.MCMCSampling(self.log_posterior, 3, self.x0,
                                          method=pints.AdaptiveCovarianceMCMC)
        mcmc.set_max_iterations(300)  # make it as small as possible
        self.samples = mcmc.run()

    def test_function(self):
        """
        Tests the function function.
        """
        pints.plot.function(self.log_posterior, self.real_parameters)


    def test_function_between_points(self):
        """
        Tests the function_between_points function.
        """
        pints.plot.function(self.log_posterior,
                            self.real_parameters * 0.8,
                            self.real_parameters * 1.2)

    def test_histogram(self):
        """
        Tests the histogram function.
        """
        # Test it can plot without error
        pints.plot.histogram(self.samples, ref_parameters=self.log_posterior)

        # Check invalid ref_parameter input
        self.assertRaises(
            ValueError, pints.plot.histogram,
            self.samples, self.log_posterior[0]
        )


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
