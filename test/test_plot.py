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
import matplotlib

# Avoid DISPLAY problem...
matplotlib.use('Agg')

debug = False


class TestPlot(unittest.TestCase):
    """
    Tests Pints plot methods.
    """
    def __init__(self, name):
        super(TestPlot, self).__init__(name)

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
        self.problem = pints.SingleOutputProblem(
            self.model, self.times, self.values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        self.lower = [0.01, 400, self.noise * 0.1]
        self.upper = [0.02, 600, self.noise * 100]
        self.log_prior = pints.UniformLogPrior(
            self.lower,
            self.upper
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
        mcmc.set_log_to_screen(False)
        self.samples = mcmc.run()

    def test_function(self):
        """
        Tests the function function.
        """
        # Test it can plot without error
        pints.plot.function(self.log_posterior, self.real_parameters)

        # Check lower and upper bounds input gives no error
        pints.plot.function(self.log_posterior, self.real_parameters,
                            self.lower, self.upper)

        # Check evaluations gives no error
        pints.plot.function(self.log_posterior, self.real_parameters,
                            evaluations=5)

        # Check invalid function input
        self.assertRaises(
            ValueError, pints.plot.function, self.real_parameters,
            self.real_parameters
        )

        # Check invalid dimension input
        self.assertRaises(
            ValueError, pints.plot.function, self.log_posterior,
            list(self.real_parameters) + [0]
        )

        # Check invalid evaluations input
        self.assertRaises(
            ValueError, pints.plot.function, self.log_posterior,
            self.real_parameters, evaluations=-1
        )

    def test_function_between_points(self):
        """
        Tests the function_between_points function.
        """
        # Test it can plot without error
        pints.plot.function_between_points(self.log_posterior,
                                           self.real_parameters * 0.8,
                                           self.real_parameters * 1.2)

        # Check the two points are reversible
        pints.plot.function_between_points(self.log_posterior,
                                           self.real_parameters * 1.2,
                                           self.real_parameters * 0.8)

        # Check padding gives no error
        pints.plot.function_between_points(self.log_posterior,
                                           self.real_parameters * 0.8,
                                           self.real_parameters * 1.2,
                                           padding=0.5)

        # Check evaluations gives no error
        pints.plot.function_between_points(self.log_posterior,
                                           self.real_parameters * 0.8,
                                           self.real_parameters * 1.2,
                                           evaluations=5)

        # Check invalid function input
        self.assertRaises(
            ValueError, pints.plot.function_between_points,
            self.real_parameters,
            self.real_parameters * 1.2,
            self.real_parameters * 0.8
        )

        # Check invalid dimension input
        self.assertRaises(
            ValueError, pints.plot.function_between_points,
            self.log_posterior,
            list(self.real_parameters) + [0],
            self.real_parameters * 0.8
        )

        # Check invalid padding input
        self.assertRaises(
            ValueError, pints.plot.function_between_points,
            self.log_posterior,
            self.real_parameters * 1.2,
            self.real_parameters * 0.8,
            padding=-1
        )

        # Check invalid evaluations input
        self.assertRaises(
            ValueError, pints.plot.function_between_points,
            self.log_posterior,
            self.real_parameters * 1.2,
            self.real_parameters * 0.8,
            evaluations=-1
        )

    def test_histogram(self):
        """
        Tests the histogram function.
        """
        few_samples = self.samples[:, ::10, :]
        # Test it can plot without error
        fig, axes = pints.plot.histogram(self.samples,
                                         ref_parameters=self.real_parameters)
        # Check it returns matplotlib figure and axes
        # self.assertIsInstance(axes, self.matplotlibAxesClass)

        # Test compatiblity with one chain only
        pints.plot.histogram([self.samples[0]])

        # Check n_percentiles gives no error
        pints.plot.histogram(few_samples, n_percentiles=50)

        # Check invalid ref_parameter input
        self.assertRaises(
            ValueError, pints.plot.histogram,
            self.samples, [self.real_parameters[0]]
        )

    def test_trace(self):
        """
        Tests the trace function.
        """
        few_samples = self.samples[:, ::10, :]
        # Test it can plot without error
        fig, axes = pints.plot.trace(self.samples,
                                     ref_parameters=self.real_parameters)

        # Test compatiblity with one chain only
        pints.plot.trace([self.samples[0]])

        # Check n_percentiles gives no error
        pints.plot.trace(few_samples, n_percentiles=50)

        # Check invalid ref_parameter input
        self.assertRaises(
            ValueError, pints.plot.trace,
            self.samples, [self.real_parameters[0]]
        )

    def test_autocorrelation(self):
        """
        Tests the autocorrelation function.
        """
        # Test it can plot without error
        pints.plot.autocorrelation(self.samples[0], max_lags=20)

        # Check invalid input of samples
        self.assertRaises(
            ValueError, pints.plot.autocorrelation, self.samples
        )

    def test_series(self):
        """
        Tests the series function.
        """
        few_samples = self.samples[0][::30, :]
        # Test it can plot without error
        pints.plot.series(self.samples[0], self.problem)

        # Test thinning gives no error
        pints.plot.series(few_samples, self.problem, thinning=1)

        # Check invalid input of samples
        self.assertRaises(
            ValueError, pints.plot.series, self.samples, self.problem
        )

    def test_pairwise(self):
        """
        Tests the pairwise function.
        """
        few_samples = self.samples[0][::30, :]
        # Test it can plot without error
        pints.plot.pairwise(self.samples[0],
                            ref_parameters=self.real_parameters)

        # Test kde gives no error
        pints.plot.pairwise(few_samples, kde=True)

        # Test opacity gives no error
        pints.plot.pairwise(few_samples, opacity=0.2)

        # Test n_percentiles gives no error
        pints.plot.pairwise(few_samples, n_percentiles=50)

        # Check invalid input of samples
        self.assertRaises(
            ValueError, pints.plot.pairwise, self.samples
        )


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
