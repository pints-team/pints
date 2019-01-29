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

# Select matplotlib backend that doesn't require a screen
matplotlib.use('Agg')

debug = False


class TestPlot(unittest.TestCase):
    """
    Tests Pints plot methods.
    """
    def __init__(self, name):
        super(TestPlot, self).__init__(name)

        # Create toy model (single output)
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

        # Create toy model (multi-output)
        self.model2 = toy.LotkaVolterraModel()
        self.real_parameters2 = self.model2.suggested_parameters()
        self.times2 = self.model2.suggested_times()[::10]  # down sample it
        self.values2 = self.model2.simulate(self.real_parameters2, self.times2)

        # Add noise
        self.noise2 = 0.05
        self.values2 += np.random.normal(0, self.noise2, self.values2.shape)

        # Create an object with links to the model and time series
        self.problem2 = pints.MultiOutputProblem(
            self.model2, self.times2, self.values2)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        self.log_prior2 = pints.UniformLogPrior([1, 1, 1, 1], [6, 6, 6, 6])
        # Create a log likelihood
        self.log_likelihood2 = pints.KnownNoiseLogLikelihood(self.problem2,
                                                             self.noise2)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        self.log_posterior2 = pints.LogPosterior(
            self.log_likelihood2, self.log_prior2)

        # Run MCMC
        self.x02 = [
            self.real_parameters2 * 1.1,
            self.real_parameters2 * 0.9,
            self.real_parameters2 * 1.05
        ]
        mcmc = pints.MCMCSampling(self.log_posterior2, 3, self.x02,
                                  method=pints.AdaptiveCovarianceMCMC)
        mcmc.set_max_iterations(300)  # make it as small as possible
        mcmc.set_log_to_screen(False)
        self.samples2 = mcmc.run()

        # Create toy model (single-output, single-parameter)
        self.real_parameters3 = [0]
        self.log_posterior3 = toy.NormalLogPDF(self.real_parameters3, [1])
        self.lower3 = [-3]
        self.upper3 = [3]

        # Run MCMC
        self.x03 = [[1], [-2], [3]]
        mcmc = pints.MCMCSampling(self.log_posterior3, 3, self.x03,
                                  method=pints.AdaptiveCovarianceMCMC)
        mcmc.set_max_iterations(300)  # make it as small as possible
        mcmc.set_log_to_screen(False)
        self.samples3 = mcmc.run()

    def test_function(self):
        """
        Tests the function function.
        """
        # Test it can plot without error
        pints.plot.function(self.log_posterior, self.real_parameters)

        # Check lower and upper bounds input gives no error
        pints.plot.function(self.log_posterior, self.real_parameters,
                            self.lower, self.upper)
        # Check invalid lower bound
        self.assertRaisesRegexp(
            ValueError,
            'Lower bounds must have same number of parameters as function\.',
            pints.plot.function, self.log_posterior,
            self.real_parameters, self.lower[:-1], self.upper
        )
        # Check invalid upper bound
        self.assertRaisesRegexp(
            ValueError,
            'Upper bounds must have same number of parameters as function\.',
            pints.plot.function, self.log_posterior,
            self.real_parameters, self.lower, self.upper[:-1]
        )

        # Check evaluations gives no error
        pints.plot.function(self.log_posterior, self.real_parameters,
                            evaluations=5)

        # Check invalid function input
        self.assertRaisesRegexp(
            ValueError, 'Given function must be pints\.LogPDF or ' +
            'pints\.ErrorMeasure\.', pints.plot.function, self.real_parameters,
            self.real_parameters
        )

        # Check invalid n_param input
        self.assertRaisesRegexp(
            ValueError, 'Given point \`x\` must have same number of parameters'
            + ' as function\.', pints.plot.function, self.log_posterior,
            list(self.real_parameters) + [0]
        )

        # Check invalid evaluations input
        self.assertRaisesRegexp(
            ValueError, 'Number of evaluations must be greater than zero\.',
            pints.plot.function, self.log_posterior,
            self.real_parameters, evaluations=-1
        )

        # Test it works with single parameter
        pints.plot.function(self.log_posterior3, self.real_parameters3)
        # Test bounds as well
        pints.plot.function(self.log_posterior3, self.real_parameters3,
                            self.lower3, self.upper3)

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
        self.assertRaisesRegexp(
            ValueError, 'Given function must be pints\.LogPDF or ' +
            'pints\.ErrorMeasure\.', pints.plot.function_between_points,
            self.real_parameters,
            self.real_parameters * 1.2,
            self.real_parameters * 0.8
        )

        # Check invalid n_param input
        self.assertRaisesRegexp(
            ValueError, 'Both points must have the same number of parameters'
            + ' as the given function\.', pints.plot.function_between_points,
            self.log_posterior,
            list(self.real_parameters) + [0],
            self.real_parameters * 0.8
        )

        # Check invalid padding input
        self.assertRaisesRegexp(
            ValueError, 'Padding cannot be negative\.',
            pints.plot.function_between_points,
            self.log_posterior,
            self.real_parameters * 1.2,
            self.real_parameters * 0.8,
            padding=-1
        )

        # Check invalid evaluations input
        self.assertRaisesRegexp(
            ValueError, 'The number of evaluations must be 3 or greater\.',
            pints.plot.function_between_points,
            self.log_posterior,
            self.real_parameters * 1.2,
            self.real_parameters * 0.8,
            evaluations=-1
        )

        # Test it works with single parameter
        pints.plot.function_between_points(self.log_posterior3,
                                           self.lower3,
                                           self.upper3)

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

        # Check invalid samples input
        self.assertRaisesRegexp(
            ValueError, 'All samples must have the same number of' +
            ' parameters\.', pints.plot.histogram,
            [self.samples[0, :, :], self.samples[1:, :, :-1]]
        )

        # Check invalid ref_parameter input
        self.assertRaisesRegexp(
            ValueError, 'Length of \`ref\_parameters\` must be same as' +
            ' number of parameters\.', pints.plot.histogram,
            self.samples, [self.real_parameters[0]]
        )

        # Test it works with single parameter
        few_samples3 = self.samples3[:, ::10, :]
        pints.plot.histogram(few_samples3)
        pints.plot.histogram(few_samples3,
                             ref_parameters=self.real_parameters3)

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

        # Check invalid samples input
        self.assertRaisesRegexp(
            ValueError, 'All samples must have the same number of' +
            ' parameters\.', pints.plot.trace,
            [self.samples[0, :, :], self.samples[1:, :, :-1]]
        )

        # Check invalid ref_parameter input
        self.assertRaisesRegexp(
            ValueError, 'Length of \`ref\_parameters\` must be same as' +
            ' number of parameters\.', pints.plot.trace,
            self.samples, [self.real_parameters[0]]
        )

        # Test it works with single parameter
        few_samples3 = self.samples3[:, ::10, :]
        pints.plot.trace(few_samples3)
        pints.plot.trace(few_samples3,
                         ref_parameters=self.real_parameters3)

    def test_autocorrelation(self):
        """
        Tests the autocorrelation function.
        """
        # Test it can plot without error
        pints.plot.autocorrelation(self.samples[0], max_lags=20)

        # Check invalid input of samples
        self.assertRaisesRegexp(
            ValueError, '\`samples\` must be of shape \(n_sample\,'
            ' n_parameters\)\.', pints.plot.autocorrelation, self.samples
        )

        # Test it works with single parameter
        pints.plot.autocorrelation(self.samples3[0], max_lags=20)

    def test_series(self):
        """
        Tests the series function.
        """
        few_samples = self.samples[0][::30, :]
        # Test it can plot without error
        pints.plot.series(self.samples[0], self.problem)

        # Test thinning gives no error
        pints.plot.series(few_samples, self.problem, thinning=1)
        # Test invalid thinning input
        self.assertRaisesRegexp(
            ValueError, 'Thinning rate must be \`None\` or an integer' +
            ' greater than zero\.', pints.plot.series, few_samples,
            self.problem, thinning=0
        )

        # Check invalid input of samples
        self.assertRaisesRegexp(
            ValueError, '\`samples\` must be of shape \(n_sample\,'
            ' n_parameters\)\.', pints.plot.series, self.samples, self.problem
        )

        # Check reference parameters gives no error
        pints.plot.series(few_samples, self.problem,
                          ref_parameters=self.real_parameters)
        # Check invalid reference parameters input
        self.assertRaisesRegexp(
            ValueError, 'Length of \`ref_parameters\` must be same as number'
            ' of parameters\.', pints.plot.series, few_samples, self.problem,
            self.real_parameters[:-2]
        )

        # Test mutli-output
        few_samples2 = self.samples2[0][::30, :]
        # Test it can plot without error
        pints.plot.series(self.samples2[0], self.problem2)

        # Test thinning gives no error
        pints.plot.series(few_samples2, self.problem2, thinning=1)
        # Test invalid thinning input
        self.assertRaisesRegexp(
            ValueError, 'Thinning rate must be \`None\` or an integer greater'
            ' than zero\.', pints.plot.series, few_samples2, self.problem2,
            thinning=0
        )

        # Check invalid input of samples
        self.assertRaisesRegexp(
            ValueError, '\`samples\` must be of shape \(n_sample\,'
            ' n_parameters\)\.', pints.plot.series, self.samples2,
            self.problem2
        )

        # Check reference parameters gives no error
        pints.plot.series(few_samples2, self.problem2,
                          ref_parameters=self.real_parameters2)
        # Check invalid reference parameters input
        self.assertRaisesRegexp(
            ValueError, 'Length of \`ref_parameters\` must be same as number'
            ' of parameters\.', pints.plot.series, few_samples2, self.problem2,
            self.real_parameters2[:-2]
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
        pints.plot.pairwise(few_samples, kde=True,
                            ref_parameters=self.real_parameters)

        # Test opacity gives no error
        pints.plot.pairwise(few_samples, opacity=0.2)

        # Test opacity auto setting gives no error
        pints.plot.pairwise(few_samples[:5, :])

        # Test n_percentiles gives no error
        pints.plot.pairwise(few_samples, n_percentiles=50)

        # Check invalid input of samples
        self.assertRaisesRegexp(
            ValueError, '\`samples\` must be of shape \(n_sample\,'
            ' n_parameters\)\.', pints.plot.pairwise, self.samples
        )

        # Check invalid ref_parameter input
        self.assertRaisesRegexp(
            ValueError, 'Length of \`ref_parameters\` must be same as number'
            ' of parameters\.', pints.plot.pairwise,
            few_samples, ref_parameters=[self.real_parameters[0]]
        )

        # Test single parameter
        few_samples3 = self.samples3[0][::30, :]
        # Check this is invalid
        self.assertRaisesRegexp(
            ValueError, 'Number of parameters must be larger than 2\.',
            pints.plot.pairwise, few_samples3
        )


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
