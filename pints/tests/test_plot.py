#!/usr/bin/env python3
#
# Tests the PINTS plot methods.
#
# These tests all simply check that the code runs without errors. The generated
# output is not actually checked.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import pints.toy as toy
import pints.plot
import unittest
import numpy as np
import matplotlib

# Select matplotlib backend that doesn't require a screen
matplotlib.use('Agg')  # noqa


class TestPlot(unittest.TestCase):
    """
    Tests Pints plot methods.
    """

    @classmethod
    def setUpClass(cls):

        # Number of samples: Make this as small as possible to speed up testing
        n_samples = 300

        # Create toy model (single output)
        cls.model = toy.LogisticModel()
        cls.real_parameters = [0.015, 500]
        cls.times = np.linspace(0, 1000, 100)  # small problem
        cls.values = cls.model.simulate(cls.real_parameters, cls.times)

        # Add noise
        cls.noise = 10
        cls.values += np.random.normal(0, cls.noise, cls.values.shape)
        cls.real_parameters.append(cls.noise)
        cls.real_parameters = np.array(cls.real_parameters)

        # Create an object with links to the model and time series
        cls.problem = pints.SingleOutputProblem(
            cls.model, cls.times, cls.values)

        # Create a uniform prior over both the parameters and the new noise
        # variable
        cls.lower = [0.01, 400, cls.noise * 0.1]
        cls.upper = [0.02, 600, cls.noise * 100]
        cls.log_prior = pints.UniformLogPrior(
            cls.lower,
            cls.upper
        )

        # Create a log likelihood
        cls.log_likelihood = pints.GaussianLogLikelihood(cls.problem)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        cls.log_posterior = pints.LogPosterior(
            cls.log_likelihood, cls.log_prior)

        # Run MCMC
        cls.x0 = [
            cls.real_parameters * 1.1,
            cls.real_parameters * 0.9,
            cls.real_parameters * 1.05
        ]
        mcmc = pints.MCMCController(cls.log_posterior, 3, cls.x0)
        mcmc.set_max_iterations(n_samples)
        mcmc.set_log_to_screen(False)
        cls.samples = mcmc.run()

        # Create toy model (multi-output)
        cls.model2 = toy.LotkaVolterraModel()
        cls.real_parameters2 = cls.model2.suggested_parameters()
        cls.times2 = cls.model2.suggested_times()[::10]  # downsample it
        cls.values2 = cls.model2.simulate(cls.real_parameters2, cls.times2)

        # Add noise
        cls.noise2 = 0.05
        cls.values2 += np.random.normal(0, cls.noise2, cls.values2.shape)

        # Create an object with links to the model and time series
        cls.problem2 = pints.MultiOutputProblem(
            cls.model2, cls.times2, np.log(cls.values2))

        # Create a uniform prior over both the parameters and the new noise
        # variable
        cls.log_prior2 = pints.UniformLogPrior([0, 0, 0, 0], [6, 6, 6, 6])
        # Create a log likelihood
        cls.log_likelihood2 = pints.GaussianKnownSigmaLogLikelihood(
            cls.problem2, cls.noise2)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        cls.log_posterior2 = pints.LogPosterior(
            cls.log_likelihood2, cls.log_prior2)

        # Run MCMC
        cls.x02 = [
            cls.real_parameters2 * 1.1,
            cls.real_parameters2 * 0.9,
            cls.real_parameters2 * 1.05
        ]
        mcmc = pints.MCMCController(cls.log_posterior2, 3, cls.x02)
        mcmc.set_max_iterations(n_samples)
        mcmc.set_log_to_screen(False)
        cls.samples2 = mcmc.run()

        # Create toy model (single-output, single-parameter)
        cls.real_parameters3 = [0]
        cls.log_posterior3 = toy.GaussianLogPDF(cls.real_parameters3, [1])
        cls.lower3 = [-3]
        cls.upper3 = [3]

        # Run MCMC
        cls.x03 = [[1], [-2], [3]]
        mcmc = pints.MCMCController(cls.log_posterior3, 3, cls.x03)
        mcmc.set_max_iterations(n_samples)
        mcmc.set_log_to_screen(False)
        cls.samples3 = mcmc.run()

    def test_function(self):
        # Tests the function function.

        # Test it can plot without error
        pints.plot.function(self.log_posterior, self.real_parameters)

        # Check lower and upper bounds input gives no error
        pints.plot.function(self.log_posterior, self.real_parameters,
                            self.lower, self.upper)
        # Check invalid lower bound
        self.assertRaisesRegex(
            ValueError,
            'Lower bounds must have same number of parameters as function',
            pints.plot.function, self.log_posterior,
            self.real_parameters, self.lower[:-1], self.upper
        )
        # Check invalid upper bound
        self.assertRaisesRegex(
            ValueError,
            'Upper bounds must have same number of parameters as function',
            pints.plot.function, self.log_posterior,
            self.real_parameters, self.lower, self.upper[:-1]
        )

        # Check evaluations gives no error
        pints.plot.function(
            self.log_posterior, self.real_parameters, evaluations=5)

        # Check invalid function input
        self.assertRaisesRegex(
            ValueError,
            r'Given function must be pints\.LogPDF or pints\.ErrorMeasure\.',
            pints.plot.function, self.real_parameters, self.real_parameters
        )

        # Check invalid n_param input
        self.assertRaisesRegex(
            ValueError,
            r'Given point \`x\` must have same number of parameters as func',
            pints.plot.function,
            self.log_posterior, list(self.real_parameters) + [0]
        )

        # Check invalid evaluations input
        self.assertRaisesRegex(
            ValueError,
            r'Number of evaluations must be greater than zero\.',
            pints.plot.function, self.log_posterior,
            self.real_parameters, evaluations=-1
        )

        # Test it works with single parameter
        pints.plot.function(self.log_posterior3, self.real_parameters3)
        # Test bounds as well
        pints.plot.function(self.log_posterior3, self.real_parameters3,
                            self.lower3, self.upper3)

        # Close figure objects
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_function_between_points(self):
        # Tests the function_between_points function.

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
        self.assertRaisesRegex(
            ValueError, r'Given function must be pints\.LogPDF or ' +
            r'pints\.ErrorMeasure\.', pints.plot.function_between_points,
            self.real_parameters,
            self.real_parameters * 1.2,
            self.real_parameters * 0.8
        )

        # Check invalid n_param input
        self.assertRaisesRegex(
            ValueError, r'Both points must have the same number of parameters'
            r' as the given function\.', pints.plot.function_between_points,
            self.log_posterior,
            list(self.real_parameters) + [0],
            self.real_parameters * 0.8
        )

        # Check invalid padding input
        self.assertRaisesRegex(
            ValueError, r'Padding cannot be negative\.',
            pints.plot.function_between_points,
            self.log_posterior,
            self.real_parameters * 1.2,
            self.real_parameters * 0.8,
            padding=-1
        )

        # Check invalid evaluations input
        self.assertRaisesRegex(
            ValueError, r'The number of evaluations must be 3 or greater\.',
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

        # Close figure objects
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_histogram(self):
        # Tests the histogram function.

        few_samples = self.samples[:, ::10, :]
        # Test it can plot without error
        fig, axes = pints.plot.histogram(self.samples,
                                         ref_parameters=self.real_parameters)
        # Check it returns matplotlib figure and axes
        # self.assertIsInstance(axes, self.matplotlibAxesClass)

        # Test compatiblity with one chain only
        pints.plot.histogram([self.samples[0]])

        # Check kde gives no error
        pints.plot.histogram(few_samples, kde=True)

        # Check n_percentiles gives no error
        pints.plot.histogram(few_samples, n_percentiles=50)

        # Check that setting parameter names gives no error
        names = ['some_name'] * len(self.real_parameters)
        pints.plot.histogram(few_samples, parameter_names=names)

        # Check invalid parameter_names input
        self.assertRaisesRegex(
            ValueError,
            r'Length of \`parameter\_names\` must be same as number of'
            r' parameters\.',
            pints.plot.histogram, self.samples,
            parameter_names=['some_name']
        )

        # Check invalid ref_parameter input
        self.assertRaisesRegex(
            ValueError,
            r'Length of \`ref\_parameters\` must be same as number of'
            r' parameters\.',
            pints.plot.histogram, self.samples,
            ref_parameters=[self.real_parameters[0]]
        )

        # Test it works with single parameter
        few_samples3 = self.samples3[:, ::10, :]
        pints.plot.histogram(few_samples3)
        pints.plot.histogram(few_samples3,
                             ref_parameters=self.real_parameters3)

        # Close figure objects
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_trace(self):
        # Tests the trace function.

        few_samples = self.samples[:, ::10, :]
        # Test it can plot without error
        fig, axes = pints.plot.trace(self.samples,
                                     ref_parameters=self.real_parameters)

        # Test compatiblity with one chain only
        pints.plot.trace([self.samples[0]])

        # Check n_percentiles gives no error
        pints.plot.trace(few_samples, n_percentiles=50)

        # Check that setting parameter names gives no error
        names = ['some_name'] * len(self.real_parameters)
        pints.plot.trace(few_samples, parameter_names=names)

        # Check invalid samples input
        self.assertRaisesRegex(
            ValueError,
            r'All samples must have the same number of parameters\.',
            pints.plot.trace, [self.samples[0, :, :], self.samples[1:, :, :-1]]
        )

        # Check invalid parameter_names input
        self.assertRaisesRegex(
            ValueError,
            r'Length of \`parameter\_names\` must be same as number of'
            r' parameters\.',
            pints.plot.trace, self.samples,
            parameter_names=['some_name']
        )

        # Check invalid ref_parameter input
        self.assertRaisesRegex(
            ValueError,
            r'Length of \`ref\_parameters\` must be same as number of',
            pints.plot.trace, self.samples,
            ref_parameters=[self.real_parameters[0]]
        )

        # Test it works with single parameter
        few_samples3 = self.samples3[:, ::10, :]
        pints.plot.trace(few_samples3)
        pints.plot.trace(few_samples3,
                         ref_parameters=self.real_parameters3)

        # Close figure objects
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_autocorrelation(self):
        # Tests the autocorrelation function.

        # Test it can plot without error
        pints.plot.autocorrelation(self.samples[0], max_lags=20)

        # Check that setting parameter names gives no error
        names = ['some_name'] * len(self.real_parameters)
        pints.plot.autocorrelation(self.samples[0], parameter_names=names)

        # Check invalid input of samples
        self.assertRaisesRegex(
            ValueError, r'\`samples\` must be of shape \(n_sample\,'
            r' n_parameters\)\.', pints.plot.autocorrelation, self.samples
        )

        # Check invalid parameter_names input
        self.assertRaisesRegex(
            ValueError,
            r'Length of \`parameter\_names\` must be same as number of'
            r' parameters\.',
            pints.plot.autocorrelation, self.samples[0],
            parameter_names=['some_name']
        )

        # Test it works with single parameter
        pints.plot.autocorrelation(self.samples3[0], max_lags=20)

        # Close figure objects
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_series(self):
        # Tests the series function.

        few_samples = self.samples[0][::30, :]
        # Test it can plot without error
        pints.plot.series(self.samples[0], self.problem)

        # Test thinning gives no error
        pints.plot.series(few_samples, self.problem, thinning=1)
        # Test invalid thinning input
        self.assertRaisesRegex(
            ValueError, r'Thinning rate must be \`None\` or an integer'
            r' greater than zero\.', pints.plot.series, few_samples,
            self.problem, thinning=0
        )

        # Check invalid input of samples
        self.assertRaisesRegex(
            ValueError, r'\`samples\` must be of shape \(n_sample\,'
            r' n_parameters\)\.', pints.plot.series, self.samples, self.problem
        )

        # Check reference parameters gives no error
        pints.plot.series(few_samples, self.problem,
                          ref_parameters=self.real_parameters)
        # Check invalid reference parameters input
        self.assertRaisesRegex(
            ValueError, r'Length of \`ref_parameters\` must be same as number'
            r' of parameters\.', pints.plot.series, few_samples, self.problem,
            self.real_parameters[:-2]
        )

        # Test mutli-output
        few_samples2 = self.samples2[0][::30, :]
        # Test it can plot without error
        pints.plot.series(self.samples2[0], self.problem2)

        # Test thinning gives no error
        pints.plot.series(few_samples2, self.problem2, thinning=1)
        # Test invalid thinning input
        self.assertRaisesRegex(
            ValueError,
            r'Thinning rate must be \`None\` or an integer greater than zero',
            pints.plot.series, few_samples2, self.problem2, thinning=0
        )

        # Check invalid input of samples
        self.assertRaisesRegex(
            ValueError, r'\`samples\` must be of shape \(n_sample\,'
            r' n_parameters\)\.', pints.plot.series, self.samples2,
            self.problem2
        )

        # Check reference parameters gives no error
        pints.plot.series(few_samples2, self.problem2,
                          ref_parameters=self.real_parameters2)
        # Check invalid reference parameters input
        self.assertRaisesRegex(
            ValueError,
            r'Length of \`ref_parameters\` must be same as number of'
            r' parameters\.',
            pints.plot.series, few_samples2, self.problem2,
            self.real_parameters2[:-2]
        )

        # Close figure objects
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_pairwise(self):
        # Tests the pairwise function.

        few_samples = self.samples[0][::30, :]
        # Test it can plot without error
        pints.plot.pairwise(self.samples[0],
                            ref_parameters=self.real_parameters)

        # Test kde gives no error
        pints.plot.pairwise(few_samples, kde=True,
                            ref_parameters=self.real_parameters)

        # Test heatmap gives no error
        pints.plot.pairwise(few_samples, heatmap=True,
                            ref_parameters=self.real_parameters)

        # Check kde and heatmap error
        self.assertRaisesRegex(
            ValueError, r'Cannot use \`kde\` and \`heatmap\` together\.',
            pints.plot.pairwise, self.samples,
            kde=True, heatmap=True
        )

        # Test opacity gives no error
        pints.plot.pairwise(few_samples, opacity=0.2)

        # Test opacity auto setting gives no error
        pints.plot.pairwise(few_samples[:5, :])

        # Test n_percentiles gives no error
        pints.plot.pairwise(few_samples, n_percentiles=50)

        # Check that setting parameter names gives no error
        names = ['some_name'] * len(self.real_parameters)
        pints.plot.pairwise(few_samples, parameter_names=names)

        # Check invalid input of samples
        self.assertRaisesRegex(
            ValueError, r'\`samples\` must be of shape \(n_sample\,'
            r' n_parameters\)\.', pints.plot.pairwise, self.samples
        )

        # Check invalid parameter_names input
        self.assertRaisesRegex(
            ValueError,
            r'Length of \`parameter\_names\` must be same as number of'
            r' parameters\.',
            pints.plot.pairwise, few_samples,
            parameter_names=['some_name']
        )

        # Check invalid ref_parameter input
        self.assertRaisesRegex(
            ValueError, r'Length of \`ref_parameters\` must be same as number'
            r' of parameters\.', pints.plot.pairwise,
            few_samples, ref_parameters=[self.real_parameters[0]]
        )

        # Test single parameter
        few_samples3 = self.samples3[0][::30, :]
        # Check this is invalid
        self.assertRaisesRegex(
            ValueError, r'Number of parameters must be larger than 2\.',
            pints.plot.pairwise, few_samples3
        )

        # Close figure objects
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_surface(self):

        # Choose some points
        np.random.seed(1)
        points = np.random.normal(-2, 10, [100, 2])
        values = np.random.uniform(0, 10, len(points))

        # Check that duplicate points are handled
        points[:-10] = points[10:]

        # Create a plot
        pints.plot.surface(points, values)

        # Create a plot with boundaries
        b = pints.RectangularBoundaries([3, 5], [8, 7])
        pints.plot.surface(points, values, b)

        # Points must be 2-dimensional
        bad_points = np.random.uniform(-2, 10, [20, 3])
        self.assertRaisesRegex(
            ValueError, r'two-dimensional parameters',
            pints.plot.surface, bad_points, values)

        # Number of values must match number of points
        bad_values = values[:-1]
        self.assertRaisesRegex(
            ValueError, r'number of values must match',
            pints.plot.surface, points, bad_values)

        # Three-dimensional boundaries
        bad_b = pints.RectangularBoundaries([0, 5, 1], [1, 9, 3])
        self.assertRaisesRegex(
            ValueError, r'boundaries must be two-dimensional',
            pints.plot.surface, points, values, bad_b)

        # Close figure objects
        import matplotlib.pyplot as plt
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
