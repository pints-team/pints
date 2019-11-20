#
# Test the methods in pints.residuals_diagnostics
#
# This file is part of PINTS
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing informating, see the LICENSE file distributed with the PINTS
#  software package.
#
import pints
import pints.residuals_diagnostics
import pints.toy as toy
import unittest
import numpy as np
import matplotlib

# Select matplotlib backend that doesn't require a screen
matplotlib.use('Agg')


class TestResidualsDiagnostics(unittest.TestCase):
    """
    Tests Pints residuals diagnostics methods.
    """
    def __init__(self, name):
        super(TestResidualsDiagnostics, self).__init__(name)

        # Create a single output optimization toy model
        self.model1 = toy.LogisticModel()
        self.real_parameters1 = [0.015, 500]
        self.times1 = np.linspace(0, 1000, 100)
        self.values1 = self.model1.simulate(self.real_parameters1, self.times1)

        # Add noise
        self.noise1 = 50
        self.values1 += np.random.normal(0, self.noise1, self.values1.shape)

        # Set up optimisation problem
        self.problem1 = pints.SingleOutputProblem(
            self.model1, self.times1, self.values1)
        self.score1 = pints.SumOfSquaresError(self.problem1)
        self.boundaries1 = pints.RectangularBoundaries([0, 200], [1, 1000])
        self.x01 = np.array([0.5, 500])

        # Run the optimisation
        optimiser = pints.OptimisationController(
            self.score1,
            self.x01,
            boundaries=self.boundaries1,
            method=pints.XNES,
        )
        optimiser.set_log_to_screen(False)
        self.found_parameters1, self.found_value1 = optimiser.run()

        # Create a multiple output MCMC toy model
        self.model2 = toy.LotkaVolterraModel()
        self.real_parameters2 = self.model2.suggested_parameters()
        # Downsample the times for speed
        self.times2 = self.model2.suggested_times()[::10]
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
        self.log_likelihood2 = pints.GaussianKnownSigmaLogLikelihood(
            self.problem2, self.noise2)

        # Create an un-normalised log-posterior (log-likelihood + log-prior)
        self.log_posterior2 = pints.LogPosterior(
            self.log_likelihood2, self.log_prior2)

        # Run MCMC
        self.x02 = [
            self.real_parameters2 * 1.1,
            self.real_parameters2 * 0.9,
            self.real_parameters2 * 1.05
        ]
        mcmc = pints.MCMCController(self.log_posterior2, 3, self.x02)
        mcmc.set_max_iterations(300)  # make it as small as possible
        mcmc.set_log_to_screen(False)
        self.samples2 = mcmc.run()

    def test_calculate_residuals(self):
        """
        Test the calculate_residuals function.
        """
        # Test that it runs on an optimisation result
        fn_residuals = pints.residuals_diagnostics.calculate_residuals(
            np.array([self.found_parameters1]), self.problem1)

        # Test that it runs on MCMC samples without thinning
        pints.residuals_diagnostics.calculate_residuals(self.samples2[0],
                                                        self.problem2)

        # Test that it runs on MCMC samples with thinning
        pints.residuals_diagnostics.calculate_residuals(self.samples2[0],
                                                        self.problem2,
                                                        thinning=10)

        # Check the message when the input is wrong dimension
        self.assertRaisesRegexp(
            ValueError,
            r'\`parameters\` must be of shape \(n_samples\, n_parameters\)\.',
            pints.residuals_diagnostics.calculate_residuals,
            self.samples2,
            self.model2
        )

        # Check the message when the thinning is invalid
        self.assertRaisesRegexp(
            ValueError,
            r'Thinning rate must be \`None\` or an integer greater than '
            r'zero\.',
            pints.residuals_diagnostics.calculate_residuals,
            self.samples2[0],
            self.problem2,
            thinning=0
        )

        # Test that the calculated residuals are correct. Compare the manual
        # calculation of model predictions minus actual values to the function
        # return from above
        manual_residuals = self.problem1.values() - \
                           self.problem1.evaluate(self.found_parameters1)
        fn_residuals = fn_residuals[0, 0, :]
        self.assertTrue(np.allclose(manual_residuals, fn_residuals))

    def test_acorr(self):
        """
        Test the acorr function.
        """
        # Test that it runs without error
        pints.residuals_diagnostics.acorr(np.random.normal(0, 1, 50), 11)
        pints.residuals_diagnostics.acorr(np.random.normal(0, 1, 49), 10)

        # Test that the autocorrelations are correct. Define a fixed series
        # with precomputed autocorrelation. The ground truth autocorrelation
        # was given by the matplotlib.pyplot.acorr function
        example_series = np.array([-4.16663146, -0.06785731, -5.32403073,
                        -8.44891444, -5.73192276, 9.24119792, -8.96992,
                        -6.00931999,  2.93184439, -2.08975285])
        example_series_acorr_lag5 = np.array([-0.05123492, 0.23477934,
                        0.39043056, -0.11692329, -0.03192777, 1., -0.03192777,
                        -0.11692329,  0.39043056,  0.23477934, -0.05123492])

        # Check that the precomputed autocorrelation agrees with the
        # pints.residuals_diagnostics function
        self.assertTrue(np.allclose(example_series_acorr_lag5,
                        pints.residuals_diagnostics.acorr(example_series, 5)))

    def test_plot_residuals_autocorrelation(self):
        """
        Test the plot residuals autocorrelation function.
        """
        # Test that it runs with an optimisation result
        pints.residuals_diagnostics.plot_residuals_autocorrelation(
            np.array([self.found_parameters1]),
            self.problem1
        )

        # Test that it runs with a multiple output MCMC result
        fig = pints.residuals_diagnostics.plot_residuals_autocorrelation(
            self.samples2[0],
            self.problem2
        )

        # Test that the multiple output figure has multiple axes
        self.assertGreaterEqual(len(fig.axes), 2)

        # Test an invalid significance level
        self.assertRaisesRegexp(
            ValueError,
            r'significance level must fall between 0 and 1',
            pints.residuals_diagnostics.plot_residuals_autocorrelation,
            self.samples2[0],
            self.problem2,
            significance_level=-1
        )

        # Test an invalid credible interval
        self.assertRaisesRegexp(
            ValueError,
            r'posterior interval must fall between 0 and 1',
            pints.residuals_diagnostics.plot_residuals_autocorrelation,
            self.samples2[0],
            self.problem2,
            posterior_interval=1.5
        )
