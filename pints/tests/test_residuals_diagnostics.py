#!/usr/bin/env python3
#
# Test the methods in pints.residuals_diagnostics
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import matplotlib
import numpy as np
import pints
import pints.residuals_diagnostics
import pints.toy as toy
import unittest

# Select matplotlib backend that doesn't require a screen
matplotlib.use('Agg')


class TestResidualsDiagnostics(unittest.TestCase):
    """
    Tests Pints residuals diagnostics methods.
    """

    @classmethod
    def setUpClass(cls):
        # Create a single output optimisation toy model
        cls.model1 = toy.LogisticModel()
        cls.real_parameters1 = [0.015, 500]
        cls.times1 = np.linspace(0, 1000, 100)
        cls.values1 = cls.model1.simulate(cls.real_parameters1, cls.times1)

        # Add noise
        cls.noise1 = 50
        cls.values1 += np.random.normal(0, cls.noise1, cls.values1.shape)

        # Set up optimisation problem
        cls.problem1 = pints.SingleOutputProblem(
            cls.model1, cls.times1, cls.values1)

        # Instead of running the optimisation, choose fixed values to serve as
        # the results
        cls.found_parameters1 = np.array([0.0149, 494.6])

        # Create a multiple output MCMC toy model
        cls.model2 = toy.LotkaVolterraModel()
        cls.real_parameters2 = cls.model2.suggested_parameters()
        # Downsample the times for speed
        cls.times2 = cls.model2.suggested_times()
        cls.values2 = cls.model2.suggested_values()

        # Set up 2-output MCMC problem
        cls.problem2 = pints.MultiOutputProblem(
            cls.model2, cls.times2, np.log(cls.values2))

        # Instead of running MCMC, generate three chains which actually contain
        # independent samples near the true values (faster than MCMC)
        samples = np.zeros((3, 50, 4))
        for chain_idx in range(3):
            for parameter_idx in range(4):
                if parameter_idx == 0 or parameter_idx == 2:
                    chain = np.random.normal(3.01, .2, 50)
                else:
                    chain = np.random.normal(1.98, .2, 50)
                samples[chain_idx, :, parameter_idx] = chain
        cls.samples2 = samples

    def test_calculate_residuals(self):
        # Test the calculate_residuals function

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
        self.assertRaisesRegex(
            ValueError,
            r'\`parameters\` must be of shape \(n_samples\, n_parameters\)\.',
            pints.residuals_diagnostics.calculate_residuals,
            self.samples2,
            self.model2
        )

        # Check the message when the thinning is invalid
        self.assertRaisesRegex(
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
        # Test the acorr function

        # Test that it runs without error
        pints.residuals_diagnostics.acorr(np.random.normal(0, 1, 50), 11)
        pints.residuals_diagnostics.acorr(np.random.normal(0, 1, 49), 10)

        # Test that the autocorrelations are correct. Define a fixed series
        # with precomputed autocorrelation. The ground truth autocorrelation
        # was given by the matplotlib.pyplot.acorr function
        example_series = np.array([-4.16663146, -0.06785731, -5.32403073,
                                   -8.44891444, -5.73192276, 9.24119792,
                                   -8.96992, -6.00931999, 2.93184439,
                                   -2.08975285])
        example_series_acorr_lag5 = np.array([-0.05123492, 0.23477934,
                                              0.39043056, -0.11692329,
                                              -0.03192777, 1., -0.03192777,
                                              -0.11692329, 0.39043056,
                                              0.23477934, -0.05123492])

        # Check that the precomputed autocorrelation agrees with the
        # pints.residuals_diagnostics function
        self.assertTrue(np.allclose(example_series_acorr_lag5,
                        pints.residuals_diagnostics.acorr(example_series, 5)))

    def test_plot_residuals_autocorrelation(self):
        # Test the plot residuals autocorrelation function

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
        self.assertRaisesRegex(
            ValueError,
            r'significance level must fall between 0 and 1',
            pints.residuals_diagnostics.plot_residuals_autocorrelation,
            self.samples2[0],
            self.problem2,
            significance_level=-1
        )

        # Test an invalid credible interval
        self.assertRaisesRegex(
            ValueError,
            r'posterior interval must fall between 0 and 1',
            pints.residuals_diagnostics.plot_residuals_autocorrelation,
            self.samples2[0],
            self.problem2,
            posterior_interval=1.5
        )

    def test_plot_residuals_vs_output(self):
        # Test the function which plots residuals against output magnitudes

        # Test that it runs with an optimisation result
        pints.residuals_diagnostics.plot_residuals_vs_output(
            np.array([self.found_parameters1]),
            self.problem1
        )

        # Test that it runs with multiple ouputs and MCMC
        fig = pints.residuals_diagnostics.plot_residuals_vs_output(
            self.samples2[0],
            self.problem2
        )

        # Test that the multiple output figure has multiple axes
        self.assertGreaterEqual(len(fig.axes), 2)

        # Check the message when the input is wrong dimension
        self.assertRaisesRegex(
            ValueError,
            r'\`parameters\` must be of shape',
            pints.residuals_diagnostics.plot_residuals_vs_output,
            self.samples2,
            self.model2
        )

        # Check the message when the thinning is invalid
        self.assertRaisesRegex(
            ValueError,
            'Thinning rate must be',
            pints.residuals_diagnostics.plot_residuals_vs_output,
            self.samples2[0],
            self.problem2,
            thinning=0
        )

    def test_plot_residuals_distance(self):
        # Test the function that plots the distance matrix of residuals

        # Test that it runs with an optimisation result
        pints.residuals_diagnostics.plot_residuals_distance(
            np.array([self.found_parameters1]),
            self.problem1
        )

        # Test that it runs with multiple ouputs and MCMC
        fig = pints.residuals_diagnostics.plot_residuals_distance(
            self.samples2[0],
            self.problem2
        )

        # Test that the multiple output figure has multiple axes
        self.assertGreaterEqual(len(fig.axes), 2)

        # Check the message when the thinning is invalid
        self.assertRaisesRegex(
            ValueError,
            'Thinning rate must be',
            pints.residuals_diagnostics.plot_residuals_distance,
            self.samples2[0],
            self.problem2,
            thinning=0
        )

    def test_plot_residuals_binned_autocorrelation(self):
        # Test the function that plots the binned residuals autocorrelations

        # Test that it runs with an optimisation result
        pints.residuals_diagnostics.plot_residuals_binned_autocorrelation(
            np.array([self.found_parameters1]),
            self.problem1,
            n_bins=5
        )

        # Test that it runs with multiple ouputs and MCMC
        fig = pints.residuals_diagnostics.\
            plot_residuals_binned_autocorrelation(
                self.samples2[0],
                self.problem2,
                n_bins=5
            )

        # Test that the multiple output figure has multiple axes
        self.assertGreaterEqual(len(fig.axes), 2)

        # Check the message when the thinning is invalid
        self.assertRaisesRegex(
            ValueError,
            'Thinning rate must be',
            pints.residuals_diagnostics.plot_residuals_binned_autocorrelation,
            self.samples2[0],
            self.problem2,
            thinning=0,
            n_bins=5
        )

        # Check the message when the number of bins is invalid
        self.assertRaisesRegex(
            ValueError,
            'n_bins must be',
            pints.residuals_diagnostics.plot_residuals_binned_autocorrelation,
            self.samples2[0],
            self.problem2,
            n_bins=-1
        )

        # Check the message when the number of bins is too big
        self.assertRaisesRegex(
            ValueError,
            'n_bins must not exceed',
            pints.residuals_diagnostics.plot_residuals_binned_autocorrelation,
            self.samples2[0],
            self.problem2,
            n_bins=1000
        )

    def test_plot_residuals_binned_std(self):
        # Test the function that plots the binned residuals standard deviation

        # Test that it runs with an optimisation result
        pints.residuals_diagnostics.plot_residuals_binned_std(
            np.array([self.found_parameters1]),
            self.problem1,
            n_bins=5
        )

        # Test that it runs with multiple ouputs and MCMC
        fig = pints.residuals_diagnostics.\
            plot_residuals_binned_std(
                self.samples2[0],
                self.problem2,
                n_bins=5
            )

        # Test that the multiple output figure has multiple axes
        self.assertGreaterEqual(len(fig.axes), 2)

        # Check the message when the thinning is invalid
        self.assertRaisesRegex(
            ValueError,
            'Thinning rate must be',
            pints.residuals_diagnostics.plot_residuals_binned_std,
            self.samples2[0],
            self.problem2,
            thinning=0,
            n_bins=5
        )

        # Check the message when the number of bins is invalid
        self.assertRaisesRegex(
            ValueError,
            'n_bins must be',
            pints.residuals_diagnostics.plot_residuals_binned_std,
            self.samples2[0],
            self.problem2,
            n_bins=-1
        )

        # Check the message when the number of bins is too big
        self.assertRaisesRegex(
            ValueError,
            'n_bins must not exceed',
            pints.residuals_diagnostics.plot_residuals_binned_std,
            self.samples2[0],
            self.problem2,
            n_bins=1000
        )


if __name__ == '__main__':
    unittest.main()
