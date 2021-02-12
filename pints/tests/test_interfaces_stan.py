#!/usr/bin/env python
#
# Tests interface with Stan
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import pints
import unittest
import numpy as np

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

try:
    import pints.interfaces
    have_stan = True
except ImportError:
    have_stan = False


@unittest.skipIf(not have_stan, 'PyStan not found')
class TestStanLogPDF(unittest.TestCase):
    """ Tests StanLogPDF. """

    @classmethod
    def setUpClass(cls):
        """ Set up problem for tests. """

        # Create toy model
        cls.code = '''
        data {
            int<lower=0> N;
            real y[N];
        }

        parameters {
            real mu;
            real sigma;
        }

        model {
            y ~ normal(mu, sigma);
        }'''

        cls.code1 = '''
        data {
            int<lower=0> N;
            real y[N];
        }

        parameters {
            real mu;
            real<lower=0> sigma;
        }

        model {
            y ~ normal(mu, sigma);
        }'''

        cls.data = {'N': 1, 'y': [0]}

    def test_calling(self):
        # tests instantiation
        stanmodel = pints.interfaces.StanLogPDF(stan_code=self.code)

        # tests mistakenly calling model before data supplied
        x = [1, 2]
        self.assertRaises(RuntimeError, stanmodel, x)
        self.assertRaises(RuntimeError, stanmodel.evaluateS1, x)

        # test vals and sensitivities: first supply data
        stanmodel.create_stan_model_fit(stan_data=self.data)
        # add log(2) since this accounts for constant
        self.assertAlmostEqual(stanmodel(x), -0.8181471805599453)
        val, dp = stanmodel.evaluateS1(x)
        self.assertEqual(val, stanmodel(x))
        self.assertEqual(dp[0], -0.25)
        self.assertEqual(dp[1], -0.375)

        # check getters
        self.assertEqual(stanmodel.names()[0], 'mu')
        self.assertEqual(stanmodel.names()[1], 'sigma')
        self.assertEqual(stanmodel.n_parameters(), 2)

        # change data
        stanmodel.create_stan_model_fit(stan_data={'N': 2, 'y': [3, 4]})
        self.assertAlmostEqual(stanmodel(x), -3.011294361119891)
        val, dp = stanmodel.evaluateS1(x)
        self.assertEqual(val, stanmodel(x))
        self.assertEqual(dp[0], 1.25)
        self.assertEqual(dp[1], 0.625)

        # check constrained model
        stanmodel.create_stan_model_fit(stan_data=self.data)
        stanmodel1 = pints.interfaces.StanLogPDF(stan_code=self.code1,
                                                 stan_data=self.data)
        # note the below contains the Jacobian transform -log(2)
        # so subtract this to make it equal to above
        y = [1, np.log(2)]
        self.assertEqual(stanmodel1(y) - np.log(2), stanmodel(x))
        val, dp = stanmodel1.evaluateS1(y)
        self.assertEqual(val, stanmodel1(y))
        self.assertEqual(dp[0], -0.25)
        self.assertEqual(dp[1], 0.25)


if __name__ == '__main__':
    unittest.main()
