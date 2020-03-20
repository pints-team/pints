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
import pints.interfaces

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


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
        }
        model {
            mu ~ normal(0, 10);
            y ~ normal(mu, 1);
        }'''

        N = 10
        cls.data = {'N': N, 'y': np.zeros(N)}

    def test_instantiation(self):
        # tests instantiation
        stanmodel = pints.interfaces.StanLogPDF(stan_code=self.code,
                                                stan_data=self.data)

        # test vals and sensitivities
        data = np.ones(10)
        self.assertEqual(stanmodel(data), -5.005)
        val, dp = stanmodel.evaluateS1(data)
        self.assertEqual(val, stanmodel(data))
        self.assertEqual(dp[0], -10.01)

        # check getters
        self.assertEqual(stanmodel.names()[0], 'mu')
        self.assertEqual(stanmodel.n_parameters(), 1)
        various = stanmodel.pickled_form()
        stanmodel1 = various['StanLogPDF']
        self.assertEqual(stanmodel(data), stanmodel1(data))


if __name__ == '__main__':
    unittest.main()
