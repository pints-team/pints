#!/usr/bin/env python
#
# Tests interface with Stan
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import numpy as np

import pints
import pints.toy

from shared import SubCapture


try:
    from pints.interfaces.stan import StanLogPDF
    have_stan = True
except ImportError:
    have_stan = False

debug = False


@unittest.skipIf(not have_stan, 'PyStan not found')
class TestStanLogPDF(unittest.TestCase):
    """ Tests StanLogPDF. """

    @classmethod
    def setUpClass(cls):
        """ Set up problem for tests. """

        # Create toy normal models
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
            }
        '''

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
            }
        '''

        cls.data = {'N': 1, 'y': [0]}

        # Eight schools model, with 10 (not 3 or 18) parameters
        cls.code2 = '''
            data {
              int<lower=0> J;
              real y[J];
              real<lower=0> sigma[J];
            }

            parameters {
              // Lets get some real mu in here
              real mu;
              /* And a little bit of tau */
              real<lower=0> tau;
              /* Let's annoy /*
                 the developers parameters {
                 real schmu;
                 } */
              // We now define the real theta_tilde[J];
              /* real_theta_tilde[J]; ? Yes!
                 We certainly define that here. */
              real theta_tilde[J]; // This is it being defined
              /* Great job everyone! We defined
              real_theta_tilde[J];
              in this parameters{} block.
              */
            }

            transformed parameters {
              real theta[J];
              for (j in 1:J)
                theta[j] = mu + tau * theta_tilde[j];
            }

            model {
              tau ~ cauchy(0, 5);
              mu ~ normal(0, 5);
              theta_tilde ~ normal(0, 1);
              y ~ normal(theta, sigma);
            }
        '''
        model = pints.toy.EightSchoolsLogPDF()
        cls.data2 = model.data()

    def test_calling(self):

        # tests instantiation
        with SubCapture(dump_on_error=True) as c:
            stanmodel = StanLogPDF(self.code, self.data)
        if debug:
            print('# Test instantiation')
            print(c.text())

        # test vals and sensitivities
        x = [1, 2]
        with SubCapture(dump_on_error=True) as c:
            # add log(2) since this accounts for constant
            fx = stanmodel(x)
            val, dp = stanmodel.evaluateS1(x)
        if debug:
            print('# Test values and sensitivities')
            print(c.text())
        self.assertAlmostEqual(fx, -0.8181471805599453)
        self.assertEqual(val, fx)
        self.assertEqual(dp[0], -0.25)
        self.assertEqual(dp[1], -0.375)

        # check getters
        self.assertEqual(stanmodel.names()[0], 'mu')
        self.assertEqual(stanmodel.names()[1], 'sigma')
        self.assertEqual(stanmodel.n_parameters(), 2)

        # alternate data
        with SubCapture(dump_on_error=True) as c:
            stanmodel = StanLogPDF(self.code, {'N': 2, 'y': [3, 4]})
            fx = stanmodel(x)
            val, dp = stanmodel.evaluateS1(x)
        if debug:
            print('# Test change of data')
            print(c.text())
        self.assertAlmostEqual(fx, -3.011294361119891)
        self.assertEqual(val, fx)
        self.assertEqual(dp[0], 1.25)
        self.assertEqual(dp[1], 0.625)

        # test -inf returned with negative sd
        x_err = [1, -1]
        with SubCapture(dump_on_error=True) as c:
            f_err = stanmodel(x_err)
            val, dp = stanmodel.evaluateS1(x_err)
        if debug:
            print('# Test negative inf is returned')
            print(c.text())
        self.assertEqual(f_err, -np.inf)
        self.assertEqual(val, -np.inf)

        # check constrained model
        with SubCapture(dump_on_error=True) as c:
            stanmodel0 = StanLogPDF(self.code, self.data)
            stanmodel1 = StanLogPDF(code=self.code1, data=self.data)
        if debug:
            print('# Test with a constrained model')
            print(c.text())

        # note the below contains the Jacobian transform -log(2)
        # so subtract this to make it equal to above
        y = [1, np.log(2)]
        with SubCapture(dump_on_error=True) as c:
            fx = stanmodel0(x)
            fy = stanmodel1(y)
            val, dp = stanmodel1.evaluateS1(y)
        if debug:
            print('# Further constrained model test')
            print(c.text())
        self.assertEqual(fy - np.log(2), fx)
        self.assertEqual(val, fy)
        self.assertEqual(dp[0], -0.25)
        self.assertEqual(dp[1], 0.25)
        self.assertIsInstance(dp, np.ndarray)

    def test_vector_parameters_model(self):
        # tests interface with stan models with vectorised parameters
        with SubCapture(dump_on_error=True) as c:
            stanmodel = StanLogPDF(self.code2, self.data2)
            stanmodel(np.random.uniform(size=10))
        if debug:
            print('Test vectorised parameters')
            print(c.text())
        self.assertEqual(stanmodel.n_parameters(), 10)

        # Test parameter count is 2 + J
        j = 3
        data = {
            'J': j,
            'y': self.data2['y'][:j],
            'sigma': self.data2['sigma'][:j],
        }
        with SubCapture(dump_on_error=True) as c:
            stanmodel = StanLogPDF(self.code2, data)
            stanmodel(np.random.uniform(size=5))
        self.assertEqual(stanmodel.n_parameters(), 5)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
