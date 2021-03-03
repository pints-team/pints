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

# Consistent unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

try:
    from pints.interfaces.stan import StanLogPDF
    have_stan = True
except ImportError:
    have_stan = False


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

        # create eight schools model
        cls.code2 = """
        data {
            int<lower=0> J;
            real y[J];
            real<lower=0> sigma[J];
        }

        parameters {
            real<lower=0> tau;
            real theta_tilde[J];
            real mu;
        }

        transformed parameters {
            real theta[J];
            for (j in 1:J)
                theta[j] = mu + tau * theta_tilde[j];
        }

        model {
            mu ~ normal(0, 5);
            tau ~ cauchy(0, 5);
            theta_tilde ~ normal(0, 1);
            y ~ normal(theta, sigma);
        }
        """
        model = pints.toy.EightSchoolsLogPDF()
        cls.data2 = model.data()

    def test_calling(self):
        # tests instantiation
        #stanmodel = StanLogPDF(stan_code=self.code)

        if True:

            for i in range(50):
                print('Run', i)

                stanmodel = StanLogPDF(stan_code=self.code)

                # test mistakenly calling model before data supplied
                x = [1, 2]
                self.assertRaises(RuntimeError, stanmodel, x)
                self.assertRaises(RuntimeError, stanmodel.evaluateS1, x)

                # test vals and sensitivities: first supply data
                data = {'y': [0], 'N': 1}
                stanmodel.update_data(stan_data=data)

                # check getters
                self.assertEqual(stanmodel.names()[0], 'mu')
                self.assertEqual(stanmodel.names()[1], 'sigma')
                self.assertEqual(stanmodel.n_parameters(), 2)

                # add log(2) since this accounts for constant
                x = [1, 2]
                fx = stanmodel(x)
                # names ['mu', 'sigma']
                # self._names ['sigma', 'mu']
                # self._index [1, 0]
                # self._dict {'mu': [], 'sigma': []}

                val, dp = stanmodel.evaluateS1(x)
                self.assertAlmostEqual(fx, -0.8181471805599453)
                self.assertEqual(val, fx)
                self.assertEqual(dp[0], -0.25)
                self.assertEqual(dp[1], -0.375)

        if False:
            stanmodel = StanLogPDF(stan_code=self.code)

            # change data
            stanmodel.update_data(stan_data={'N': 2, 'y': [3, 4]})
            self.assertEqual(stanmodel.names()[0], 'mu')
            self.assertEqual(stanmodel.names()[1], 'sigma')
            self.assertEqual(stanmodel.n_parameters(), 2)

            #  names ['mu', 'sigma']
            #  self._names ['sigma', 'mu']
            #  self._index [1, 0]
            #  self._dict {'mu': [], 'sigma': []}
            #[2, 1]
            #[2, 1]

            x = [1, 2]
            fx = stanmodel(x)
            val, dp = stanmodel.evaluateS1(x)
            self.assertAlmostEqual(fx, -3.011294361119891)
            self.assertEqual(val, fx)
            self.assertEqual(dp[0], 1.25)
            self.assertEqual(dp[1], 0.625)

        if False:
            stanmodel = StanLogPDF(stan_code=self.code)

            # test -inf returned with negative sd
            x_err = [1, -1]
            f_err = stanmodel(x_err)
            val, dp = stanmodel.evaluateS1(x_err)
            self.assertEqual(f_err, -np.inf)
            self.assertEqual(val, -np.inf)

            # check constrained model
            stanmodel.update_data(stan_data=self.data)
            stanmodel1 = StanLogPDF(stan_code=self.code1, stan_data=self.data)

            # note the below contains the Jacobian transform -log(2)
            # so subtract this to make it equal to above
            y = [1, np.log(2)]
            fx = stanmodel(x)
            fy = stanmodel1(y)
            val, dp = stanmodel1.evaluateS1(y)
            self.assertEqual(fy - np.log(2), fx)
            self.assertEqual(val, fy)
            self.assertEqual(dp[0], -0.25)
            self.assertEqual(dp[1], 0.25)


    def test_vector_parameters_model(self):
        # tests interface with stan models with vectorised parameters

        # data2 = {'J': 8, 'y': self._y_j, 'sigma': self._sigma_j}

        if False:
            stanmodel = StanLogPDF(stan_code=self.code2, stan_data=self.data2)
            #tanmodel(np.random.uniform(size=10))
            stanmodel(np.arange(10))

        # Default parameter order (mu, tau, theta)
        # (stanfit.unconstrained_param_names())
        # names ['mu', 'tau', 'theta_tilde.1', 'theta_tilde.2', 'theta_tilde.3', 'theta_tilde.4', 'theta_tilde.5', 'theta_tilde.6', 'theta_tilde.7', 'theta_tilde.8']

        # self._initialise_dict_index(names)
        # self._names ['theta_tilde', 'mu', 'tau']
        # self._index [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]

        # self._dict = {self._names[i]: [] for i in range(len(self._names))}
        # self._dict {'theta_tilde': [], 'mu': [], 'tau': []}
        # [0 1 2 3 4 5 6 7 8 9]
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Another run
        # names ['mu', 'tau', 'theta_tilde.1', 'theta_tilde.2', 'theta_tilde.3', 'theta_tilde.4', 'theta_tilde.5', 'theta_tilde.6', 'theta_tilde.7', 'theta_tilde.8']
        # self._names ['theta_tilde', 'tau', 'mu']
        # self._index [2, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        # self._dict {'theta_tilde': [], 'tau': [], 'mu': []}
        # [0 1 2 3 4 5 6 7 8 9]
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Another rnu
        # names ['mu', 'tau', 'theta_tilde.1', 'theta_tilde.2', 'theta_tilde.3', 'theta_tilde.4', 'theta_tilde.5', 'theta_tilde.6', 'theta_tilde.7', 'theta_tilde.8']
        # self._names ['mu', 'tau', 'theta_tilde']
        # self._index [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        # self._dict {'mu': [], 'tau': [], 'theta_tilde': []}
        # [0 1 2 3 4 5 6 7 8 9]
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # With tau and mu swapped in parameter vector (tau, mu, theta)
        # names ['tau', 'mu', 'theta_tilde.1', 'theta_tilde.2', 'theta_tilde.3', 'theta_tilde.4', 'theta_tilde.5', 'theta_tilde.6', 'theta_tilde.7', 'theta_tilde.8']
        # self._names ['mu', 'tau', 'theta_tilde']
        # self._index [1, 0, 2, 2, 2, 2, 2, 2, 2, 2]
        # self._dict {'mu': [], 'tau': [], 'theta_tilde': []}
        # [0 1 2 3 4 5 6 7 8 9]
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # With order (tau, theta, mu)
        # names ['tau', 'theta_tilde.1', 'theta_tilde.2', 'theta_tilde.3', 'theta_tilde.4', 'theta_tilde.5', 'theta_tilde.6', 'theta_tilde.7', 'theta_tilde.8', 'mu']
        # self._names ['mu', 'theta_tilde', 'tau']
        # self._index [2, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        # self._dict {'mu': [], 'theta_tilde': [], 'tau': []}
        # [0 1 2 3 4 5 6 7 8 9]
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



if __name__ == '__main__':
    import sys
    print(sys.version)

    unittest.main()
