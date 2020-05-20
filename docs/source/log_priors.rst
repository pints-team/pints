**********
Log-priors
**********

.. currentmodule:: pints

A number of :class:`LogPriors<pints.LogPrior>` are provided for use in e.g.
Bayesian inference.

Example::

    p = pints.GaussianLogPrior(mean=0, variance=1)
    x = p(0.1)


Beta-distributed parameter log-prior
************************************

.. autoclass:: BetaLogPrior

Cauchy-distributed parameter log-prior
**************************************

.. autoclass:: CauchyLogPrior

Composed log-prior
******************

.. autoclass:: ComposedLogPrior

Exponentially-distributed parameter log-prior
*********************************************

.. autoclass:: ExponentialLogPrior

Gamma-distributed parameter log-prior
*************************************

.. autoclass:: GammaLogPrior

Gaussian-distributed parameter log-prior
****************************************

.. autoclass:: GaussianLogPrior

Half-Cauchy-distributed parameter log-prior
*******************************************

.. autoclass:: HalfCauchyLogPrior

Inverse Gamma-distributed parameter log-prior
*********************************************

.. autoclass:: InverseGammaLogPrior

Log-normally-distributed parameter log-prior
********************************************

.. autoclass:: LogNormalLogPrior

Multivariate-Gaussian-distributed parameters log-prior
******************************************************

.. autoclass:: MultivariateGaussianLogPrior

Normally-distributed parameter log-prior
****************************************

.. autoclass:: NormalLogPrior

Student-t-distributed parameter log-prior
*****************************************

.. autoclass:: StudentTLogPrior

Uniformly-distributed parameter log-prior
*****************************************

.. autoclass:: UniformLogPrior

