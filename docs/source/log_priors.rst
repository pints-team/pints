**********
Log-priors
**********

.. module:: pints

A number of :class:`LogPriors<pints.LogPrior>` are provided for use in e.g.
Bayesian inference.

Example::

    p = pints.GaussianLogPrior(mean=0, variance=1)
    x = p(0.1)

.. autoclass:: CauchyLogPrior

.. autoclass:: ComposedLogPrior

.. autoclass:: GaussianLogPrior

.. autoclass:: HalfCauchyLogPrior

.. autoclass:: MultivariateGaussianLogPrior

.. autoclass:: StudentTLogPrior

.. autoclass:: UniformLogPrior

