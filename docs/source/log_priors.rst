**********
Log-priors
**********

.. module:: pints

A number of :class:`LogPriors<pints.LogPrior>` are provided for use in e.g.
Bayesian inference.

Example::

    p = pints.NormalLogPrior(mean=0, variance=1)
    x = p(0.1)

.. autoclass:: ComposedLogPrior

.. autoclass:: MultivariateNormalLogPrior

.. autoclass:: NormalLogPrior

.. autoclass:: UniformLogPrior

