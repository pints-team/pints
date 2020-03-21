**********
Log-priors
**********

.. currentmodule:: pints

A number of :class:`LogPriors<pints.LogPrior>` are provided for use in e.g.
Bayesian inference.

Example::

    p = pints.GaussianLogPrior(mean=0, variance=1)
    x = p(0.1)

.. autoclass:: BetaLogPrior

.. autoclass:: CauchyLogPrior

.. autoclass:: ComposedLogPrior

.. autoclass:: ExponentialLogPrior

.. autoclass:: GammaLogPrior

.. autoclass:: GaussianLogPrior

.. autoclass:: HalfCauchyLogPrior

.. autoclass:: InverseGammaLogPrior

.. autoclass:: LogNormalLogPrior

.. autoclass:: MultivariateGaussianLogPrior

.. autoclass:: NormalLogPrior

.. autoclass:: StudentTLogPrior

.. autoclass:: UniformLogPrior

