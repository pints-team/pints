**********
Log-priors
**********

.. currentmodule:: pints

A number of :class:`LogPriors<pints.LogPrior>` are provided for use in e.g.
Bayesian inference.

Example::

    p = pints.GaussianLogPrior(mean=0, variance=1)
    x = p(0.1)

Overview:

- :class:`BetaLogPrior`
- :class:`CauchyLogPrior`
- :class:`ComposedLogPrior`
- :class:`ExponentialLogPrior`
- :class:`GammaLogPrior`
- :class:`GaussianLogPrior`
- :class:`HalfCauchyLogPrior`
- :class:`InverseGammaLogPrior`
- :class:`LogNormalLogPrior`
- :class:`LogUniformLogPrior`
- :class:`MultivariateGaussianLogPrior`
- :class:`NormalLogPrior`
- :class:`StudentTLogPrior`
- :class:`TruncatedGaussianLogPrior`
- :class:`UniformLogPrior`


.. autoclass:: BetaLogPrior

.. autoclass:: CauchyLogPrior

.. autoclass:: ComposedLogPrior

.. autoclass:: ExponentialLogPrior

.. autoclass:: GammaLogPrior

.. autoclass:: GaussianLogPrior

.. autoclass:: HalfCauchyLogPrior

.. autoclass:: InverseGammaLogPrior

.. autoclass:: LogNormalLogPrior

.. autoclass:: LogUniformLogPrior

.. autoclass:: MultivariateGaussianLogPrior

.. autoclass:: NormalLogPrior

.. autoclass:: StudentTLogPrior

.. autoclass:: TruncatedGaussianLogPrior

.. autoclass:: UniformLogPrior
