***************
Log-likelihoods
***************

.. currentmodule:: pints

The classes below all implement the :class:`ProblemLogLikelihood` interface,
and can calculate a log-likelihood based on some time-series :class:`Problem`
and an assumed noise model.

Example::

    logpdf = pints.GaussianLogLikelihood(problem)
    x = [1, 2, 3]
    fx = logpdf(x)

Overview:

- :class:`AR1LogLikelihood`
- :class:`ARMA11LogLikelihood`
- :class:`CauchyLogLikelihood`
- :class:`ConstantAndMultiplicativeGaussianLogLikelihood`
- :class:`GaussianIntegratedLogUniformLogLikelihood`
- :class:`GaussianIntegratedUniformLogLikelihood`
- :class:`GaussianKnownSigmaLogLikelihood`
- :class:`GaussianLogLikelihood`
- :class:`KnownNoiseLogLikelihood`
- :class:`LogNormalLogLikelihood`
- :class:`MultiplicativeGaussianLogLikelihood`
- :class:`ScaledLogLikelihood`
- :class:`StudentTLogLikelihood`
- :class:`UnknownNoiseLogLikelihood`


.. autoclass:: AR1LogLikelihood

.. autoclass:: ARMA11LogLikelihood

.. autoclass:: CauchyLogLikelihood

.. autoclass:: ConstantAndMultiplicativeGaussianLogLikelihood

.. autoclass:: GaussianIntegratedLogUniformLogLikelihood

.. autoclass:: GaussianIntegratedUniformLogLikelihood

.. autoclass:: GaussianKnownSigmaLogLikelihood

.. autoclass:: GaussianLogLikelihood

.. autoclass:: KnownNoiseLogLikelihood

.. autoclass:: LogNormalLogLikelihood

.. autoclass:: MultiplicativeGaussianLogLikelihood

.. autoclass:: ScaledLogLikelihood

.. autoclass:: StudentTLogLikelihood

.. autoclass:: UnknownNoiseLogLikelihood
