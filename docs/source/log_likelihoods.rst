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


.. autoclass:: AR1LogLikelihood

.. autoclass:: ARMA11LogLikelihood

.. autoclass:: CauchyLogLikelihood

.. autoclass:: CensoredGaussianLogLikelihood

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
