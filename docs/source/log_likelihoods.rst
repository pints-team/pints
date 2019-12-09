***************
Log-likelihoods
***************

.. currentmodule:: pints

The classes below all implement the :class:`ProblemLogLikelihood` interface,
and can calculate a log-likelihood based on some time-series :class:`Problem`.

Example::

    logpdf = pints.GaussianLogLikelihood(problem)
    x = [1, 2, 3]
    fx = logpdf(x)

.. autoclass:: AR1LogLikelihood

.. autoclass:: ARMA11LogLikelihood

.. autoclass:: CauchyLogLikelihood

.. autoclass:: GaussianIntegratedUniformLogLikelihood

.. autoclass:: GaussianKnownSigmaLogLikelihood

.. autoclass:: GaussianLogLikelihood

.. autoclass:: KnownNoiseLogLikelihood

.. autoclass:: MultiplicativeGaussianLogLikelihood

.. autoclass:: ScaledLogLikelihood

.. autoclass:: StudentTLogLikelihood

.. autoclass:: UnknownNoiseLogLikelihood
