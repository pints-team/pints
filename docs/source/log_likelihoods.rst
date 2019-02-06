***************
Log-likelihoods
***************

.. module:: pints

The classes below all implement the :class:`ProblemLogLikelihood` interface,
and can calculate a log-likelihood based on some time-series :class:`Problem`.

Example::

    logpdf = pints.GaussianLogLikelihood(problem)
    x = [1, 2, 3]
    fx = logpdf(x)

.. autoclass:: AR1LogLikelihood

.. autoclass:: ARMA11LogLikelihood

.. autoclass:: CauchyLogLikelihood

.. autoclass:: GaussianKnownSigmaLogLikelihood

.. autoclass:: GaussianLogLikelihood

.. autoclass:: ScaledLogLikelihood

.. autoclass:: StudentTLogLikelihood

