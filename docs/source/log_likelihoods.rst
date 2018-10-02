***************
Log-likelihoods
***************

.. module:: pints

The classes below all implement the :class:`ProblemLogLikelihood` interface,
and can calculate a log-likelihood based on some time-series :class:`Problem`.

Example::

    logpdf = pints.UnknownNoiseLogLikelihood(problem)
    x = [1, 2, 3]
    fx = logpdf(x)

.. autoclass:: CauchyLogLikelihood

.. autoclass:: KnownNoiseLogLikelihood

.. autoclass:: ScaledLogLikelihood

.. autoclass:: StudentTLogLikelihood

.. autoclass:: UnknownNoiseLogLikelihood

