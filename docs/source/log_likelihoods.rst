***************
Log-likelihoods
***************

.. module:: pints

:class:`LogLikelihoods<pints.LogLikelihood>` are callable objects that
calculate the logarithm of the likelihood that a given parameter set gave rise
to a :class:`problem's<pints.SingleSeriesProblem>` data.

Example::

    loglikelihood = pints.UnknownNoiseLogLikelihood(problem)
    x = [1, 2, 3]
    fx = loglikelihood(x)

.. autoclass:: KnownNoiseLogLikelihood

.. autoclass:: UnknownNoiseLogLikelihood

.. autoclass:: ScaledLogLikelihood

.. autoclass:: SumOfIndependentLogLikelihoods

