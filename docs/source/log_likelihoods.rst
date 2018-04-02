***************
Log-likelihoods
***************

.. module:: pints

:class:`LogLikelihoods<pints.LogLikelihood>` are callable objects that
calculate the logarithm of a likelihood.

Example::

    loglikelihood = pints.UnknownNoiseLogLikelihood(problem)
    x = [1, 2, 3]
    fx = loglikelihood(x)

.. autoclass:: KnownNoiseLogLikelihood

.. autoclass:: ScaledLogLikelihood

.. autoclass:: SumOfIndependentLogLikelihoods

.. autoclass:: UnknownNoiseLogLikelihood

