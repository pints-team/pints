***************
Log-likelihoods
***************

.. currentmodule:: pints

The classes below all implement the :class:`LogLikelihood` interface.
Most are :class:`ProblemLogLikelihood` implementations, which calculate a
log-likelihood based on some time-series :class:`Problem` and an assumed noise
model.
Some are methods combining other likelihoods.

Example::

    log_likelihood = pints.GaussianLogLikelihood(problem)
    x = [1, 2, 3]
    fx = log_likelihood(x)


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

.. autoclass:: PooledLogLikelihood

.. autoclass:: PooledLogPDF

.. autoclass:: ScaledLogLikelihood

.. autoclass:: StudentTLogLikelihood

.. autoclass:: SumOfIndependentLogLikelihoods

.. autoclass:: SumOfIndependentLogPDFs

.. autoclass:: UnknownNoiseLogLikelihood

