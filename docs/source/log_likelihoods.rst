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
- :class:`GaussianIntegratedUniformLogLikelihood`
- :class:`GaussianKnownSigmaLogLikelihood`
- :class:`GaussianLogLikelihood`
- :class:`KnownNoiseLogLikelihood`
- :class:`MultiplicativeGaussianLogLikelihood`
- :class:`ScaledLogLikelihood`
- :class:`StudentTLogLikelihood`
- :class:`UnknownNoiseLogLikelihood`

Autoregressive order 1 noise
****************************

.. autoclass:: AR1LogLikelihood

Autoregressive moving average noise
***********************************

.. autoclass:: ARMA11LogLikelihood

Cauchy noise
************

.. autoclass:: CauchyLogLikelihood

Gaussian integrated uniform noise
*********************************

.. autoclass:: GaussianIntegratedUniformLogLikelihood

Gaussian known sigma noise
**************************

.. autoclass:: GaussianKnownSigmaLogLikelihood

Gaussian noise
**************

.. autoclass:: GaussianLogLikelihood

Known noise
***********

.. autoclass:: KnownNoiseLogLikelihood

Multiplicative (heteroscedastic) Gaussian noise
***********************************************

.. autoclass:: MultiplicativeGaussianLogLikelihood

Scaled log-likelihood
*********************

.. autoclass:: ScaledLogLikelihood

Student-t noise
***************

.. autoclass:: StudentTLogLikelihood

Unknown noise
*************

.. autoclass:: UnknownNoiseLogLikelihood
