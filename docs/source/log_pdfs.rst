********
Log-PDFs
********

.. module:: pints

:class:`LogPDFs<pints.LogPDF>` are callable objects that represent
distributions, including likelihoods and Bayesian priors and posteriors.
They are unnormalised, i.e. their area does not necessarily sum up to 1, and
for efficiency reasons we always work with the logarithm e.g. a log-likelihood
instead of a likelihood.

Example::

    p = pints.NormalLogPrior(mean=0, variance=1)
    x = p(0.1)

.. autoclass:: LogPDF

.. autoclass:: LogPrior

.. autoclass:: LogPosterior

.. autoclass:: ProblemLogLikelihood

.. autoclass:: SumOfIndependentLogPDFs

