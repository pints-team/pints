.. _mcmc:

.. module:: pints


****
MCMC
****

Pints provides a number of MCMC methods, all implementing the :class:`MCMC`
interface, that can be used to sample from an unknown
:class:`PDF<pints.LogPDF>` (usually a Bayesian
:class:`Posterior<pints.LogPosterior>`).

.. toctree::

    mcmc
    adaptive_covariance_mcmc
    differential_evolution_mcmc
    dream_mcmc

