*************
MCMC Samplers
*************

.. currentmodule:: pints

Pints provides a number of MCMC methods, all implementing the :class:`MCMC`
interface, that can be used to sample from an unknown
:class:`PDF<pints.LogPDF>` (usually a Bayesian
:class:`Posterior<pints.LogPosterior>`).

.. toctree::

    running
    base_classes
    adaptive_covariance_mcmc
    differential_evolution_mcmc
    dream_mcmc
    emcee_hammer_mcmc
    hamiltonian_mcmc
    mala_mcmc
    metropolis_mcmc
    population_mcmc
    slice_doubling_mcmc
    slice_stepout_mcmc
    slice_generalised_elliptical_mcmc
