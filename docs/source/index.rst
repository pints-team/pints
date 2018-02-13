.. Root of all pints docs

.. _GitHub: https://github.com/martinjrobins/pints
.. _Detailed examples: https://github.com/martinjrobins/pints/blob/master/examples/EXAMPLES.md

Welcome to the pints documentation
==================================

**Pints** is hosted on GitHub_, where you can find **downloads** and
**installation instructions**.

`Detailed examples`_ can also be found there.

*This* page provides the *API*, or *developer documentation* for ``pints``.

* :ref:`genindex`
* :ref:`search`

Contents
========

.. module:: pints

.. toctree::

    boundaries
    core_classes_and_methods
    diagnostic_plots
    error_measures
    function_evaluation
    io
    log_likelihoods
    log_pdfs
    log_priors
    mcmc_samplers/index
    nested_samplers/index
    optimisers/index
    toy/index
    utilities

Hierarchy of methods
====================

Pints contains different types of methods, that can be roughly arranged into a
hierarhcy, as follows.

Sampling
--------

#. MCMC

   - :class:`AdaptiveCovarianceMCMC`, works on any :class:`LogPDF`.
   - :class:`DifferentialEvolutionMCMC`, works on any :class:`LogPDF`.
   - DREAM

#. Nested sampling

   - :class:`NestedEllipsoidSampler`, requires a :class:`LogLikelihood` and a
     :class:`LogPrior` that can be sampled from.
   - :class:`NestedRejectionSampler`, requires a :class:`LogLikelihood` and a
     :class:`LogPrior` that can be sampled from.

#. Particle based samplers

   - PopulationMCMC
   - SMC

#. Likelihood free sampling (Need distance between data and states, e.g. least squares?)

   - ABC-MCMC
   - ABC-SMC

#. Score function based (Need derivatives of LogPosterior)

   - MALA
   - HMC
   - NUTS

#. Differential geometric methods (Need Hessian of LogPosterior)

   - smMALA
   - RMHMC


Optimisation
------------

All methods shown here are derivative-free methods that work on any
:class:`ErrorMeasure` or :class:`LogPDF`.

1. Particle-based methods

   - Evolution strategies (global/local methods)

     - :class:`CMA-ES`
     - :class:`SNES`
     - :class:`xNES`

   - :class:`PSO` (global method)

