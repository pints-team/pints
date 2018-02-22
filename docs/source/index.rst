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
hierarchy, as follows.

Sampling
--------

#. :class:`MCMC without gradients<MCMCSampler>`

   - :class:`AdaptiveCovarianceMCMC`, works on any :class:`LogPDF`.
   - :class:`DifferentialEvolutionMCMC`, works on any :class:`LogPDF`.
   - DREAM
   - emcee (MCMC Hammer)
   - Metropolis
   - Metropolis-Hastings
   - PopulationMCMC

#. :class:`Nested sampling<NestedSampler>`

   - :class:`NestedEllipsoidSampler`, requires a :class:`LogLikelihood` and a
     :class:`LogPrior` that can be sampled from.
   - :class:`NestedRejectionSampler`, requires a :class:`LogLikelihood` and a
     :class:`LogPrior` that can be sampled from.

#. Particle based samplers

   - SMC

#. Likelihood free sampling (Need distance between data and states, e.g. least squares?)

   - ABC-MCMC
   - ABC-SMC

#. 1st order sensitivity MCMC samplers (Need derivatives of :class:`LogPosterior`)

   - MALA
   - HMC
   - NUTS

#. Differential geometric methods (Need Hessian of :class:`LogPosterior`)

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



Problems in Pints
=================

Pints defines :class:`Problem classes<SingleSeriesProblem>` that wrap around
models and data, and over which :class:`error measures<ErrorMeasure>` or
:class:`log-likelihoods<LogLikelihoods>` can be defined.

To find the appropriate type of ``Problem`` to use, see the overview below:

#. Systems with a single observable output

   - Single data set: Use a :class:`SingleSeriesProblem` and any of the
     appropriate error measures or log-likelihoods
   - Multiple, independent data sets: Define multiple
     :class:`SingleSeriesProblems<SingleSeriesProblem>` and an error measure
     / log-likelihood on each, and then combine using e.g.
     :class:`SumOfErrors` or :class:`SumOfIndependentLogLikelihoods`.

#. Systems with multiple observable outputs

   - Not implemented yet!

