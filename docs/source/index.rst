.. Root of all pints docs

.. _GitHub: https://github.com/pints-team/pints
.. _Detailed examples: https://github.com/pints-team/pints/blob/main/examples/README.md

.. module:: pints

Welcome to the pints documentation
==================================

**Pints** is hosted on GitHub_, where you can find **downloads** and
**installation instructions**.

`Detailed examples`_ can also be found there.

*This* page provides the *API*, or *developer documentation* for ``pints``.

* :ref:`genindex`
* :ref:`search`



Defining inference problems in PINTS
====================================

PINTS provides methods to sample distributions, implemented as a
:class:`LogPDF`, and to optimise functions, implemented as an
:class:`ErrorMeasure` or a :class:`LogPDF`.

Users can define LogPDF or ErrorMeasure implementations directly, or they can
use PINTS' :class:`ForwardModel` and problem classes to set up their problems,
and then choose one of many predefined pdfs or errors.

PINTS defines :class:`single<SingleOutputProblem>` and
:class:`multi-output<MultiOutputProblem>` problem classes that wrap around
a model and data, and over which :class:`error measures<ProblemErrorMeasure>`
or :class:`log-likelihoods<ProblemLogLikelihood>` can be defined.

To find the appropriate type of Problem to use, see the overview below:

#. Systems with a single observable output

   - Single data set: Use a :class:`SingleOutputProblem` and any of the
     appropriate error measures or log-likelihoods
   - Multiple, independent data sets: Define multiple
     :class:`SingleOutputProblems<SingleOutputProblem>` and an error measure
     / log-likelihood on each, and then combine using e.g.
     :class:`SumOfErrors` or :class:`SumOfIndependentLogPDFs`.

#. Systems with multiple observable outputs

   - Single data set: Use a :class:`MultiOutputProblem` and any of the
     appropriate error measures or log-likelihoods


Provided methods
================

PINTS contains different types of methods, that can be roughly arranged into
the classification shown below.

Sampling
--------

#. :class:`MCMC without gradients<MCMCSampler>`, work on any :class:`LogPDF`.

   - :class:`MetropolisRandomWalkMCMC`
   - Adaptive methods

     - :class:`AdaptiveCovarianceMC`
     - :class:`DramACMC`
     - :class:`HaarioACMC`
     - :class:`HaarioBardenetACMC`
     - :class:`RaoBlackwellACMC`

   - :class:`PopulationMCMC`
   - Differential evolution methods

     - :class:`DifferentialEvolutionMCMC`
     - :class:`DreamMCMC`
     - :class:`EmceeHammerMCMC`

   - Slice sampling

     - :class:`SliceDoublingMCMC`
     - :class:`SliceRankShrinkingMCMC`
     - :class:`SliceStepoutMCMC`

#. First order sensitivity MCMC samplers, require a :class:`LogPDF` that
   provides first order sensitivities.

   - :class:`Hamiltonian Monte Carlo<HamiltonianMCMC>`
   - :class:`Metropolis-Adjusted Langevin Algorithm (MALA) <MALAMCMC>`
   - :class:`Monomial Gamma HMC <MonomialGammaHamiltonianMCMC>`
   - :class:`No U-Turn Sampler with dual averaging (NUTS) <NoUTurnMCMC>`
   - :class:`RelativisticMCMC`

#. :class:`Nested sampling<NestedSampler>`, require a :class:`LogPDF` and a
   :class:`LogPrior` that can be sampled from.

   - :class:`NestedEllipsoidSampler`
   - :class:`NestedRejectionSampler`

#. :class:`ABC sampling<ABCSampler>`, require a :class:`LogPrior` that can be
   sampled from from and an :class:`ErrorMeasure`.

   - :class:`ABCSMC`
   - :class:`RejectionABC`


Optimisation
------------

1. Particle, or population-based methods, work on any :class:`ErrorMeasure` or
   :class:`LogPDF`.

   - Evolution strategies

     - :class:`CMAES`
     - :class:`SNES`
     - :class:`XNES`
     - :class:`BareCMAES`

   - :class:`PSO`

2. General derivative-free methods

   - :class:`NelderMead`

3. Gradient-descent methods, require first order sensitivities

   - :class:`GradientDescent`
   - :class:`Adam`

4. General derivative-using methods

   - :class:`IRPropMin`


Contents
========

.. toctree::
    :maxdepth: 2

    abc_samplers/index
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
    mcmc_diagnostics
    nested_samplers/index
    noise_generators
    optimisers/index
    noise_model_diagnostics
    toy/index
    toy/stochastic/index
    transformations
    utilities
