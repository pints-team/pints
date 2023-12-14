#
# Change point tests for PINTS.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

# Import all problem classes straight into this module, so that they can be
# addressed as e.g. pints.cptests.RunMcmcMethodOnAnnulus.
from ._problems import (    # noqa
    RunMcmcMethodOnAnnulus,
    RunMcmcMethodOnBanana,
    RunMcmcMethodOnCone,
    RunMcmcMethodOnCorrelatedGaussian,
    RunMcmcMethodOnHighDimensionalGaussian,
    RunMcmcMethodOnMultimodalGaussian,
    RunMcmcMethodOnProblem,
    RunMcmcMethodOnTwoDimGaussian,
    RunOptimiserOnBoundedFitzhughNagumo,
    RunOptimiserOnBoundedUntransformedLogistic,
    RunOptimiserOnProblem,
    RunOptimiserOnRosenbrockError,
    RunOptimiserOnTwoDimParabola,
)

# Import all test modules (not methods!) directly into this method, so that
# they can be addressed as e.g.
# pints.cptests.dram_acmc.two_dim_gaussian().
from . import (     # noqa
    differential_evolution_mcmc,
    dram_acmc,
    dream_mcmc,
    emcee_hammer_mcmc,
    haario_acmc,
    haario_bardenet_acmc,
    hamiltonian_mcmc,
    mala_mcmc,
    metropolis_random_walk_mcmc,
    monomial_gamma_hamiltonian_mcmc,
    no_u_turn_mcmc,
    population_mcmc,
    relativistic_mcmc,
    slice_stepout_mcmc,
    slice_doubling_mcmc,
)


# Test discovery methods
from ._discovery import tests   # noqa
