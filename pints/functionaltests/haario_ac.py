#!/usr/bin/env python3
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

from __future__ import division

import pints

from ._problems import (RunMcmcMethodOnTwoDimGaussian,
                        RunMcmcMethodOnBanana)


def test_haario_acmc_on_two_dim_gaussian():
    problem = RunMcmcMethodOnTwoDimGaussian(
        method=pints.HaarioACMC,
        n_chains=4,
        n_iterations=4000,
        n_warmup=1000
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }


def test_haario_acmc_on_banana():
    problem = RunMcmcMethodOnBanana(
        method=pints.HaarioACMC,
        n_chains=4,
        n_iterations=4000,
        n_warmup=1000
    )

    return {
        'kld': problem.estimate_kld(),
        'mean-ess': problem.estimate_mean_ess()
    }
