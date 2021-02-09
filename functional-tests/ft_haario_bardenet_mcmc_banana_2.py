#!/usr/bin/env python3
#
# Functional test for Haario-Bardenet MCMC on the Banana problem
#
# PyTest style
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import functional_testing
import pints


def test_haario_bardenet_mcmc_banana(self):
    # This test is discovered because it's in a directory ft_...
    # And because its name starts with test_
    # Arguments can be passed to functional testing by using a decorator, e.g.
    #   @specialthing(1, 2, 3)
    #   def test_haario_bardenet...

    problem = ...

    controller = ...

    run

    return 1,2,3


def test_haario_norma(self):
    # Again, we could have a method's tests grouped in a single file
