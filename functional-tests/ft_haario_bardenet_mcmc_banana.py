#!/usr/bin/env python3
#
# Functional test for Haario-Bardenet MCMC on the Banana problem
#
# unittest style
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import functional_testing
import pints


class HaarioBardenetMCMCBananaTest(functional_testing.Test):
    # This test is discovered because it is in a file called ft_...
    # It implements a shared interface, so that it can maybe call some methods
    # to customise things, or to do a set-up and tear-down

    def run(self):

        problem = ...

        controller = ...

        run

        return 1,2,3


    def run_2(self):
        # Maybe multiple tests can go in one file?
