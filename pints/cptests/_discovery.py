#
# Change point test discovery methods for PINTS.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import inspect

import pints.cptests as cpt


def tests(method=None):
    """
    Returns a list of all change point tests, each represented as a tuple
    ``(method, test)`` where ``method`` is the PINTS class being tested and
    ``test`` is a callable that returns the test results.

    If the optional argument ``method`` is given, only tests for this method
    are returned.
    """
    # Get all modules imported into this module
    modules = [getattr(cpt, x) for x in dir(cpt) if not x.startswith('_')]
    modules = [x for x in modules if inspect.ismodule(x)]

    # Look for (explicitly defined) tests
    tests = []
    for module in modules:
        try:
            m_method = module._method
            m_tests = module._change_point_tests
        except AttributeError:
            continue

        if method is None or method == m_method:
            for test in m_tests:
                tests.append((m_method, test))

    return tests

