#!/usr/bin/env python3
#
# Tests the version number.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2019, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
import unittest
import pints


class TestVersion(unittest.TestCase):
    """
    Tests the version number information.
    """

    def test_version_properties(self):

        self.assertEqual(type(pints.__version_int__), tuple)
        self.assertEqual(len(pints.__version_int__), 3)
        self.assertEqual(type(pints.__version_int__[0]), int)
        self.assertEqual(type(pints.__version_int__[1]), int)
        self.assertEqual(type(pints.__version_int__[2]), int)

        self.assertEqual(
            pints.__version__,
            '.'.join([str(x) for x in pints.__version_int__])
        )

    def test_version_method(self):

        self.assertEqual(pints.version(), pints.__version_int__)
        self.assertEqual(pints.version(True), 'Pints ' + pints.__version__)

