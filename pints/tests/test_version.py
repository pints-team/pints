#!/usr/bin/env python3
#
# Tests the version number.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
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


if __name__ == '__main__':
    unittest.main()
