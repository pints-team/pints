#!/usr/bin/env python3
#
# Tests the version number.
#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#
import unittest
import datetime


class TestCopyright(unittest.TestCase):
    """
    Tests that the copyright information in LICENSE.md is up-to-date.
    """

    def test_copyright(self):

        current_year = str(datetime.datetime.now().year)

        with open('LICENSE.md', 'r') as license_file:
            license_text = license_file.read()
            self.assertIn('Copyright (c) 2017-' + current_year, license_text)
