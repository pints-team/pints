#!/usr/bin/env python
#
# Tests if pints can be loaded without issues.
#
import unittest


class TestBasics(unittest.TestCase):
    def test_import(self):
        import pints
        pints.version() # Avoid 'unused import' warnings
