#!/usr/bin/env python3
#
# Tests if pints can be loaded without issues.
#
import unittest


class TestBasics(unittest.TestCase):
    def test_import(self):
        import pints
        del(pints)
