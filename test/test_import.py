#!/usr/bin/env python
#
# Simply tests if the pints module can be imported
#
import unittest

class TestUM(unittest.TestCase):
     def setUp(self):
        pass
     def test_import(self):
        import pints

if __name__ == '__main__':
    unittest.main()
