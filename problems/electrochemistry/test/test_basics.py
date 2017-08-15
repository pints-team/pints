#
# Tests if the electrochemistry stuff can be loaded without issues.
#
import unittest
class TestBasics(unittest.TestCase):
    def test_import(self):
        import pints
        import electrochemistry
        
