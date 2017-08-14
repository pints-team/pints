#
# Tests if the electrochemistry stuff can be loaded without issues.
#
import unittest
class TestBasics(unittest.TestCase):
    def test_import(self):
        import pints
        import electrochemistry
        #self.assertEqual(1, 2)
    def test_hello(self):
        print('HELLO')
        self.assertEqual(3, 4)
        
