import unittest
import utils.utils as utils

class UtilsTests(unittest.TestCase):

    def test_to_camel_cases(self):
        self.assertEqual(utils.to_camel_cases('a_b_c'), 'ABC')
        self.assertEqual(utils.to_camel_cases('your_first_name'), 'YourFirstName')
        self.assertEqual(utils.to_camel_cases('_net'), 'Net')
    
    def test_make_func_call(self):
        self.assertEqual(utils.make_func_call('pow', 2, 3), 'pow(2, 3)')
        self.assertEqual(utils.make_func_call('what'), 'what()')

if __name__ == '__main__':
    unittest.main()