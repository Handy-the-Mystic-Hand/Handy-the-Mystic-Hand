import unittest

# This will discover all tests in the 'tests' directory and run them.
test_suite = unittest.TestLoader().discover('backend/tests')
unittest.TextTestRunner().run(test_suite)


