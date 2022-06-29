import unittest

import test_rock
import test_stress
import test_tensor
import test_wellbore
import test_model

loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(test_stress))
suite.addTests(loader.loadTestsFromModule(test_rock))
suite.addTests(loader.loadTestsFromModule(test_tensor))
suite.addTests(loader.loadTestsFromModule(test_wellbore))
suite.addTests(loader.loadTestsFromModule(test_model))


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite)
