import math
import unittest

import main


class DifferentiationTests(unittest.TestCase):
    def test_central_difference_polynomial(self):
        self.assertAlmostEqual(
            main.central_difference(lambda x: x*x, 3.0, 0.25), 6.0)

    def test_runge_cancels_quadratic_error(self):
        self.assertAlmostEqual(main.runge_romberg(2.01, 2.04), 2.0)

    def test_aitken_geometric_error(self):
        value, order = main.aitken(2.01, 2.04, 2.16)
        self.assertAlmostEqual(value, 2.0)
        self.assertAlmostEqual(order, 2.0)

    def test_invalid_and_degenerate_input(self):
        for step in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                main.central_difference(main.moisture, 1, step)
        with self.assertRaises(ValueError):
            main.aitken(1, 1, 1)

    def test_experiment(self):
        result = main.experiment()
        self.assertEqual(result['steps'][0], 1e-20)
        self.assertEqual(result['steps'][-1], 1e3)
        self.assertEqual(result['best_error'], min(result['errors']))
        self.assertAlmostEqual(result['order'], 2.0, places=4)
        errors = [abs(value - result['exact'])
                  for _, value in result['comparisons']]
        self.assertLess(errors[3], errors[0])
        self.assertLess(errors[4], errors[0])


if __name__ == '__main__':
    unittest.main()
