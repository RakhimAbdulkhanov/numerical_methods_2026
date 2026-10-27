import math
import unittest

import mpmath as mp

import main


class IntegrationTests(unittest.TestCase):
    def test_simpson_polynomials(self):
        for degree in range(4):
            expected = 2**(degree + 1) / (degree + 1)
            self.assertAlmostEqual(main.simpson(lambda x: x**degree,
                                               0, 2, 10), expected)

    def test_invalid_counts(self):
        for n in (0, 1, 3, 2.0, True):
            with self.assertRaises(ValueError):
                main.simpson(main.load, 0, 24, n)

    def test_reference_against_independent_quadrature(self):
        f = lambda x: (50 + 20 * mp.sin(mp.pi*x/12)
                       + 5 * mp.exp(-mp.mpf('0.2')*(x-12)**2))
        independent = mp.quad(f, [0, 6, 12, 18, 24])
        self.assertLess(abs(independent - main.exact_integral()), mp.mpf('1e-50'))

    def test_adaptive_function_count_and_accuracy(self):
        calls = []
        def function(x):
            calls.append(x)
            return math.exp(x)
        result = main.adaptive_simpson(function, 0, 1, 1e-9)
        self.assertEqual(result.evaluations, len(calls))
        self.assertEqual(len(calls), len(set(calls)))
        self.assertLess(abs(result.value - math.expm1(1)), 1e-9)
        self.assertLessEqual(result.estimated_error, 1e-9)
        self.assertEqual(result.evaluations, 4 * result.intervals + 1)

    def test_adaptive_limits(self):
        with self.assertRaises(ValueError):
            main.adaptive_simpson(main.load, 0, 24, 0)
        with self.assertRaises(ArithmeticError):
            main.adaptive_simpson(math.exp, 0, 1, 1e-20, max_depth=0)

    def test_extrapolation(self):
        self.assertAlmostEqual(main.runge_romberg(1.01, 1.16), 1)
        value, order = main.aitken(1.01, 1.16, 3.56)
        self.assertAlmostEqual(value, 1)
        self.assertAlmostEqual(order, 4)

    def test_experiment_constraints_and_actual_error(self):
        result = main.experiment()
        self.assertLessEqual(result['opt_error'], main.TARGET)
        for n, error in zip(result['counts'], result['errors']):
            if n < result['n_opt']:
                self.assertGreater(error, main.TARGET)
        self.assertFalse(result['n0_compliant'])
        self.assertLess(result['n_opt'] / 8, 10)
        for delta, row in result['adaptive']:
            self.assertLess(main.absolute_error(row.value, result['exact']), delta)
            self.assertLessEqual(row.estimated_error, delta)


if __name__ == '__main__':
    unittest.main()
