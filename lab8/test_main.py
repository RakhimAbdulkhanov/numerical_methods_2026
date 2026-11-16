import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import main


class NonlinearEquationsTests(unittest.TestCase):
    def test_known_root_all_six_methods(self):
        # рівняння x^2 - 2 = 0 має точний додатний корінь sqrt(2)
        f = lambda x: x * x - 2.0
        df = lambda x: 2.0 * x
        d2f = lambda x: 2.0
        target = math.sqrt(2.0)
        results = [
            main.simple_iteration(f, 1.5, -0.3),
            main.newton(f, df, 1.5),
            main.chebyshev(f, df, d2f, 1.5),
            main.secant(f, 1.0, 1.5),
            main.muller_parabola(f, 1.0, 2.0, 1.5),
            main.inverse_interpolation(f, 1.0, 2.0, 1.5),
        ]
        for res in results:
            self.assertAlmostEqual(res.root, target, places=10)
            self.assertLess(res.residual, main.EPS)
            self.assertEqual(len(res.history), res.iterations + 1)

    def test_muller_complex_and_quadratic_exactness(self):
        res_complex = main.muller_parabola(lambda z: z * z + 1.0, 0.0, 1.0, 0.5)
        self.assertAlmostEqual(abs(res_complex.root - 1j), 0.0, places=10)
        res_quad = main.muller_parabola(
            lambda x: x * x - 5.0 * x + 6.0, 0.0, 1.0, 1.5
        )
        self.assertAlmostEqual(res_quad.root, 2.0, places=10)

    def test_tabulation_and_brackets(self):
        f = lambda x: (x - 1.25) * (x - 2.75)
        nodes = main.tabulate(f, 0.0, 4.0, 0.5)
        self.assertEqual(len(nodes), 9)
        brackets = main.find_brackets(nodes)
        self.assertEqual(len(brackets), 2)
        self.assertEqual(brackets[0][:2], (1.0, 1.5))
        self.assertEqual(brackets[0][3], "decreasing")
        self.assertEqual(brackets[1][:2], (2.5, 3.0))
        self.assertEqual(brackets[1][3], "increasing")

    def test_horner_and_polynomial_solvers_against_numpy(self):
        # многочлен (x - 4)(x - 6)(x^2 - 2x + 2) четвертого степеня
        coeffs = [48.0, -68.0, 46.0, -12.0, 1.0]
        val, quot = main.horner_eval(coeffs, 4.0, return_quotient=True)
        self.assertAlmostEqual(val, 0.0, places=12)
        self.assertAlmostEqual(main.horner_eval(quot, 6.0), 0.0, places=12)
        real_res = main.newton_horner(coeffs, 3.5)
        lin_res = main.lin_method(coeffs, 0.8, 1.2)
        np_roots = np.roots(coeffs[::-1])
        self.assertLess(np.min(np.abs(np_roots - real_res.root)), 1e-10)
        z = complex(lin_res.alpha, lin_res.beta)
        self.assertLess(np.min(np.abs(np_roots - z)), 1e-9)
        self.assertLess(lin_res.residual, 1e-8)

    def test_invalid_inputs_and_singular_guards(self):
        with self.assertRaises(ValueError):
            main.tabulate(main.f_trans, 1.0, 1.0, 0.1)
        with self.assertRaises(ValueError):
            main.tabulate(main.f_trans, 0.0, 1.0, -0.1)
        with self.assertRaises(ValueError):
            main.simple_iteration(main.f_trans, 0.25, 0.0)
        with self.assertRaises(ArithmeticError):
            main.simple_iteration(lambda x: x * x + 1.0, 2.0, 2.0, max_iter=15)
        with self.assertRaises(ZeroDivisionError):
            main.newton(lambda x: x * x - 1.0, lambda x: 2.0 * x, 0.0)
        with self.assertRaises(ZeroDivisionError):
            main.chebyshev(
                lambda x: x * x - 1.0, lambda x: 2.0 * x, lambda x: 2.0, 0.0
            )
        with self.assertRaises(ValueError):
            main.secant(main.f_trans, 0.25, 0.25)
        with self.assertRaises(ZeroDivisionError):
            main.secant(lambda x: x * x, -1.0, 1.0)
        with self.assertRaises(ValueError):
            main.muller_parabola(main.f_trans, 0.2, 0.2, 0.3)
        with self.assertRaises(ValueError):
            main.inverse_interpolation(main.f_trans, 0.2, 0.3, 0.2)
        with self.assertRaises(ZeroDivisionError):
            main.inverse_interpolation(lambda x: x * x, -1.0, 1.0, 2.0)
        with self.assertRaises(ValueError):
            main.horner_eval([], 1.0)
        with self.assertRaises(ValueError):
            main.newton_horner([5.0], 1.0)
        with self.assertRaises(ValueError):
            main.lin_method([1.0, 2.0, 1.0], 0.5, 0.5)
        with self.assertRaises(ValueError):
            main.lin_method([-10.0, 12.0, -7.0, 1.0], 4.0, 0.1)
        with self.assertRaises(ArithmeticError):
            main.lin_method([-10.0, 12.0, -7.0, 1.0], 0.5, 0.5, max_iter=1)

    def test_file_io_in_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            tab_path = tmp / "tab.txt"
            nodes = main.tabulate(main.f_trans, 0.0, 1.0, 0.5)
            main.save_tabulation(nodes, tab_path)
            loaded = main.read_tabulation(tab_path)
            self.assertEqual(len(loaded), len(nodes))
            for (x0, y0), (x1, y1) in zip(nodes, loaded):
                self.assertAlmostEqual(x0, x1)
                self.assertAlmostEqual(y0, y1, places=9)

            poly_path = tmp / "poly.txt"
            poly_path.write_text("# коефіцієнти\n-10, 12\n-7 1\n",
                                 encoding="utf-8")
            self.assertEqual(main.read_poly_coeffs(poly_path),
                             [-10.0, 12.0, -7.0, 1.0])

            empty_path = tmp / "empty.txt"
            empty_path.write_text("   \n", encoding="utf-8")
            with self.assertRaises(ValueError):
                main.read_poly_coeffs(empty_path)

    def test_experiment_consistency(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tab_path = Path(tmp_dir) / "tab.txt"
            res = main.experiment(tab_path=tab_path)
            self.assertEqual(len(res["nodes"]), 41)
            self.assertEqual(len(res["root_runs"]), 2)
            self.assertEqual(res["root_runs"][0]["trend"], "increasing")
            self.assertEqual(res["root_runs"][1]["trend"], "decreasing")
            for run in res["root_runs"]:
                self.assertLess(-2.0, run["stability"])
                self.assertLess(run["stability"], 0.0)
                roots = [item[1].root for item in run["methods"]]
                for _, m_res in run["methods"]:
                    self.assertLess(m_res.residual, main.EPS)
                    self.assertAlmostEqual(m_res.root, roots[1], places=9)
            self.assertAlmostEqual(res["real_root"].root, 5.0, places=10)
            self.assertAlmostEqual(res["complex_root"].alpha, 1.0, places=9)
            self.assertAlmostEqual(res["complex_root"].beta, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
