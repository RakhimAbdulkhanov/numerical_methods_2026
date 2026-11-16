import math
import unittest

import numpy as np

import main


class DifferentialEquationsTests(unittest.TestCase):
    def test_rk4_exact_on_quartic(self):
        x, y = main.rk4_solve(lambda t, u: 4.0 * t**3, 0.0, 0.0, 2.0, 0.25)
        self.assertAlmostEqual(x[-1], 2.0)
        for xi, yi in zip(x, y):
            self.assertAlmostEqual(yi, xi**4, places=12)

    def test_adams2_exact_on_quadratic(self):
        res = main.adams2_pc(lambda t, u: 2.0 * t, 0.0, 0.0, 2.0, 0.2)
        self.assertAlmostEqual(res.x[-1], 2.0)
        for xi, yi in zip(res.x, res.y):
            self.assertAlmostEqual(yi, xi**2, places=12)
        self.assertLess(float(np.max(np.abs(res.est_err))), 1e-13)

    def test_adams2_convergence_and_estimator(self):
        f = lambda t, u: -u
        r_coarse = main.adams2_pc(f, 0.0, 1.0, 1.0, 0.05)
        r_fine = main.adams2_pc(f, 0.0, 1.0, 1.0, 0.025)
        err_coarse = abs(r_coarse.y[-1] - math.exp(-1.0))
        err_fine = abs(r_fine.y[-1] - math.exp(-1.0))
        ratio = err_coarse / err_fine
        self.assertGreater(ratio, 3.8)
        self.assertLess(ratio, 4.3)
        # локальна оцінка адамса спадає як h^3
        est_ratio = abs(r_coarse.est_err[0]) / abs(r_fine.est_err[0])
        self.assertGreater(est_ratio, 7.5)
        self.assertLess(est_ratio, 8.6)
        # звіряєм оцінку -(y_cor - y_pr)/6 з похибкою першого кроку адамса
        x0, x1, x2 = r_fine.x[0], r_fine.x[1], r_fine.x[2]
        y1_ex = math.exp(-x1)
        _, y2_one, est_one, _ = main._adams2_step(
            f, x2, y1_ex, f(x1, y1_ex), f(x0, 1.0), 0.025, 1e-12, 20
        )
        true_local = math.exp(-x2) - y2_one
        self.assertAlmostEqual(est_one, true_local, places=7)

    def test_rk4_runge_local_error_matches_exact_step(self):
        h = 0.025
        x, y = main.rk4_solve(main.ode_rhs, 0.0, 1.0, 1.0, h)
        phis = main.rk4_runge_error(main.ode_rhs, x, y, h)
        for i in range(len(phis)):
            xn, yn = float(x[i]), float(y[i])
            y_step = main.rk4_step(main.ode_rhs, xn, yn, h)
            # одно-крокова похибка при точному старті збігається з оцінкою рунге
            exact_from_node = yn * math.exp(
                math.sin(2.0 * (xn + h)) - math.sin(2.0 * xn)
            )
            true_local = exact_from_node - y_step
            self.assertAlmostEqual(phis[i], true_local, delta=1e-11)

    def test_adaptive_solvers_behavior(self):
        adams_a = main.adams2_adaptive(
            main.ode_rhs, main.A, main.Y0, main.B, main.ADAMS_H0, main.ADAMS_EPS
        )
        self.assertAlmostEqual(adams_a.x[0], main.A)
        self.assertAlmostEqual(adams_a.x[-1], main.B)
        self.assertTrue(np.all(np.diff(adams_a.x) > 0))
        self.assertGreater(len(set(np.round(adams_a.h[:-1], 6))), 1)
        self.assertLessEqual(float(np.max(adams_a.est_err)), main.ADAMS_EPS)

        rk4_a = main.rk4_adaptive(
            main.ode_rhs, main.A, main.Y0, main.B, main.RK4_H0, main.RK4_EPS
        )
        self.assertAlmostEqual(rk4_a.x[0], main.A)
        self.assertAlmostEqual(rk4_a.x[-1], main.B)
        self.assertTrue(np.all(np.diff(rk4_a.x) > 0))
        self.assertGreater(len(set(np.round(rk4_a.h[:-1], 6))), 1)
        phi_a, _ = main.exact_error(rk4_a.x, rk4_a.y)
        self.assertLess(float(np.max(np.abs(phi_a))), main.RK4_EPS)

    def test_invalid_inputs_and_guards(self):
        for bad_h in (0.0, -0.1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                main.rk4_step(main.ode_rhs, 0.0, 1.0, bad_h)
            with self.assertRaises(ValueError):
                main.adams2_pc(main.ode_rhs, 0.0, 1.0, 1.0, bad_h)
        with self.assertRaises(ValueError):
            main.rk4_solve(main.ode_rhs, 2.0, 1.0, 1.0, 0.1)
        with self.assertRaises(ValueError):
            main.rk4_runge_error(main.ode_rhs, [0.0], [1.0], 0.1)
        with self.assertRaises(ValueError):
            main.adams2_adaptive(main.ode_rhs, 0.0, 1.0, 1.0, 0.1, 1e-4, 0.5, 0.1)
        with self.assertRaises(ArithmeticError):
            main.adams2_pc(lambda t, u: 1000.0 * u, 0.0, 1.0, 2.0, 0.5, max_corr=5)
        with self.assertRaises(ArithmeticError):
            main.rk4_adaptive(main.ode_rhs, 0.0, 1.0, 1.0, 0.05, 1e-20, h_min=0.01)
        with self.assertRaises(ArithmeticError):
            main.adams2_adaptive(main.ode_rhs, 0.0, 1.0, 1.0, 0.05, 1e-20, h_min=0.01)

    def test_experiment_consistency(self):
        res = main.experiment()
        self.assertEqual(len(res["adams_fixed"].x) - 1, 600)
        self.assertTrue(res["adams_optimal"])
        self.assertLess(len(res["adams_adapt"].h), 600)
        self.assertFalse(res["rk4_optimal"])
        self.assertGreater(res["rk4_h_req"], main.H_FIXED)
        self.assertLess(len(res["rk4_adapt"].h), 600)
        self.assertLess(float(np.max(np.abs(res["rk4_adapt_phi"]))), main.RK4_EPS)
        last_order = res["rk4_sweep"][-1][3]
        self.assertGreater(last_order, 3.95)
        self.assertLess(last_order, 4.1)


if __name__ == "__main__":
    unittest.main()
