import math
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import main


def newton_system_reference():
    x = np.array([1.0, 1.7], dtype=float)
    for _ in range(30):
        f = main.system_equations(x)
        jac = np.array([
            [2.0 * x[0], 2.0 * x[1]],
            [-math.exp(x[0]), 1.0],
        ], dtype=float)
        step = np.linalg.solve(jac, -f)
        x = x + step
        if np.linalg.norm(step) < 1e-15:
            break
    return x


class HookeJeevesTests(unittest.TestCase):
    def test_quadratic_analytic_minima(self):
        bowl = lambda x: (x[0] - 3.0) ** 2 + 4.0 * (x[1] + 2.0) ** 2
        res = main.hooke_jeeves(bowl, [0.0, 0.0], [0.5, 0.5])
        self.assertTrue(res.converged)
        self.assertTrue(np.allclose(res.x, [3.0, -2.0], atol=1e-6))
        self.assertLess(res.value, 1e-12)
        self.assertGreaterEqual(res.pattern_moves, 1)

        tilted = lambda x: (x[0] - x[1]) ** 2 + 10.0 * (x[0] + x[1] - 2.0) ** 2
        res_t = main.hooke_jeeves(tilted, [0.0, 0.0], [0.2, 0.2])
        self.assertTrue(res_t.converged)
        self.assertTrue(np.allclose(res_t.x, [1.0, 1.0], atol=1e-6))

    def test_pdf_benchmark_functions(self):
        res_rosen = main.hooke_jeeves(
            main.rosenbrock, [-1.2, 0.0], [0.2, 0.2], q=2.0, p=2.0
        )
        self.assertTrue(res_rosen.converged)
        self.assertTrue(np.allclose(res_rosen.x, [1.0, 1.0], atol=1e-5))
        self.assertLess(res_rosen.value, 1e-10)

        power_fn = lambda x: (10.0 * (x[0] - x[1]) ** 2 + (x[0] - 1.0) ** 2) ** 4
        res_pow = main.hooke_jeeves(power_fn, [-1.2, 0.0], [0.2, 0.2])
        self.assertTrue(res_pow.converged)
        self.assertTrue(np.allclose(res_pow.x, [1.0, 1.0], atol=1e-6))

        miele_fn = lambda x: (
            (math.exp(x[0]) - x[1]) ** 4
            + 100.0 * (x[1] - x[2]) ** 6
            + math.tan(x[2] - x[3]) ** 4
            + x[0] ** 8
            + (x[3] - 1.0) ** 2
        )
        res_miele = main.hooke_jeeves(miele_fn, [1.0, 2.0, 2.0, 2.0], [0.2] * 4)
        self.assertTrue(res_miele.converged)
        self.assertTrue(np.allclose(res_miele.x, [0.0, 1.0, 1.0, 1.0], atol=1e-4))
        self.assertLess(res_miele.value, 1e-12)

    def test_system_against_newton_reference(self):
        ref = newton_system_reference()
        self.assertLess(np.linalg.norm(main.system_equations(ref)), 1e-14)
        res = main.hooke_jeeves(
            main.system_objective,
            main.X0_SYSTEM,
            main.DELTA0_SYSTEM,
            q=main.Q,
            p=main.P,
            eps1=main.EPS1,
            eps2=main.EPS2,
        )
        self.assertTrue(res.converged)
        self.assertTrue(np.allclose(res.x, ref, atol=1e-6))
        self.assertLess(res.value, 1e-12)
        self.assertLess(np.linalg.norm(main.system_equations(res.x)), 1e-6)

    def test_invalid_inputs_and_stalling_guards(self):
        fn = main.rosenbrock
        bad_calls = [
            ([], []),
            ([[1.0, 2.0]], [[0.1, 0.1]]),
            ([math.nan, 0.0], [0.1, 0.1]),
            ([0.0, 0.0], [0.1]),
            ([0.0, 0.0], [0.0, 0.1]),
            ([0.0, 0.0], [-0.2, 0.1]),
        ]
        for x0, d0 in bad_calls:
            with self.assertRaises(ValueError):
                main.hooke_jeeves(fn, x0, d0)

        for kwargs in [
            dict(q=1.0),
            dict(q=0.5),
            dict(p=0.0),
            dict(eps1=0.0),
            dict(eps2=-1e-5),
            dict(max_iter=0),
            dict(max_iter=True),
        ]:
            with self.assertRaises(ValueError):
                main.hooke_jeeves(fn, [0.0, 0.0], [0.2, 0.2], **kwargs)

        with self.assertRaises(ValueError):
            main.hooke_jeeves(lambda x: math.nan, [0.0, 0.0], [0.2, 0.2])

        # перевіряєм зупинку при вичерпанні max_iter та необмеженій функції
        truncated = main.hooke_jeeves(fn, [-1.2, 0.0], [0.2, 0.2], max_iter=2)
        self.assertFalse(truncated.converged)
        self.assertLessEqual(truncated.steps, 2)

        unbounded = main.hooke_jeeves(
            lambda x: -x[0] - x[1], [0.0, 0.0], [0.5, 0.5], max_iter=25
        )
        self.assertFalse(unbounded.converged)

        def overflow_on_far(x):
            if abs(x[0]) > 5.0:
                raise OverflowError("за межами діапазону")
            return x[0] ** 2 + x[1] ** 2

        res_ovf = main.hooke_jeeves(overflow_on_far, [1.0, 1.0], [2.0, 2.0])
        self.assertTrue(res_ovf.converged)
        self.assertTrue(np.allclose(res_ovf.x, [0.0, 0.0], atol=1e-6))

    def test_experiment_and_plots(self):
        result = main.experiment()
        sys_res = result["system"]
        ros_res = result["rosenbrock"]
        self.assertTrue(sys_res.converged)
        self.assertTrue(ros_res.converged)
        self.assertGreaterEqual(sys_res.pattern_moves, 1)
        self.assertEqual(sys_res.steps, len(sys_res.trajectory) - 1)
        self.assertTrue(np.allclose(sys_res.x, sys_res.trajectory[-1]))
        self.assertTrue(np.all(np.diff(sys_res.values) < 0))
        self.assertLess(sys_res.value, 1e-12)
        self.assertLess(ros_res.value, 1e-10)

        figs = main.make_plots(result)
        try:
            self.assertEqual(len(figs), 3)
            for fig in figs:
                self.assertGreaterEqual(len(fig.axes), 1)
        finally:
            for fig in figs:
                plt.close(fig)

    def test_save_trajectory_io(self):
        result = main.experiment()
        sys_res = result["system"]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "trajectory.txt"
            saved = main.save_trajectory(
                sys_res.trajectory, sys_res.values, out_file
            )
            self.assertEqual(saved, out_file)
            lines = out_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), len(sys_res.trajectory) + 1)
            for idx, line in enumerate(lines[1:]):
                parts = line.split()
                self.assertEqual(int(parts[0]), idx)
                self.assertAlmostEqual(
                    float(parts[1]), sys_res.trajectory[idx, 0], places=10
                )
                self.assertAlmostEqual(
                    float(parts[2]), sys_res.trajectory[idx, 1], places=10
                )
                self.assertTrue(
                    math.isclose(
                        float(parts[3]), sys_res.values[idx],
                        rel_tol=1e-12, abs_tol=1e-15,
                    )
                )


if __name__ == "__main__":
    unittest.main()
