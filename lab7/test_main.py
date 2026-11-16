from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import main


class IterativeMethodsTests(unittest.TestCase):
    def test_norms_and_matvec_against_numpy(self):
        a = np.array([[1.0, -2.0, 3.0], [-4.0, 0.5, 1.5]])
        x = np.array([2.0, -5.0, 1.0])
        np.testing.assert_allclose(main.matvec(a, x), a @ x, rtol=1e-15, atol=1e-15)
        self.assertAlmostEqual(
            main.vector_norm(x), np.linalg.norm(x, ord=np.inf)
        )
        self.assertAlmostEqual(
            main.matrix_norm(a), np.linalg.norm(a, ord=np.inf)
        )

    def test_solvers_on_analytic_small_system(self):
        a = np.array([
            [10.0, 1.0, -1.0],
            [1.0, 10.0, 1.0],
            [-1.0, 1.0, 10.0],
        ])
        x_true = np.array([1.0, -2.0, 3.0])
        b = a @ x_true
        x0 = np.zeros(3)
        ref = np.linalg.solve(a, b)

        for solver in (main.simple_iteration, main.jacobi, main.seidel):
            res = solver(a, b, x0, eps=1e-14)
            self.assertLess(res.c_norm, 1.0)
            self.assertLess(res.steps[-1], 1e-14)
            self.assertLessEqual(res.residuals[-1], 1e-14)
            self.assertEqual(len(res.residuals), res.iterations)
            self.assertEqual(len(res.steps), res.iterations)
            np.testing.assert_allclose(res.solution, ref, rtol=1e-13, atol=1e-13)

        # перевіряєм перший крок Якобі та Зейделя на ручному 2x2 прикладі
        a2 = np.array([[4.0, 1.0], [1.0, 3.0]])
        b2 = np.array([1.0, 2.0])
        j1 = main.jacobi(a2, b2, [0.0, 0.0], eps=10.0, max_iter=1)
        s1 = main.seidel(a2, b2, [0.0, 0.0], eps=10.0, max_iter=1)
        np.testing.assert_allclose(j1.solution, [0.25, 2.0 / 3.0])
        np.testing.assert_allclose(s1.solution, [0.25, 7.0 / 12.0])

    def test_file_io_in_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            a_file = tmp_dir / "A.txt"
            b_file = tmp_dir / "B.txt"
            a_src, b_src, _ = main.generate_system(n=8, x_val=2.5, seed=11)
            main.save_system(a_src, b_src, a_file, b_file)

            a_loaded = main.read_matrix(a_file)
            b_loaded = main.read_vector(b_file)
            np.testing.assert_allclose(a_loaded, a_src, rtol=1e-15, atol=1e-15)
            np.testing.assert_allclose(b_loaded, b_src, rtol=1e-15, atol=1e-15)

            empty_file = tmp_dir / "empty.txt"
            empty_file.write_text("", encoding="utf-8")
            ragged_file = tmp_dir / "ragged.txt"
            ragged_file.write_text("1 2\n3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                main.read_matrix(empty_file)
            with self.assertRaises(ValueError):
                main.read_vector(empty_file)
            with self.assertRaises(ValueError):
                main.read_matrix(ragged_file)

    def test_invalid_inputs_and_guards(self):
        a = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x0 = np.zeros(2)

        with self.assertRaises(ValueError):
            main.simple_iteration([[1.0, 2.0, 3.0]], [1.0], [0.0])
        with self.assertRaises(ValueError):
            main.jacobi(a, [1.0], x0)
        with self.assertRaises(ValueError):
            main.seidel(a, b, x0, eps=0.0)
        with self.assertRaises(ValueError):
            main.simple_iteration(a, b, x0, tau=2.0 / main.matrix_norm(a))
        with self.assertRaises(ValueError):
            main.simple_iteration(np.array([[1.0, 2.0], [2.0, 1.0]]), b, x0)

        zero_diag = np.array([[0.0, 1.0], [1.0, 3.0]])
        with self.assertRaises(ValueError):
            main.jacobi(zero_diag, b, x0)
        with self.assertRaises(ValueError):
            main.seidel(zero_diag, b, x0)

        divergent = np.array([[1.0, 3.0], [3.0, 1.0]])
        with self.assertRaises(ArithmeticError):
            main.jacobi(divergent, b, x0, eps=1e-12, max_iter=20)
        with self.assertRaises(ArithmeticError):
            main.seidel(divergent, b, x0, eps=1e-12, max_iter=20)

    def test_experiment_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            a_file = tmp_dir / "A.txt"
            b_file = tmp_dir / "B.txt"
            result = main.experiment(a_path=a_file, b_path=b_file)

            self.assertEqual(result["n"], 100)
            self.assertGreater(result["diag_margin"], 0.0)
            self.assertGreater(result["tau"], 0.0)
            self.assertLess(result["tau"], 2.0 / result["norm_a"])

            a_loaded = main.read_matrix(a_file)
            b_loaded = main.read_vector(b_file)
            ref = np.linalg.solve(a_loaded, b_loaded)

            iters = {}
            for name, _, res in result["methods"]:
                iters[name] = res.iterations
                self.assertLess(res.c_norm, 1.0)
                self.assertLess(res.steps[-1], result["eps"])
                self.assertLessEqual(res.residuals[-1], result["eps"])
                self.assertLess(result["errors"][name], 1e-13)
                np.testing.assert_allclose(res.solution, ref, rtol=1e-13, atol=1e-13)

            self.assertLess(iters["Seidel"], iters["Jacobi"])
            self.assertLess(iters["Jacobi"], iters["Simple iteration"])

            figs = main.make_plots(result)
            self.assertEqual(len(figs), 2)
            for fig in figs:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
