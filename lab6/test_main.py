import math
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import main


class LuAndRefinementTests(unittest.TestCase):
    def test_lu_decompose_and_solve_exact_small(self):
        a = np.array([
            [2.0, 4.0, -2.0],
            [4.0, 9.0, -3.0],
            [-2.0, -3.0, 7.0],
        ])
        expected_l = np.array([
            [2.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [-2.0, 1.0, 4.0],
        ])
        expected_u = np.array([
            [1.0, 2.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        x_true = np.array([1.0, 2.0, -1.0])
        b = a @ x_true
        l_mat, u_mat = main.lu_decompose(a)
        np.testing.assert_allclose(l_mat, expected_l, atol=1e-14)
        np.testing.assert_allclose(u_mat, expected_u, atol=1e-14)
        np.testing.assert_allclose(l_mat @ u_mat, a, atol=1e-14)
        x = main.solve_lu(l_mat, u_mat, b)
        np.testing.assert_allclose(x, x_true, atol=1e-14)
        np.testing.assert_allclose(x, np.linalg.solve(a, b), atol=1e-14)

    def test_matvec_residual_and_vector_norm(self):
        a = np.array([[1.0, 1e16, -1e16], [1.0, -1.0, 0.0]])
        x = np.array([2.5, 1.0, 1.0])
        b = np.array([2.5, 1.5])
        np.testing.assert_allclose(main.matvec(a, x), b, atol=1e-15)
        res = main.residual_vector(a, x, b)
        np.testing.assert_allclose(res, [0.0, 0.0], atol=1e-15)
        self.assertEqual(main.vector_norm([0.5, -3.25, 1.0]), 3.25)

    def test_file_io_and_write_lu(self):
        a = np.array([[4.0, 2.0], [1.0, 3.0]])
        b = np.array([10.0, 10.0])
        l_mat, u_mat = main.lu_decompose(a)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savetxt(root / "A.txt", a, fmt="%.17g")
            np.savetxt(root / "B.txt", b, fmt="%.17g")
            main.write_lu(root / "LU.txt", l_mat, u_mat)

            np.testing.assert_array_equal(main.read_matrix(root / "A.txt"), a)
            np.testing.assert_array_equal(main.read_vector(root / "B.txt"), b)
            lu_text = (root / "LU.txt").read_text(encoding="utf-8")
            self.assertIn("# L", lu_text)
            self.assertIn("# U", lu_text)
            l_part, u_part = lu_text.split("# U")
            l_read = np.loadtxt(l_part.splitlines())
            u_read = np.loadtxt(u_part.splitlines())
            np.testing.assert_allclose(l_read, l_mat, atol=1e-15)
            np.testing.assert_allclose(u_read, u_mat, atol=1e-15)

            (root / "empty.txt").write_text("# comment only\n",
                                            encoding="utf-8")
            (root / "rect.txt").write_text("1 2 3\n4 5 6\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                main.read_matrix(root / "empty.txt")
            with self.assertRaises(ValueError):
                main.read_matrix(root / "rect.txt")
            with self.assertRaises(ValueError):
                main.read_vector(root / "A.txt")

    def test_invalid_singular_and_nonconvergent_guards(self):
        with self.assertRaises(ValueError):
            main.lu_decompose([[0.0, 1.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            main.lu_decompose([[1.0, 2.0], [2.0, 4.0]])
        with self.assertRaises(ValueError):
            main.lu_decompose([[1.0, math.nan], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            main.solve_lu(np.eye(2), [[2.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
        with self.assertRaises(ValueError):
            main.vector_norm([])

        a = np.array([[1.0, 1.0], [0.0, 1.0]])
        l_mat, u_mat = main.lu_decompose(a)
        b = np.array([2.0, 1.0])
        with self.assertRaises(ValueError):
            main.refine_solution(a, l_mat, u_mat, b, [0.0, 0.0], eps0=0.0)
        # перешкода з 1 ітерацією і великим збуренням перевіряє ArithmeticError
        a_bad = np.array([[1.0, 1.0], [1.0, 1.0001]])
        l_bad, u_bad = main.lu_decompose(a_bad)
        with self.assertRaises(ArithmeticError):
            main.refine_solution(
                a_bad, l_bad, u_bad, [2.0, 2.0001], [0.0, 0.0],
                eps0=1e-18, max_iter=1,
            )

    def test_refinement_converges_from_perturbed_guess(self):
        a = np.array([[5.0, 1.0, 0.5], [1.0, 4.0, 0.2], [0.5, 0.2, 3.0]])
        x_true = np.array([2.5, 2.5, 2.5])
        b = main.matvec(a, x_true)
        l_mat, u_mat = main.lu_decompose(a)
        x0 = x_true + np.array([1e-3, -2e-3, 1e-3])
        refined, iters, history = main.refine_solution(
            a, l_mat, u_mat, b, x0, eps0=1e-14, x_true=x_true
        )
        self.assertEqual(iters, 2)
        self.assertLessEqual(history[-1]["dx"], 1e-14)
        self.assertLessEqual(history[-1]["residual"], 1e-14)
        np.testing.assert_allclose(refined, x_true, atol=1e-14)

    def test_experiment_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = main.experiment(data_dir=root)
            for name in ("A.txt", "B.txt", "LU.txt"):
                self.assertTrue((root / name).is_file())
            a = main.read_matrix(root / "A.txt")
            b = main.read_vector(root / "B.txt")
            ref_x = np.linalg.solve(a, b)
            np.testing.assert_allclose(result["x_refined"], ref_x, atol=1e-13)

        self.assertEqual(result["n"], 100)
        self.assertGreaterEqual(result["iterations"], 1)
        self.assertLessEqual(result["final_dx"], main.EPS0)
        self.assertLessEqual(result["final_residual"], main.EPS0)
        self.assertLess(result["final_residual"], result["initial_residual"])
        self.assertLess(result["final_error"], result["initial_error"])
        figs = main.make_plots(result)
        self.assertEqual(len(figs), 2)
        for fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
