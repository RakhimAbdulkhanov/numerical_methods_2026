from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
A_PATH = BASE_DIR / "A.txt"
B_PATH = BASE_DIR / "B.txt"
N = 100
X_TRUE = 2.5
X0_VAL = 1.0
EPS0 = 1e-14
SEED = 7


@dataclass
class IterResult:
    solution: np.ndarray
    iterations: int
    residuals: list[float]
    steps: list[float]
    c_norm: float
    tau: float | None = None


def generate_system(n=N, x_val=X_TRUE, seed=SEED):
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("розмірність матриці має бути додатним цілим числом")
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.03, 0.03, size=(n, n))
    np.fill_diagonal(a, 0.0)
    off_diag = np.abs(a).sum(axis=1)
    # підсилюєм діагональ шоб норма була менша й точність 1e-14 досягалась без проблем
    np.fill_diagonal(a, off_diag + rng.uniform(1.5, 2.5, size=n))
    x_true = np.full(n, x_val, dtype=float)
    b = matvec(a, x_true)
    return a, b, x_true


def save_system(a, b, a_path=A_PATH, b_path=B_PATH):
    np.savetxt(a_path, a, fmt="%.17g")
    np.savetxt(b_path, b, fmt="%.17g")


def read_matrix(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            rows.append([float(val) for val in parts])
    if not rows:
        raise ValueError("файл матриці порожній")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("рядки матриці мають різну довжину")
    mat = np.array(rows, dtype=float)
    if not np.all(np.isfinite(mat)):
        raise ValueError("матриця містить нескінченні або NaN значення")
    return mat


def read_vector(path):
    values = [
        float(token)
        for token in Path(path).read_text(encoding="utf-8").split()
    ]
    if not values:
        raise ValueError("файл вектора порожній")
    vec = np.array(values, dtype=float)
    if not np.all(np.isfinite(vec)):
        raise ValueError("вектор містить нескінченні або NaN значення")
    return vec


def matvec(a, x):
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)
    if a.ndim != 2 or x.ndim != 1 or a.shape[1] != x.shape[0]:
        raise ValueError("несумісні розмірності матриці та вектора")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("матриця і вектор не можуть бути порожніми")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(x))):
        raise ValueError("матриця або вектор містять нескінченні значення")
    return np.array([math.fsum(row * x) for row in a], dtype=float)


def vector_norm(v):
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("вектор має бути непорожнім одновимірним масивом")
    if not np.all(np.isfinite(v)):
        raise ValueError("вектор містить нескінченні або NaN значення")
    return float(np.max(np.abs(v)))


def matrix_norm(a):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("матриця має бути непорожнім двовимірним масивом")
    if not np.all(np.isfinite(a)):
        raise ValueError("матриця містить нескінченні або NaN значення")
    return max(math.fsum(np.abs(row)) for row in a)


def _validate_system(a, b, x0, eps, max_iter):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
        raise ValueError("матриця має бути непорожньою квадратною")
    n = a.shape[0]
    if b.shape != (n,) or x0.shape != (n,):
        raise ValueError("розмірність векторів має збігатися з матрицею")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(x0))):
        raise ValueError("вхідні дані містять нескінченні або NaN значення")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("точність eps має бути додатною і скінченною")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("max_iter має бути додатним цілим числом")
    return a, b, x0.copy()


def _iterate(a, b, x, eps, max_iter, step_fn, c_norm, tau=None):
    residuals = []
    steps = []
    for k in range(1, max_iter + 1):
        x_next = step_fn(x)
        if not np.all(np.isfinite(x_next)):
            raise ArithmeticError("ітераційний процес розійшовся")
        step_norm = vector_norm(x_next - x)
        residual_norm = vector_norm(matvec(a, x_next) - b)
        steps.append(step_norm)
        residuals.append(residual_norm)
        x = x_next
        if step_norm < eps and residual_norm <= eps:
            return IterResult(x, k, residuals, steps, c_norm, tau)
    raise ArithmeticError("метод не зійшовся за задану кількість ітерацій")


def simple_iteration(a, b, x0, eps=1e-14, tau=None, max_iter=5000):
    a, b, x = _validate_system(a, b, x0, eps, max_iter)
    norm_a = matrix_norm(a)
    if norm_a == 0:
        raise ValueError("норма нульової матриці не дозволяє обрати tau")
    if tau is None:
        tau = 1.5 / norm_a
    if not math.isfinite(tau) or tau <= 0 or tau >= 2.0 / norm_a:
        raise ValueError("параметр tau має бути в межах 0 < tau < 2 / ||A||")
    c_mat = np.eye(a.shape[0]) - tau * a
    c_norm = matrix_norm(c_mat)
    if c_norm >= 1.0:
        raise ValueError("не виконується умова стискання ||E - tau*A|| < 1")
    d_vec = tau * b
    return _iterate(
        a, b, x, eps, max_iter,
        lambda cur: matvec(c_mat, cur) + d_vec,
        c_norm, tau=float(tau),
    )


def jacobi(a, b, x0, eps=1e-14, max_iter=5000):
    a, b, x = _validate_system(a, b, x0, eps, max_iter)
    diag = np.diag(a)
    if np.any(diag == 0.0):
        raise ValueError("нульовий діагональний елемент у методі Якобі")
    c_mat = -((a - np.diag(diag)) / diag[:, None])
    c_norm = matrix_norm(c_mat)
    d_vec = b / diag
    return _iterate(
        a, b, x, eps, max_iter,
        lambda cur: matvec(c_mat, cur) + d_vec,
        c_norm,
    )


def seidel(a, b, x0, eps=1e-14, max_iter=5000):
    a, b, x = _validate_system(a, b, x0, eps, max_iter)
    diag = np.diag(a)
    if np.any(diag == 0.0):
        raise ValueError("нульовий діагональний елемент у методі Зейделя")
    n = a.shape[0]
    c_mat = -np.linalg.solve(np.tril(a), np.triu(a, k=1))
    c_norm = matrix_norm(c_mat)

    def step(cur):
        nxt = cur.copy()
        for i in range(n):
            # беремо вже оновлені координати зліва і старі справа
            left = math.fsum(a[i, :i] * nxt[:i]) if i > 0 else 0.0
            right = math.fsum(a[i, i + 1:] * cur[i + 1:]) if i + 1 < n else 0.0
            nxt[i] = (b[i] - left - right) / diag[i]
        return nxt

    return _iterate(a, b, x, eps, max_iter, step, c_norm)


def experiment(a_path=A_PATH, b_path=B_PATH):
    a_gen, b_gen, x_true = generate_system(N, X_TRUE, SEED)
    save_system(a_gen, b_gen, a_path, b_path)
    a = read_matrix(a_path)
    b = read_vector(b_path)
    n = a.shape[0]
    x0 = np.full(n, X0_VAL, dtype=float)
    norm_a = matrix_norm(a)
    diag = np.abs(np.diag(a))
    off_diag = np.sum(np.abs(a), axis=1) - diag
    diag_margin = float(np.min(diag - off_diag))

    si_res = simple_iteration(a, b, x0, eps=EPS0)
    jac_res = jacobi(a, b, x0, eps=EPS0)
    sei_res = seidel(a, b, x0, eps=EPS0)

    methods = [
        ("Simple iteration", "Проста ітерація", si_res),
        ("Jacobi", "Метод Якобі", jac_res),
        ("Seidel", "Метод Зейделя", sei_res),
    ]
    errors = {
        name: vector_norm(res.solution - x_true)
        for name, _, res in methods
    }
    return dict(
        n=n, eps=EPS0, x_true=x_true, x0=x0,
        norm_a=norm_a, diag_margin=diag_margin,
        tau=si_res.tau, methods=methods, errors=errors,
    )


def print_results(result):
    print(f"n = {result['n']}; eps0 = {result['eps']:.0e}; "
          f"x_true = {X_TRUE:g}; x0 = {X0_VAL:g}")
    print(f"||A||_inf = {result['norm_a']:.6f}; "
          f"min_diag_margin = {result['diag_margin']:.6f}; "
          f"tau = {result['tau']:.6f}")
    print(f"{'method':<17} {'iters':>5} {'||C||_inf':>10} "
          f"{'||X(k)-X(k-1)||':>16} {'||AX(k)-B||':>13} {'max_err':>12}")
    for name, _, res in result["methods"]:
        print(f"{name:<17} {res.iterations:5d} {res.c_norm:10.6f} "
              f"{res.steps[-1]:16.6e} {res.residuals[-1]:13.6e} "
              f"{result['errors'][name]:12.6e}")


def make_plots(result):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    fig1, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    for (_, label, res), color, marker in zip(result["methods"], colors, markers):
        iters = range(1, res.iterations + 1)
        axes[0].semilogy(iters, res.residuals, marker=marker, markersize=4,
                         linewidth=1.5, color=color, label=label)
        axes[1].semilogy(iters, res.steps, marker=marker, markersize=4,
                         linewidth=1.5, color=color, label=label)

    for ax in axes:
        ax.axhline(result["eps"], color="red", linestyle="--", linewidth=1.1,
                   label=f"eps0 = {result['eps']:.0e}")
        ax.set_xlabel("Ітерація k")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8.5)

    axes[0].set_ylabel(r"$\|A X^{(k)} - B\|_\infty$")
    axes[0].set_title("Збіжність нев'язки за ітераціями")
    axes[1].set_ylabel(r"$\|X^{(k)} - X^{(k-1)}\|_\infty$")
    axes[1].set_title("Норма різниці сусідніх наближень")

    fig2, ax2 = plt.subplots(figsize=(7.5, 4.0), layout="constrained")
    labels = [label for _, label, _ in result["methods"]]
    counts = [res.iterations for _, _, res in result["methods"]]
    bars = ax2.bar(labels, counts, color=colors, width=0.5)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, count + 0.4, str(count),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(counts) * 1.18)
    ax2.set_ylabel("Кількість ітерацій")
    ax2.set_title(f"Кількість ітерацій до точності eps0 = {result['eps']:.0e}")
    ax2.grid(True, axis="y", alpha=0.3)
    return [fig1, fig2]


def main():
    result = experiment()
    print_results(result)
    figs = make_plots(result)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
