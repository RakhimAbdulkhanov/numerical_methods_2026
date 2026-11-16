import io
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
N = 100
TRUE_VALUE = 2.5
SEED = 42
EPS0 = 1e-14
PIVOT_TOL = 1e-15


def _read_text_lines(path):
    text = Path(path).read_text(encoding="utf-8")
    lines = [
        line for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("файл не містить числових даних")
    return "\n".join(lines)


def _as_square(a, name="матриця"):
    arr = np.asarray(a, dtype=float)
    if (arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[0] != arr.shape[1]
            or not np.all(np.isfinite(arr))):
        raise ValueError(f"{name}: потрібна квадратна скінченна матриця")
    return arr


def _as_vector(v, name="вектор"):
    arr = np.asarray(v, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: потрібен непорожній 1D скінченний вектор")
    return arr


def read_matrix(path):
    raw = _read_text_lines(path)
    return _as_square(np.loadtxt(io.StringIO(raw), dtype=float, ndmin=2))


def read_vector(path):
    raw = _read_text_lines(path)
    return _as_vector(np.loadtxt(io.StringIO(raw), dtype=float, ndmin=1))


def write_lu(path, l_mat, u_mat):
    l_arr = _as_square(l_mat, "L")
    u_arr = _as_square(u_mat, "U")
    if l_arr.shape != u_arr.shape:
        raise ValueError("матриці L та U мають бути однакового розміру")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# L\n")
        np.savetxt(f, l_arr, fmt="%.17g")
        f.write("\n# U\n")
        np.savetxt(f, u_arr, fmt="%.17g")


def matvec(a, x):
    a_arr = np.asarray(a, dtype=float)
    x_arr = _as_vector(x, "x")
    if a_arr.ndim != 2 or a_arr.shape[1] != x_arr.shape[0]:
        raise ValueError("неузгоджені розміри матриці та вектора")
    if not np.all(np.isfinite(a_arr)):
        raise ValueError("матриця містить нескінченні або NaN значення")
    return np.array([math.fsum(row * x_arr) for row in a_arr], dtype=float)


def residual_vector(a, x, b):
    a_arr = np.asarray(a, dtype=float)
    x_arr = _as_vector(x, "x")
    b_arr = _as_vector(b, "B")
    if (a_arr.ndim != 2 or a_arr.shape[1] != x_arr.shape[0]
            or a_arr.shape[0] != b_arr.shape[0]):
        raise ValueError("неузгоджені розміри для обчислення нев'язки")
    if not np.all(np.isfinite(a_arr)):
        raise ValueError("матриця містить нескінченні або NaN значення")
    # додаєм b_i в тому ж fsum, шоб уникнути втрати знаків при відніманні
    return np.array(
        [math.fsum([b_arr[i], *(-(a_arr[i] * x_arr))])
         for i in range(a_arr.shape[0])],
        dtype=float,
    )


def vector_norm(v):
    return float(np.max(np.abs(_as_vector(v))))


def generate_system(n=N, value=TRUE_VALUE, seed=SEED):
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("розмірність n має бути додатним цілим числом")
    if not math.isfinite(value):
        raise ValueError("еталонний розв'язок має бути скінченним")
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.1, 0.1, size=(n, n))
    x_true = np.full(n, value, dtype=float)
    b = matvec(a, x_true)
    return a, b, x_true


def lu_decompose(a, pivot_tol=PIVOT_TOL):
    a_arr = _as_square(a)
    n = a_arr.shape[0]
    l_mat = np.zeros((n, n), dtype=float)
    u_mat = np.eye(n, dtype=float)
    for k in range(n):
        # спочатку k-й стовпець L, потім k-й рядок U
        l_mat[k:, k] = a_arr[k:, k] - l_mat[k:, :k] @ u_mat[:k, k]
        if abs(l_mat[k, k]) < pivot_tol:
            raise ValueError(f"вироджений ведучий елемент на кроці {k}")
        if k + 1 < n:
            row = a_arr[k, k + 1:] - l_mat[k, :k] @ u_mat[:k, k + 1:]
            u_mat[k, k + 1:] = row / l_mat[k, k]
    return l_mat, u_mat


def solve_lu(l_mat, u_mat, b):
    l_arr = _as_square(l_mat, "L")
    u_arr = _as_square(u_mat, "U")
    b_arr = _as_vector(b, "B")
    n = l_arr.shape[0]
    if l_arr.shape != u_arr.shape or b_arr.shape[0] != n:
        raise ValueError("розміри L, U і B мають збігатися")
    if not np.allclose(np.diag(u_arr), 1.0, atol=1e-12):
        raise ValueError("діагональ матриці U має дорівнювати 1")

    # прямий хід для L Z = B
    z = np.zeros(n, dtype=float)
    for k in range(n):
        if abs(l_arr[k, k]) < PIVOT_TOL:
            raise ValueError(f"вироджена діагональ L у рядку {k}")
        z[k] = (b_arr[k] - l_arr[k, :k] @ z[:k]) / l_arr[k, k]

    # зворотний хід для U X = Z
    x = np.zeros(n, dtype=float)
    for k in range(n - 1, -1, -1):
        x[k] = z[k] - u_arr[k, k + 1:] @ x[k + 1:]
    return x


def refine_solution(a, l_mat, u_mat, b, x0, eps0=EPS0, max_iter=50,
                    x_true=None):
    if not math.isfinite(eps0) or eps0 <= 0:
        raise ValueError("точність eps0 має бути додатною і скінченною")
    if (not isinstance(max_iter, int) or isinstance(max_iter, bool)
            or max_iter <= 0):
        raise ValueError("max_iter має бути додатним цілим числом")
    x = _as_vector(x0, "x0").copy()
    history = [{
        "iter": 0,
        "dx": None,
        "residual": vector_norm(residual_vector(a, x, b)),
        "error": vector_norm(x - x_true) if x_true is not None else None,
    }]
    for it in range(1, max_iter + 1):
        r = residual_vector(a, x, b)
        dx = solve_lu(l_mat, u_mat, r)
        x = x + dx
        dx_norm = vector_norm(dx)
        res_norm = vector_norm(residual_vector(a, x, b))
        err_norm = vector_norm(x - x_true) if x_true is not None else None
        history.append({
            "iter": it,
            "dx": dx_norm,
            "residual": res_norm,
            "error": err_norm,
        })
        if dx_norm <= eps0 and res_norm <= eps0:
            return x, it, history
    raise ArithmeticError("метод уточнення не досяг заданої точності")


def experiment(data_dir=HERE):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    a_gen, b_gen, x_true = generate_system()
    a_path = data_dir / "A.txt"
    b_path = data_dir / "B.txt"
    lu_path = data_dir / "LU.txt"
    np.savetxt(a_path, a_gen, fmt="%.17g")
    np.savetxt(b_path, b_gen, fmt="%.17g")

    a = read_matrix(a_path)
    b = read_vector(b_path)
    l_mat, u_mat = lu_decompose(a)
    write_lu(lu_path, l_mat, u_mat)

    x0 = solve_lu(l_mat, u_mat, b)
    initial_res = vector_norm(residual_vector(a, x0, b))
    initial_err = vector_norm(x0 - x_true)
    refined, iterations, history = refine_solution(
        a, l_mat, u_mat, b, x0, eps0=EPS0, x_true=x_true
    )
    return {
        "n": N,
        "seed": SEED,
        "eps0": EPS0,
        "cond": float(np.linalg.cond(a)),
        "min_pivot": float(np.min(np.abs(np.diag(l_mat)))),
        "lu_residual": float(np.max(np.abs(l_mat @ u_mat - a))),
        "x_true": x_true,
        "x0": x0,
        "x_refined": refined,
        "initial_residual": initial_res,
        "initial_error": initial_err,
        "final_residual": history[-1]["residual"],
        "final_error": history[-1]["error"],
        "final_dx": history[-1]["dx"],
        "iterations": iterations,
        "history": history,
    }


def print_results(result):
    print(f"n = {result['n']}; seed = {result['seed']}; "
          f"eps0 = {result['eps0']:.0e}")
    print(f"cond(A) = {result['cond']:.3f}; "
          f"min |l_kk| = {result['min_pivot']:.6f}")
    print(f"max |LU - A| = {result['lu_residual']:.6e}")
    print(f"initial ||AX - B||_inf = {result['initial_residual']:.6e}")
    print(f"initial ||X - X_true||_inf = {result['initial_error']:.6e}")
    print(f"{'iter':>4} {'||dX||_inf':>15} {'||AX - B||_inf':>16} "
          f"{'||X - X_true||_inf':>20}")
    for row in result["history"]:
        dx_str = "-" if row["dx"] is None else f"{row['dx']:.6e}"
        err_str = "-" if row["error"] is None else f"{row['error']:.6e}"
        print(f"{row['iter']:4d} {dx_str:>15} "
              f"{row['residual']:16.6e} {err_str:>20}")
    print(f"iterations = {result['iterations']}")
    print(f"final ||AX - B||_inf = {result['final_residual']:.6e}")
    print(f"final ||X - X_true||_inf = {result['final_error']:.6e}")


def make_plots(result):
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    iters = [row["iter"] for row in result["history"]]
    residuals = [row["residual"] for row in result["history"]]
    errors = [row["error"] for row in result["history"]]
    dx_iters = [row["iter"] for row in result["history"]
                if row["dx"] is not None]
    dx_vals = [row["dx"] for row in result["history"]
               if row["dx"] is not None]
    ax.semilogy(iters, residuals, "o-", linewidth=2, markersize=7,
                label="||AX - B||_inf")
    ax.semilogy(iters, errors, "s--", linewidth=1.8, markersize=6,
                label="||X - X_true||_inf")
    ax.semilogy(dx_iters, dx_vals, "^:", linewidth=1.8, markersize=7,
                label="||dX||_inf")
    ax.axhline(result["eps0"], color="red", linestyle="--", linewidth=1.2,
               label=f"поріг eps0 = {result['eps0']:.0e}")
    ax.set_xticks(iters)
    ax.set(xlabel="Номер ітерації k", ylabel="Норма (log шкала)",
           title="Збіжність ітераційного уточнення розв'язку СЛАР")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig2, ax2 = plt.subplots(figsize=(9, 4.5), layout="constrained")
    idx = np.arange(1, result["n"] + 1)
    err_init = np.abs(result["x0"] - result["x_true"])
    err_final = np.abs(result["x_refined"] - result["x_true"])
    # точні 0 виносим на 1e-16 нижче 1 ULP, шоб не зникали на log-шкалі
    floor = 1e-16
    ulp = float(np.spacing(TRUE_VALUE))
    ax2.semilogy(idx, np.maximum(err_init, floor), "o-", markersize=3.2,
                 linewidth=0.9, alpha=0.85, label="X^(0) (прямий LU-розклад)")
    ax2.semilogy(idx, np.maximum(err_final, floor), "s-", markersize=3.2,
                 linewidth=0.9, alpha=0.9,
                 label=f"X^({result['iterations']}) (після уточнення)")
    ax2.axhline(ulp, color="gray", linestyle=":", linewidth=1.0,
                label=f"1 ULP біля {TRUE_VALUE} ({ulp:.2e})")
    ax2.axhline(floor, color="black", linestyle="--", linewidth=0.9, alpha=0.6,
                label="точний збіг (0 -> 1e-16)")
    ax2.set(xlabel="Індекс компоненти i", ylabel="|x_i - 2.5| (log шкала)",
            title="Покомпонентна похибка розв'язку до та після уточнення")
    ax2.set_ylim(5e-17, 4e-11)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8.5, loc="upper center", ncol=2)
    return [fig, fig2]


def main():
    result = experiment()
    print_results(result)
    make_plots(result)
    plt.show()


if __name__ == "__main__":
    main()
