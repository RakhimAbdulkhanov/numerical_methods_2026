from dataclasses import dataclass
from pathlib import Path
import cmath
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
TAB_FILE = BASE_DIR / "tabulation.txt"
POLY_FILE = BASE_DIR / "poly_coeffs.txt"
EPS = 1e-10
A, B, STEP = 0.0, 4.0, 0.1


@dataclass
class RootResult:
    root: float | complex
    iterations: int
    residual: float
    history: list[float]


@dataclass
class LinResult:
    alpha: float
    beta: float
    iterations: int
    residual: float
    history: list[tuple[float, float]]


def f_trans(x):
    return 2.0 * math.sin(x) - 0.3 * x - 0.5


def df_trans(x):
    return 2.0 * math.cos(x) - 0.3


def d2f_trans(x):
    return -2.0 * math.sin(x)


def tabulate(func, a, b, h):
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(h)):
        raise ValueError("межі та крок мають бути скінченними")
    if h <= 0 or b <= a:
        raise ValueError("потрібно b > a та h > 0")
    count = int(round((b - a) / h))
    nodes = []
    for i in range(count + 1):
        x = round(a + i * h, 10)
        if x <= b + 1e-12:
            nodes.append((x, float(func(x))))
    return nodes


def save_tabulation(nodes, path=TAB_FILE):
    lines = [f"{x:8.4f} {y:16.10f}\n" for x, y in nodes]
    Path(path).write_text("".join(lines), encoding="utf-8")


def read_tabulation(path=TAB_FILE):
    nodes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            nodes.append((float(parts[0]), float(parts[1])))
    if len(nodes) < 2:
        raise ValueError("у файлі табуляції замало вузлів")
    return nodes


def find_brackets(nodes):
    brackets = []
    for (x1, y1), (x2, y2) in zip(nodes, nodes[1:]):
        if y1 == 0.0:
            trend = "increasing" if y2 > y1 else "decreasing"
            brackets.append((x1, x1, x1, trend))
        elif y1 * y2 < 0.0:
            trend = "increasing" if y2 > y1 else "decreasing"
            brackets.append((x1, x2, 0.5 * (x1 + x2), trend))
    if len(nodes) >= 2 and nodes[-1][1] == 0.0:
        _, y_prev = nodes[-2]
        x_last, y_last = nodes[-1]
        trend = "increasing" if y_last > y_prev else "decreasing"
        brackets.append((x_last, x_last, x_last, trend))
    return brackets


def _check_solver_args(eps, max_iter):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("точність eps має бути додатною")
    if (not isinstance(max_iter, int)
            or isinstance(max_iter, bool) or max_iter < 1):
        raise ValueError("max_iter має бути додатним цілим")


def simple_iteration(f, x0, tau, eps=EPS, max_iter=500):
    _check_solver_args(eps, max_iter)
    if not math.isfinite(tau) or tau == 0:
        raise ValueError("параметр tau має бути ненульовим")
    x = float(x0)
    fx = float(f(x))
    history = [abs(fx)]
    for step in range(1, max_iter + 1):
        # для збіжності потрібно -2 < tau * F'(x) < 0
        x_next = x + tau * fx
        if not math.isfinite(x_next):
            raise ArithmeticError("проста ітерація розбіглась")
        fx_next = float(f(x_next))
        history.append(abs(fx_next))
        if abs(fx_next) < eps and abs(x_next - x) < eps:
            return RootResult(x_next, step, abs(fx_next), history)
        x, fx = x_next, fx_next
    raise ArithmeticError("проста ітерація не збіглась")


def newton(f, df, x0, eps=EPS, max_iter=100):
    _check_solver_args(eps, max_iter)
    x = float(x0)
    fx = float(f(x))
    history = [abs(fx)]
    for step in range(1, max_iter + 1):
        dfx = float(df(x))
        if dfx == 0:
            raise ZeroDivisionError("нульова похідна в методі Ньютона")
        x_next = x - fx / dfx
        if not math.isfinite(x_next):
            raise ArithmeticError("метод Ньютона розбігся")
        fx_next = float(f(x_next))
        history.append(abs(fx_next))
        if abs(fx_next) < eps and abs(x_next - x) < eps:
            return RootResult(x_next, step, abs(fx_next), history)
        x, fx = x_next, fx_next
    raise ArithmeticError("метод Ньютона не збігся")


def chebyshev(f, df, d2f, x0, eps=EPS, max_iter=100):
    _check_solver_args(eps, max_iter)
    x = float(x0)
    fx = float(f(x))
    history = [abs(fx)]
    for step in range(1, max_iter + 1):
        dfx = float(df(x))
        if dfx == 0:
            raise ZeroDivisionError("нульова похідна в методі Чебишева")
        d2fx = float(d2f(x))
        x_next = x - fx / dfx - 0.5 * (fx * fx) * d2fx / (dfx ** 3)
        if not math.isfinite(x_next):
            raise ArithmeticError("метод Чебишева розбігся")
        fx_next = float(f(x_next))
        history.append(abs(fx_next))
        if abs(fx_next) < eps and abs(x_next - x) < eps:
            return RootResult(x_next, step, abs(fx_next), history)
        x, fx = x_next, fx_next
    raise ArithmeticError("метод Чебишева не збігся")


def secant(f, x0, x1, eps=EPS, max_iter=200):
    _check_solver_args(eps, max_iter)
    if x0 == x1:
        raise ValueError("початкові наближення мають бути різними")
    x_prev, x_curr = float(x0), float(x1)
    f_prev, f_curr = float(f(x_prev)), float(f(x_curr))
    history = [abs(f_curr)]
    for step in range(1, max_iter + 1):
        denom = f_curr - f_prev
        if denom == 0:
            raise ZeroDivisionError("рівні значення функції в методі хорд")
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        if not math.isfinite(x_next):
            raise ArithmeticError("метод хорд розбігся")
        f_next = float(f(x_next))
        history.append(abs(f_next))
        if abs(f_next) < eps and abs(x_next - x_curr) < eps:
            return RootResult(x_next, step, abs(f_next), history)
        x_prev, x_curr = x_curr, x_next
        f_prev, f_curr = f_curr, f_next
    raise ArithmeticError("метод хорд не збігся")


def muller_parabola(f, x0, x1, x2, eps=EPS, max_iter=200):
    _check_solver_args(eps, max_iter)
    if len({x0, x1, x2}) < 3:
        raise ValueError("потрібні три різні початкові точки")
    x_2, x_1, x_curr = float(x0), float(x1), float(x2)
    f_2, f_1, f_curr = f(x_2), f(x_1), f(x_curr)
    history = [abs(f_curr)]
    for step in range(1, max_iter + 1):
        if x_curr == x_1 or x_1 == x_2 or x_curr == x_2:
            raise ZeroDivisionError("вузли збіглись у методі парабол")
        d10 = (f_curr - f_1) / (x_curr - x_1)
        d21 = (f_1 - f_2) / (x_1 - x_2)
        a = (d10 - d21) / (x_curr - x_2)
        b = (x_curr - x_1) * a + d10
        disc = cmath.sqrt(b * b - 4 * a * f_curr)
        # більший знаменник у -2c/(b +- sqrt) дає менший |Delta|
        den_pos, den_neg = b + disc, b - disc
        denom = den_pos if abs(den_pos) >= abs(den_neg) else den_neg
        if denom == 0:
            raise ZeroDivisionError("нульовий знаменник у методі парабол")
        x_next = x_curr - 2 * f_curr / denom
        if isinstance(x_next, complex) and abs(x_next.imag) < 1e-14:
            x_next = x_next.real
        if not cmath.isfinite(x_next):
            raise ArithmeticError("метод парабол розбігся")
        try:
            f_next = f(x_next)
        except TypeError as exc:
            raise ValueError("метод вийшов у комплексну область") from exc
        history.append(abs(f_next))
        if abs(f_next) < eps and abs(x_next - x_curr) < eps:
            return RootResult(x_next, step, abs(f_next), history)
        x_2, x_1, x_curr = x_1, x_curr, x_next
        f_2, f_1, f_curr = f_1, f_curr, f_next
    raise ArithmeticError("метод парабол не збігся")


def inverse_interpolation(f, x0, x1, x2, eps=EPS, max_iter=200):
    _check_solver_args(eps, max_iter)
    if len({x0, x1, x2}) < 3:
        raise ValueError("потрібні три різні початкові точки")
    x_2, x_1, x_curr = float(x0), float(x1), float(x2)
    y_2, y_1, y_curr = float(f(x_2)), float(f(x_1)), float(f(x_curr))
    history = [abs(y_curr)]
    for step in range(1, max_iter + 1):
        if y_2 == y_1 or y_2 == y_curr or y_1 == y_curr:
            raise ZeroDivisionError("збіглись значення y в інтерполяції")
        x_next = (
            y_1 * y_curr / ((y_2 - y_1) * (y_2 - y_curr)) * x_2
            + y_2 * y_curr / ((y_1 - y_2) * (y_1 - y_curr)) * x_1
            + y_2 * y_1 / ((y_curr - y_2) * (y_curr - y_1)) * x_curr
        )
        if not math.isfinite(x_next):
            raise ArithmeticError("зворотна інтерполяція розбіглась")
        y_next = float(f(x_next))
        history.append(abs(y_next))
        if abs(y_next) < eps and abs(x_next - x_curr) < eps:
            return RootResult(x_next, step, abs(y_next), history)
        x_2, x_1, x_curr = x_1, x_curr, x_next
        y_2, y_1, y_curr = y_1, y_curr, y_next
    raise ArithmeticError("зворотна інтерполяція не збіглась")


def read_poly_coeffs(path=POLY_FILE):
    # коефіцієнти в порядку зростання степенів a_0, a_1, ..., a_m
    tokens = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        clean = line.split("#", 1)[0].replace(",", " ")
        tokens.extend(clean.split())
    if not tokens:
        raise ValueError("файл коефіцієнтів порожній")
    coeffs = [float(token) for token in tokens]
    if len(coeffs) > 1 and coeffs[-1] == 0:
        raise ValueError("старший коефіцієнт не може бути нулем")
    return coeffs


def horner_eval(coeffs, x, return_quotient=False):
    if not coeffs:
        raise ValueError("список коефіцієнтів порожній")
    m = len(coeffs) - 1
    b = [0] * (m + 1)
    b[m] = coeffs[m]
    for i in range(m - 1, -1, -1):
        b[i] = coeffs[i] + x * b[i + 1]
    if return_quotient:
        return b[0], b[1:]
    return b[0]


def newton_horner(coeffs, x0, eps=EPS, max_iter=100):
    _check_solver_args(eps, max_iter)
    if len(coeffs) < 2 or coeffs[-1] == 0:
        raise ValueError("потрібен многочлен степеня >= 1")
    x = float(x0)
    b0, q = horner_eval(coeffs, x, return_quotient=True)
    history = [abs(b0)]
    for step in range(1, max_iter + 1):
        c1 = horner_eval(q, x)
        if c1 == 0:
            raise ZeroDivisionError("нульова похідна в Ньютона-Горнера")
        x_next = x - b0 / c1
        if not math.isfinite(x_next):
            raise ArithmeticError("метод Ньютона-Горнера розбігся")
        b0_next, q_next = horner_eval(coeffs, x_next, return_quotient=True)
        history.append(abs(b0_next))
        if abs(b0_next) < eps and abs(x_next - x) < eps:
            return RootResult(x_next, step, abs(b0_next), history)
        x, b0, q = x_next, b0_next, q_next
    raise ArithmeticError("метод Ньютона-Горнера не збігся")


def lin_method(coeffs, alpha0, beta0, eps=EPS, max_iter=200):
    _check_solver_args(eps, max_iter)
    m = len(coeffs) - 1
    if m < 3 or coeffs[-1] == 0:
        raise ValueError("метод Ліна потребує степінь >= 3")
    if not (math.isfinite(alpha0) and math.isfinite(beta0)):
        raise ValueError("початкові наближення мають бути скінченними")
    a0, a1 = coeffs[0], coeffs[1]
    alpha, beta = float(alpha0), float(beta0)
    history = [(alpha, beta)]
    for step in range(1, max_iter + 1):
        p0 = -2.0 * alpha
        q0 = alpha * alpha + beta * beta
        b = [0.0] * (m + 1)
        b[m] = float(coeffs[m])
        b[m - 1] = coeffs[m - 1] - p0 * b[m]
        for i in range(m - 2, 1, -1):
            b[i] = coeffs[i] - p0 * b[i + 1] - q0 * b[i + 2]
        if b[2] == 0:
            raise ZeroDivisionError("нульовий b2 в методі Ліна")
        q1 = a0 / b[2]
        p1 = (a1 * b[2] - a0 * b[3]) / (b[2] * b[2])
        alpha1 = -0.5 * p1
        disc = q1 - alpha1 * alpha1
        if disc < 0:
            raise ValueError("від'ємний дискримінант у методі Ліна")
        beta1 = math.sqrt(disc)
        history.append((alpha1, beta1))
        if abs(alpha1 - alpha) <= eps and abs(beta1 - beta) <= eps:
            residual = abs(horner_eval(coeffs, complex(alpha1, beta1)))
            return LinResult(alpha1, beta1, step, residual, history)
        alpha, beta = alpha1, beta1
    raise ArithmeticError("метод Ліна не збігся")


def solve_all_methods(x0, tau):
    xm1, xm2 = x0 - 0.05, x0 + 0.05
    return [
        ("Simple iteration", simple_iteration(f_trans, x0, tau, EPS)),
        ("Newton", newton(f_trans, df_trans, x0, EPS)),
        ("Chebyshev", chebyshev(f_trans, df_trans, d2f_trans, x0, EPS)),
        ("Secant", secant(f_trans, xm1, x0, EPS)),
        ("Muller (parabola)", muller_parabola(f_trans, xm2, xm1, x0, EPS)),
        ("Inverse interp.",
         inverse_interpolation(f_trans, xm2, xm1, x0, EPS)),
    ]


def experiment(tab_path=TAB_FILE, poly_path=POLY_FILE):
    nodes = tabulate(f_trans, A, B, STEP)
    save_tabulation(nodes, tab_path)
    loaded_nodes = read_tabulation(tab_path)
    brackets = find_brackets(loaded_nodes)
    taus = {"increasing": -0.5, "decreasing": 0.5}
    root_runs = []
    for left, right, x0, trend in brackets:
        tau = taus[trend]
        methods = solve_all_methods(x0, tau)
        root_runs.append(dict(
            bracket=(left, right),
            x0=x0,
            trend=trend,
            tau=tau,
            stability=tau * df_trans(x0),
            methods=methods,
        ))
    coeffs = read_poly_coeffs(poly_path)
    real_x0 = 6.0
    alpha0, beta0 = 0.5, 0.5
    real_root = newton_horner(coeffs, real_x0, EPS)
    complex_root = lin_method(coeffs, alpha0, beta0, EPS)
    return dict(
        nodes=loaded_nodes,
        brackets=brackets,
        root_runs=root_runs,
        coeffs=coeffs,
        real_x0=real_x0,
        real_root=real_root,
        lin_start=(alpha0, beta0),
        complex_root=complex_root,
    )


def print_results(result):
    print(f"[a, b] = [{A:g}, {B:g}]; h = {STEP:g}; eps = {EPS:.0e}")
    for idx, run in enumerate(result["root_runs"], 1):
        left, right = run["bracket"]
        print(
            f"root {idx} ({run['trend']}): [{left:g}, {right:g}], "
            f"x0 = {run['x0']:g}, tau = {run['tau']:g}, "
            f"tau*F'(x0) = {run['stability']:.6f}"
        )
        print(f"{'method':<18} {'iter':>4} {'root':>17} {'|F(x)|':>12}")
        for name, res in run["methods"]:
            print(f"{name:<18} {res.iterations:4d} "
                  f"{res.root:17.12f} {res.residual:12.3e}")
    print(f"coeffs = {result['coeffs']}")
    rr = result["real_root"]
    print(
        f"Newton-Horner (x0 = {result['real_x0']:g}): "
        f"iter = {rr.iterations}, x = {rr.root:.12f}, "
        f"|P(x)| = {rr.residual:.3e}"
    )
    cr = result["complex_root"]
    a0, b0 = result["lin_start"]
    print(
        f"Lin (alpha0 = {a0:g}, beta0 = {b0:g}): iter = {cr.iterations}, "
        f"z = {cr.alpha:.12f} +- {cr.beta:.12f}j, |P(z)| = {cr.residual:.3e}"
    )


def make_plots(result):
    fig1, ax1 = plt.subplots(figsize=(9, 4.2), layout="constrained")
    x_grid = np.linspace(A, B, 400)
    y_grid = [f_trans(t) for t in x_grid]
    ax1.plot(x_grid, y_grid, label="F(x) = 2sin(x) - 0.3x - 0.5")
    ax1.axhline(0, color="black", linewidth=0.8)
    colors = ["#2ca02c", "#d62728"]
    labels = ["Корінь 1 (F'(x) > 0)", "Корінь 2 (F'(x) < 0)"]
    for run, col, lbl in zip(result["root_runs"], colors, labels):
        xr = run["methods"][1][1].root
        ax1.scatter([xr], [0.0], color=col, zorder=5,
                    label=f"{lbl}: x* = {xr:.4f}")
    ax1.set(xlabel="x", ylabel="F(x)",
            title="Відокремлення коренів трансцендентного рівняння")
    ax1.grid(alpha=0.3)
    ax1.legend()

    fig2, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    titles = [
        "Збіжність для кореня 1 (F'(x) > 0)",
        "Збіжність для кореня 2 (F'(x) < 0)",
    ]
    for ax, run, title in zip(axes, result["root_runs"], titles):
        for name, res in run["methods"]:
            iters = list(range(len(res.history)))
            # піднімаєм нульові нев'язки до машинного eps для log-шкали
            errs = [max(err, 1e-16) for err in res.history]
            ax.semilogy(iters, errs, marker="o", markersize=4,
                        label=f"{name} ({res.iterations})")
        ax.axhline(EPS, color="black", linestyle="--", linewidth=1,
                   label="eps = 1e-10")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlabel="Ітерація k", ylabel="|F(x_k)|", title=title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig3, ax3 = plt.subplots(figsize=(9, 4.2), layout="constrained")
    px = np.linspace(0.0, 6.2, 400)
    py = [horner_eval(result["coeffs"], float(t)) for t in px]
    ax3.plot(px, py, label="P(x) = x^3 - 7x^2 + 12x - 10")
    ax3.axhline(0, color="black", linewidth=0.8)
    xr = result["real_root"].root
    ax3.scatter([xr], [0.0], color="#d62728", zorder=5,
                label=f"Дійсний корінь x* = {xr:.4f}")
    ax3.set(xlabel="x", ylabel="P(x)",
            title="Кубічне рівняння з одним дійсним та двома комплексними коренями")
    ax3.grid(alpha=0.3)
    ax3.legend()
    return [fig1, fig2, fig3]


def main():
    result = experiment()
    print_results(result)
    make_plots(result)
    plt.show()


if __name__ == "__main__":
    main()
