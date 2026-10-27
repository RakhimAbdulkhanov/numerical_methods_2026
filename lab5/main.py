import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


A, B = 0.0, 24.0
TARGET = 1e-12
mp.mp.dps = 60


def load(x):
    return (50 + 20 * math.sin(math.pi * x / 12)
            + 5 * math.exp(-0.2 * (x - 12)**2))


def exact_integral():
    # синус за повну добу дає нуль, гаусіан інтегруєм через erf
    c = mp.mpf("0.2")
    return 1200 + 5 * mp.sqrt(mp.pi / c) * mp.erf(12 * mp.sqrt(c))


def absolute_error(value, exact):
    return float(abs(mp.mpf(value) - exact))


def simpson(function, a, b, n):
    if not isinstance(n, int) or isinstance(n, bool) or n < 2 or n % 2:
        raise ValueError("n має бути додатним парним числом відрізків")
    h = (b - a) / n
    terms = [function(a), function(b)]
    terms.extend((4 if i % 2 else 2) * function(a + i * h)
                 for i in range(1, n))
    return h * math.fsum(terms) / 3


def runge_romberg(fine, coarse):
    return fine + (fine - coarse) / 15


def aitken(fine, medium, coarse):
    first, second = medium - fine, coarse - medium
    denominator = second - first
    if denominator == 0 or first == 0 or second == 0:
        raise ValueError("недостатньо різниці між наближеннями")
    return fine - first * first / denominator, math.log2(abs(second / first))


@dataclass
class AdaptiveResult:
    value: float
    estimated_error: float
    evaluations: int
    intervals: int


def adaptive_simpson(function, a, b, delta, max_depth=40):
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta має бути додатним і скінченним")
    if b <= a:
        raise ValueError("потрібно a < b")
    evaluations = 0

    def evaluate(x):
        nonlocal evaluations
        evaluations += 1
        return function(x)

    def panel(left, right, fl, fm, fr):
        return (right - left) * (fl + 4 * fm + fr) / 6

    def refine(left, right, fl, fm, fr, whole, tolerance, depth):
        middle = (left + right) / 2
        q1, q3 = (left + middle) / 2, (middle + right) / 2
        if q1 == left or q3 == right:
            raise ArithmeticError("крок досяг межі машинної точності")
        fq1, fq3 = evaluate(q1), evaluate(q3)
        lower = panel(left, middle, fl, fq1, fm)
        upper = panel(middle, right, fm, fq3, fr)
        correction = (lower + upper - whole) / 15
        if abs(correction) <= tolerance:
            return lower + upper + correction, abs(correction), 1
        if depth == 0:
            raise ArithmeticError("вичерпано глибину адаптивного поділу")
        lval, lerr, lc = refine(left, middle, fl, fq1, fm, lower,
                                tolerance / 2, depth - 1)
        rval, rerr, rc = refine(middle, right, fm, fq3, fr, upper,
                                tolerance / 2, depth - 1)
        return math.fsum((lval, rval)), lerr + rerr, lc + rc

    fl, fm, fr = evaluate(a), evaluate((a + b) / 2), evaluate(b)
    value, error, intervals = refine(a, b, fl, fm, fr,
                                     panel(a, b, fl, fm, fr), delta, max_depth)
    return AdaptiveResult(value, error, evaluations, intervals)


def experiment():
    exact = exact_integral()
    counts = list(range(10, 1001, 2))
    values = [simpson(load, A, B, n) for n in counts]
    errors = [absolute_error(value, exact) for value in values]
    n_opt = next((n for n, error in zip(counts, errors)
                  if error <= TARGET), None)
    if n_opt is None:
        raise ArithmeticError("цільову точність не досягнуто до n=1000")
    allowed = [n for n in range(16, n_opt // 8 + 1, 8)]
    # за порожнього діапазону явно використовуєм демонстраційне n0
    n0 = allowed[-1] if allowed else 16
    fine, medium, coarse = [simpson(load, A, B, n0 // k)
                            for k in (1, 2, 4)]
    improved, order = aitken(fine, medium, coarse)
    comparisons = [(f"S({n0})", fine), (f"S({n0 // 2})", medium),
                   (f"S({n0 // 4})", coarse),
                   ("Runge-Romberg", runge_romberg(fine, medium)),
                   ("Aitken", improved)]
    adaptive = [(delta, adaptive_simpson(load, A, B, delta))
                for delta in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)]
    return dict(exact=exact, counts=counts, errors=errors,
                n_opt=n_opt, opt_error=errors[counts.index(n_opt)],
                n0=n0, n0_compliant=bool(allowed), comparisons=comparisons,
                order=order, adaptive=adaptive)


def print_results(result):
    print(f"[a, b] = [{A:g}, {B:g}]; target = {TARGET:.0e}")
    print(f"I0 = {mp.nstr(result['exact'], 35)}")
    print(f"Nopt = {result['n_opt']}; error = {result['opt_error']:.6e}")
    if not result['n0_compliant']:
        print(f"No N0: 10 <= N0 <= {result['n_opt'] / 8:g}, N0 % 8 = 0")
        print(f"Demonstration N0 = {result['n0']} (outside prescribed range)")
    print(f"{'method':<15} {'integral':>21} {'absolute error':>15}")
    for name, value in result['comparisons']:
        print(f"{name:<15} {value:21.13f} "
              f"{absolute_error(value, result['exact']):15.6e}")
    print(f"p = {result['order']:.9f}")
    print(f"{'delta':>9} {'integral':>20} {'actual error':>13} "
          f"{'estimate':>12} {'calls':>6}")
    for delta, row in result['adaptive']:
        print(f"{delta:9.0e} {row.value:20.12f} "
              f"{absolute_error(row.value, result['exact']):13.5e} "
              f"{row.estimated_error:12.4e} {row.evaluations:6d}")


def make_plots(result):
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    x = np.linspace(A, B, 500)
    ax.plot(x, [load(t) for t in x])
    ax.set(xlabel="Час x, год", ylabel="Навантаження f(x), % CPU",
           title="Модель навантаження сервера за добу")
    ax.grid(alpha=0.3)
    fig2, ax2 = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax2.semilogy(result['counts'], result['errors'], linewidth=1)
    ax2.axhline(TARGET, color="red", linestyle="--", label="ціль 1e-12")
    ax2.scatter(result['n_opt'], result['opt_error'], color="red",
                label=f"перше досягнення: N = {result['n_opt']}")
    ax2.set(xlabel="Число відрізків N", ylabel="Абсолютна похибка",
            title="Складова формула Сімпсона: похибка для N = 10..1000")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig3, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    tolerances = [delta for delta, _ in result['adaptive']]
    actual = [absolute_error(row.value, result['exact'])
              for _, row in result['adaptive']]
    calls = [row.evaluations for _, row in result['adaptive']]
    axes[0].loglog(tolerances, actual, "o-", label="фактична похибка")
    axes[0].loglog(tolerances, tolerances, "--", label="похибка = delta")
    axes[0].set(xlabel="Параметр delta", ylabel="Абсолютна похибка")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(tolerances, calls, "o-")
    axes[1].set(xlabel="Параметр delta", ylabel="Обчислення f(x)")
    for axis in axes:
        axis.invert_xaxis()
        axis.grid(alpha=0.3)
    return [fig, fig2, fig3]


def main():
    result = experiment()
    print_results(result)
    make_plots(result)
    plt.show()


if __name__ == "__main__":
    main()
