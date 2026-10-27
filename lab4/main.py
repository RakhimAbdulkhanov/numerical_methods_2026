import math

import matplotlib.pyplot as plt
import numpy as np


X0 = 1.0
H = 1e-3


def moisture(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)


def derivative(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)


def central_difference(function, x, h):
    if not math.isfinite(h) or h <= 0:
        raise ValueError("крок має бути додатним і скінченним")
    return (function(x + h) - function(x - h)) / (2 * h)


def runge_romberg(fine, coarse, order=2):
    return fine + (fine - coarse) / (2**order - 1)


def aitken(fine, medium, coarse):
    first = medium - fine
    second = coarse - medium
    denominator = second - first
    if denominator == 0 or first == 0 or second == 0:
        raise ValueError("недостатньо різниці між наближеннями")
    # різницева форма не віднімає квадрати близьких чисел
    refined = fine - first * first / denominator
    order = math.log2(abs(second / first))
    return refined, order


def experiment():
    exact = derivative(X0)
    # перевіряєм мантиси 1..9 для кожного десяткового порядку
    steps = np.array(sorted({m * 10.0**e
                             for e in range(-20, 4)
                             for m in range(1, 10)
                             if m * 10.0**e <= 1e3}))
    values = np.array([central_difference(moisture, X0, h)
                       for h in steps])
    errors = np.abs(values - exact)
    best = int(np.argmin(errors))
    fine, medium, coarse = [central_difference(moisture, X0, k * H)
                            for k in (1, 2, 4)]
    rr = runge_romberg(fine, medium)
    improved, order = aitken(fine, medium, coarse)
    comparisons = [
        ("D(h)", fine), ("D(2h)", medium), ("D(4h)", coarse),
        ("Runge-Romberg", rr), ("Aitken", improved),
    ]
    return dict(exact=exact, steps=steps, errors=errors,
                best_h=float(steps[best]), best_error=float(errors[best]),
                comparisons=comparisons, order=order)


def print_results(result):
    print(f"x0 = {X0:g}; h = {H:g}")
    print(f"M'(1) = {result['exact']:.15f}")
    print(f"h0 = {result['best_h']:.0e}")
    print(f"R0 = {result['best_error']:.6e}")
    print(f"{'method':<15} {'derivative':>20} {'absolute error':>15}")
    for name, value in result['comparisons']:
        error = abs(value - result['exact'])
        print(f"{name:<15} {value:20.15f} {error:15.6e}")
    print(f"p = {result['order']:.9f}")


def make_plots(result):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.loglog(result['steps'], result['errors'], ".-", markersize=3)
    ax.scatter(result['best_h'], result['best_error'], color="red",
               zorder=5, label=f"найкращий перевірений h = {result['best_h']:.0e}")
    ax.set(xlabel="Крок h", ylabel="Абсолютна похибка",
           title="Центральна різниця: апроксимація та округлення")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig2, ax2 = plt.subplots(figsize=(9, 4), layout="constrained")
    labels = [name for name, _ in result['comparisons']]
    errors = [abs(value - result['exact'])
              for _, value in result['comparisons']]
    ax2.bar(labels, errors, color=["#4c78a8"] * 3 + ["#59a14f"] * 2)
    ax2.set(yscale="log", ylabel="Абсолютна похибка",
            title="Порівняння методів при h = 0.001")
    ax2.grid(True, axis="y", alpha=0.3)
    return [fig, fig2]


def main():
    result = experiment()
    print_results(result)
    make_plots(result)
    plt.show()


if __name__ == "__main__":
    main()
