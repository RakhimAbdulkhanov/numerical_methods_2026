import csv
import math
import numpy as np
import matplotlib.pyplot as plt

# читаєм експериментальні дані профілювання з csv
def read_data(filename):
    x = []
    y = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row["n"]))
            y.append(float(row["t"]))
    return np.array(x, dtype=float), np.array(y, dtype=float)

# будуєм таблицю розділених різниць рекурентно
def build_divided_diff_table(x, y):
    n = len(x)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
    return table

# обчислюєм поліном ньютона у довільній точці
def evaluate_newton(x_val, x_nodes, coeffs):
    x_val = np.asarray(x_val, dtype=float)
    scalar_input = False
    if x_val.ndim == 0:
        x_val = x_val[None]
        scalar_input = True
    res = np.zeros_like(x_val)
    for idx, xv in np.ndenumerate(x_val):
        val = coeffs[0]
        term = 1.0
        for j in range(1, len(coeffs)):
            term *= (xv - x_nodes[j - 1])
            val += coeffs[j] * term
        res[idx] = val
    return res[0] if scalar_input else res

# факторіальний многочлен через скінченні різниці
def factorial_poly_eval(u_val, diffs_list):
    val = diffs_list[0][0]
    term = 1.0
    for k in range(1, len(diffs_list)):
        term *= (u_val - (k - 1))
        val += diffs_list[k][0] / math.factorial(k) * term
    return val

# інтерполяція лагранжа для порівняння
def evaluate_lagrange(x_val, x_nodes, y_nodes):
    x_val = np.asarray(x_val, dtype=float)
    scalar_input = False
    if x_val.ndim == 0:
        x_val = x_val[None]
        scalar_input = True
    n = len(x_nodes)
    res = np.zeros_like(x_val)
    for idx, xv in np.ndenumerate(x_val):
        total = 0.0
        for i in range(n):
            term = y_nodes[i]
            for j in range(n):
                if i != j:
                    term *= (xv - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
            total += term
        res[idx] = total
    return res[0] if scalar_input else res

# тестова функція для дослідження ефекту рунге
def benchmark_profile(x):
    # моделюєм час алгоритму з локальним стрибком через кеш-промахи
    x_norm = (x - 8500.0) / 7500.0
    return 0.0045 * x + 0.0000035 * (x ** 1.15) + 35.0 / (1.0 + 25.0 * (x_norm ** 2))

def main():
    # завантажуєм точки
    data_path = "lab2/task/data.csv"
    try:
        x_nodes, y_nodes = read_data(data_path)
    except FileNotFoundError:
        x_nodes, y_nodes = read_data("task/data.csv")

    n = len(x_nodes)
    table = build_divided_diff_table(x_nodes, y_nodes)
    newton_coeffs = table[0, :]

    print("Таблиця розділених різниць:")
    col_names = [f"f[x0..x{k}]" for k in range(n)]
    header = f"{'n':>8} | {'t (мс)':>8} | " + " | ".join([f"{name:>14}" for name in col_names[1:]])
    print(header)
    print("-" * len(header))
    for i in range(n):
        row_vals = [f"{x_nodes[i]:8.0f}", f"{y_nodes[i]:8.2f}"]
        for j in range(1, n - i):
            row_vals.append(f"{table[i, j]:14.6e}")
        for j in range(n - i, n):
            row_vals.append(f"{'-':>14}")
        print(" | ".join(row_vals))

    print("\nКоефіцієнти многочлена Ньютона:")
    for k, c in enumerate(newton_coeffs):
        print(f"a_{k} = {c:16.8e}")

    # прогноз для n = 6000 методом ньютона
    target_n = 6000.0
    pred_newton = evaluate_newton(target_n, x_nodes, newton_coeffs)
    pred_lagrange = evaluate_lagrange(target_n, x_nodes, y_nodes)

    # факторіальний многочлен по логарифмічній сітці u = log2(n / 1000)
    u_nodes = np.array([0, 1, 2, 3, 4], dtype=float)
    diffs = [y_nodes.copy()]
    for k in range(1, n):
        d = np.diff(diffs[-1])
        diffs.append(d)

    u_target = math.log2(target_n / 1000.0)
    pred_factorial = factorial_poly_eval(u_target, diffs)

    print(f"\nПрогнозування часу виконання для n = {target_n:.0f}:")
    print(f"  Многочлен Ньютона:       {pred_newton:.6f} мс")
    print(f"  Многочлен Лагранжа:      {pred_lagrange:.6f} мс")
    print(f"  Факторіальний многочлен: {pred_factorial:.6f} мс")
    print(f"  Різниця Ньютон-Лагранж:  {abs(pred_newton - pred_lagrange):.2e} мс")

    # дослідження стійкості та ефекту рунге на 5, 10, 20 вузлах
    print("\nДослідження впливу кількості вузлів (5, 10, 20) та ефекту Рунге:")
    x_grid = np.linspace(1000.0, 16000.0, 400)
    y_true = benchmark_profile(x_grid)

    runge_errors = {}
    node_counts = [5, 10, 20]
    for n_pts in node_counts:
        # рівномірні вузли
        unif_x = np.linspace(1000.0, 16000.0, n_pts)
        unif_y = benchmark_profile(unif_x)
        tbl_unif = build_divided_diff_table(unif_x, unif_y)
        y_poly_unif = evaluate_newton(x_grid, unif_x, tbl_unif[0, :])
        err_unif = np.max(np.abs(y_poly_unif - y_true))
        runge_errors[n_pts] = (err_unif, y_poly_unif, unif_x, unif_y)
        print(f"  Кількість вузлів N = {n_pts:2d} | Максимальна похибка інтерполяції: {err_unif:12.6f} мс")

    # графічна візуалізація
    plt.figure(figsize=(13, 10))

    # графік 1: інтерполяція ньютона та прогноз
    plt.subplot(2, 2, 1)
    x_plot = np.linspace(1000.0, 16000.0, 300)
    y_plot_newton = evaluate_newton(x_plot, x_nodes, newton_coeffs)
    plt.plot(x_plot, y_plot_newton, "b-", label="Многочлен Ньютона $P_4(n)$", linewidth=2)
    plt.scatter(x_nodes, y_nodes, color="darkred", s=60, zorder=5, label="Експериментальні дані")
    plt.scatter([target_n], [pred_newton], color="green", s=80, marker="D", zorder=6,
                label=f"Прогноз n=6000 ({pred_newton:.2f} мс)")
    plt.title("Інтерполяція Ньютона та прогноз t(n)")
    plt.xlabel("Розмір вхідних даних n")
    plt.ylabel("Час виконання t (мс)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # графік 2: спадання коефіцієнтів розділених різниць
    plt.subplot(2, 2, 2)
    orders = np.arange(n)
    abs_coeffs = np.abs(newton_coeffs)
    plt.semilogy(orders, abs_coeffs, "ro-", linewidth=2, markersize=7)
    plt.title("Спадання коефіцієнтів розділених різниць $|a_k|$")
    plt.xlabel("Порядок розділеної різниці k")
    plt.ylabel("Величина коефіцієнта $|f[x_0..x_k]|$")
    plt.xticks(orders)
    plt.grid(True, linestyle="--", alpha=0.6)

    # графік 3: ефект рунге при N = 5, 10, 20 вузлах
    plt.subplot(2, 2, 3)
    plt.plot(x_grid, y_true, "k--", label="Еталонна функція", linewidth=2)
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    for idx, n_pts in enumerate(node_counts):
        err, y_poly, nx, ny = runge_errors[n_pts]
        plt.plot(x_grid, y_poly, label=f"N={n_pts} вузлів (max err={err:.1f})", color=colors[idx], linewidth=1.5)
    plt.ylim(-20, 140)
    plt.title("Ефект Рунге при збільшенні кількості вузлів")
    plt.xlabel("Розмір вхідних даних n")
    plt.ylabel("Час t (мс)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # графік 4: похибка ньютон vs лагранж
    plt.subplot(2, 2, 4)
    y_plot_lagrange = evaluate_lagrange(x_plot, x_nodes, y_nodes)
    diff_poly = np.abs(y_plot_newton - y_plot_lagrange)
    plt.plot(x_plot, diff_poly, "m-", label="|P_Newton(n) - L_Lagrange(n)|", linewidth=2)
    plt.title("Порівняння методів Ньютона і Лагранжа")
    plt.xlabel("Розмір вхідних даних n")
    plt.ylabel("Абсолютна різниця (мс)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
