import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# зчитуєм середньомісячну температуру з csv файлу
def load_data(filepath):
    months = []
    temps = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and len(row) >= 2:
                months.append(float(row[0]))
                temps.append(float(row[1]))
    return np.array(months, dtype=float), np.array(temps, dtype=float)

# формуєм матрицю нормальних рівнянь для степеня m
def form_matrix(x, m):
    a_mat = np.zeros((m + 1, m + 1), dtype=float)
    for i in range(m + 1):
        for j in range(m + 1):
            a_mat[i, j] = np.sum(x ** (i + j))
    return a_mat

# формуєм вектор правої частини нормальних рівнянь
def form_vector(x, y, m):
    b_vec = np.zeros(m + 1, dtype=float)
    for i in range(m + 1):
        b_vec[i] = np.sum(y * (x ** i))
    return b_vec

# метод гауса з вибором головного елемента по стовпцях
def gauss_solve(a_in, b_in):
    a = a_in.copy()
    b = b_in.copy()
    n = len(b)

    # прямий хід з перестановкою рядків
    for k in range(n):
        max_row = k + np.argmax(np.abs(a[k:, k]))
        if abs(a[max_row, k]) < 1e-14:
            raise ValueError("матриця вироджена")

        if max_row != k:
            a[[k, max_row]] = a[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = a[i, k] / a[k, k]
            a[i, k:] -= factor * a[k, k:]
            b[i] -= factor * b[k]

    # зворотній хід
    x_sol = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        sum_ax = np.sum(a[i, i + 1:] * x_sol[i + 1:])
        x_sol[i] = (b[i] - sum_ax) / a[i, i]

    return x_sol

# обчислюєм значення полінома в точках x
def evaluate_poly(x_vals, coef):
    y_vals = np.zeros_like(x_vals, dtype=float)
    for i, c in enumerate(coef):
        y_vals += c * (x_vals ** i)
    return y_vals

# рахуєм дисперсію та середньоквадратичну похибку
def calculate_variance(y_true, y_approx, m):
    n = len(y_true)
    sse = np.sum((y_true - y_approx) ** 2)
    s2 = sse / (n - m - 1)
    mse = sse / n
    return s2, mse

# форматування многочлена у зручний рядок
def format_poly(coef):
    terms = []
    for i, c in enumerate(coef):
        if i == 0:
            terms.append(f"{c:.4f}")
        else:
            sign = " + " if c >= 0 else " - "
            power = f"*x^{i}" if i > 1 else "*x"
            terms.append(f"{sign}{abs(c):.4f}{power}")
    return "".join(terms)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "task", "temperature.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join("lab3", "task", "temperature.csv")

    x, y = load_data(csv_path)
    n = len(x)

    print("Таблиця вхідних даних (24 місяці):")
    print("Місяць: ", " ".join(f"{int(val):3d}" for val in x))
    print("Темп.:  ", " ".join(f"{int(val):3d}" for val in y))
    print("-" * 75)

    degrees = [1, 2, 3, 4]
    poly_coefs = {}
    approximations = {}
    variances = {}
    mses = {}

    # апроксимація поліномами різного степеня
    for m in degrees:
        a_mat = form_matrix(x, m)
        b_vec = form_vector(x, y, m)
        coef = gauss_solve(a_mat, b_vec)
        poly_coefs[m] = coef

        y_approx = evaluate_poly(x, coef)
        approximations[m] = y_approx

        s2, mse = calculate_variance(y, y_approx, m)
        variances[m] = s2
        mses[m] = mse

        print(f"Степінь m = {m}:")
        print(f"  P_{m}(x) = {format_poly(coef)}")
        print(f"  sigma^2 = {s2:.4f}, MSE = {mse:.4f}, RMSE = {np.sqrt(mse):.4f}")

    print("-" * 75)
    optimal_m = min(variances, key=variances.get)
    print(f"Оптимальний степінь за мінімумом дисперсії: m = {optimal_m} (sigma^2 = {variances[optimal_m]:.4f})")
    print("-" * 75)

    # табуляція похибок апроксимації
    print("Табуляція значень та абсолютних похибок:")
    print(f"{'Місяць':<7}{'T_факт':<8}{'P1(x)':<9}{'|e1|':<8}{'P2(x)':<9}{'|e2|':<8}{'P3(x)':<9}{'|e3|':<8}{'P4(x)':<9}{'|e4|':<8}")
    for i in range(n):
        xi = x[i]
        yi = y[i]
        e1 = abs(yi - approximations[1][i])
        e2 = abs(yi - approximations[2][i])
        e3 = abs(yi - approximations[3][i])
        e4 = abs(yi - approximations[4][i])
        print(f"{int(xi):<7}{yi:<8.1f}{approximations[1][i]:<9.2f}{e1:<8.2f}{approximations[2][i]:<9.2f}{e2:<8.2f}{approximations[3][i]:<9.2f}{e3:<8.2f}{approximations[4][i]:<9.2f}{e4:<8.2f}")
    print("-" * 75)

    # прогноз на наступні 3 місяці
    future_months = np.array([25, 26, 27], dtype=float)
    print("Прогноз температури на наступні 3 місяці (місяці 25, 26, 27):")
    for m in degrees:
        pred = evaluate_poly(future_months, poly_coefs[m])
        pred_str = ", ".join(f"місяць {int(fm)}: {val:.2f} C" for fm, val in zip(future_months, pred))
        print(f"  m = {m}: {pred_str}")
    print("-" * 75)

    # побудова графіків
    x_dense = np.linspace(1, 27, 300)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # графік 1: апроксимація та екстраполяція
    axs[0].scatter(x, y, color="black", s=30, label="Фактичні дані (1-24 міс)", zorder=5)
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    for m in degrees:
        y_curve = evaluate_poly(x_dense, poly_coefs[m])
        axs[0].plot(x_dense, y_curve, label=f"МНК m={m}", color=colors[m], linewidth=1.8)
        pred_fut = evaluate_poly(future_months, poly_coefs[m])
        axs[0].scatter(future_months, pred_fut, color=colors[m], marker="s", s=35)

    axs[0].axvline(x=24.5, color="gray", linestyle="--", alpha=0.7, label="Межа прогнозу")
    axs[0].set_title("Апроксимація температури методом найменших квадратів та прогноз на 3 місяці")
    axs[0].set_xlabel("Місяць")
    axs[0].set_ylabel("Температура (C)")
    axs[0].set_xticks(range(1, 28))
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="upper left", fontsize=9)

    # графік 2: похибка апроксимації
    for m in degrees:
        err = np.abs(y - approximations[m])
        axs[1].plot(x, err, marker="o", markersize=4, label=f"Похибка |e| (m={m})", color=colors[m], linewidth=1.5)

    axs[1].set_title("Абсолютна похибка апроксимації |y_i - P_m(x_i)| за місяцями")
    axs[1].set_xlabel("Місяць")
    axs[1].set_ylabel("Похибка (C)")
    axs[1].set_xticks(range(1, 25))
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend(loc="upper right", fontsize=9)

    # графік 3: зміна дисперсії
    deg_list = list(variances.keys())
    var_list = [variances[d] for d in deg_list]
    axs[2].plot(deg_list, var_list, marker="s", markersize=7, color="#d62728", linewidth=2, label="Незсунена дисперсія sigma^2")
    axs[2].set_title("Залежність дисперсії від степеня апроксимуючого многочлена")
    axs[2].set_xlabel("Степінь многочлена m")
    axs[2].set_ylabel("Дисперсія sigma^2")
    axs[2].set_xticks(deg_list)
    for d, v in zip(deg_list, var_list):
        axs[2].annotate(f"{v:.2f}", (d, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    axs[2].grid(True, linestyle="--", alpha=0.5)
    axs[2].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
