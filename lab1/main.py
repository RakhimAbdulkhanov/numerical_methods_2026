import os
import json
import numpy as np
import matplotlib.pyplot as plt

# гаверсинус для обчислення відстані між двома координатами
def haversine(lat1, lon1, lat2, lon2):
    r = 6371000  # радіус землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

# метод прогонки для розв'язку трьохдіагональної спарсної слау
def solve_tridiagonal(alpha, beta, gamma, delta):
    n = len(beta)
    cap_a = np.zeros(n)
    cap_b = np.zeros(n)

    # пряма прогонка
    cap_a[0] = -gamma[0] / beta[0]
    cap_b[0] = delta[0] / beta[0]

    for i in range(1, n - 1):
        denom = alpha[i] * cap_a[i - 1] + beta[i]
        cap_a[i] = -gamma[i] / denom
        cap_b[i] = (delta[i] - alpha[i] * cap_b[i - 1]) / denom

    denom_last = alpha[n - 1] * cap_a[n - 2] + beta[n - 1]

    # зворотна прогонка
    x = np.zeros(n)
    x[n - 1] = (delta[n - 1] - alpha[n - 1] * cap_b[n - 2]) / denom_last

    for i in range(n - 2, -1, -1):
        x[i] = cap_a[i] * x[i + 1] + cap_b[i]

    return x

# побудова коефіцієнтів кубічних сплайнів
def build_cubic_spline(x_nodes, y_nodes):
    m = len(x_nodes)
    n = m - 1  # кількість інтервалів

    h = np.diff(x_nodes)

    alpha = np.zeros(n)
    beta = np.zeros(n)
    gamma = np.zeros(n)
    delta = np.zeros(n)

    # перше рівняння: c1 = 0
    beta[0] = 1.0
    gamma[0] = 0.0
    delta[0] = 0.0

    # проміжні рівняння для c2..c_{n-1}
    for i in range(1, n - 1):
        alpha[i] = h[i - 1]
        beta[i] = 2.0 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3.0 * ((y_nodes[i + 1] - y_nodes[i]) / h[i] - (y_nodes[i] - y_nodes[i - 1]) / h[i - 1])

    # останнє рівняння для cn
    if n > 1:
        alpha[n - 1] = h[n - 2]
        beta[n - 1] = 2.0 * (h[n - 2] + h[n - 1])
        gamma[n - 1] = 0.0
        delta[n - 1] = 3.0 * ((y_nodes[n] - y_nodes[n - 1]) / h[n - 1] - (y_nodes[n - 1] - y_nodes[n - 2]) / h[n - 2])

    # знаходимо вектор c
    c = solve_tridiagonal(alpha, beta, gamma, delta)

    # коефіцієнти для кожного відрізка
    a = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        a[i] = y_nodes[i]
        c_curr = c[i]
        c_next = c[i + 1] if i < n - 1 else 0.0
        d[i] = (c_next - c_curr) / (3.0 * h[i])
        b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - (h[i] / 3.0) * (c_next + 2.0 * c_curr)

    return a, b, c, d, alpha, beta, gamma, delta

# обчислення значення сплайну в точці
def evaluate_spline(x_val, x_nodes, a, b, c, d):
    n = len(x_nodes) - 1
    idx = np.searchsorted(x_nodes, x_val) - 1
    idx = np.clip(idx, 0, n - 1)

    dx = x_val - x_nodes[idx]
    return a[idx] + b[idx] * dx + c[idx] * (dx**2) + d[idx] * (dx**3)

def main():
    # завантажуєм gps точки з кешу або через open-elevation api
    json_path = os.path.join(os.path.dirname(__file__), "task", "hoverla_elevation.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        import requests
        url = ("https://api.open-elevation.com/api/v1/lookup?locations="
               "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
               "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
               "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
               "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
               "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
               "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
               "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106")
        res = requests.get(url, timeout=10)
        data = res.json()

    results = data["results"]
    num_points = len(results)

    # рахуєм кумулятивну відстань
    coords = [(p["latitude"], p["longitude"]) for p in results]
    elevations = np.array([p["elevation"] for p in results])

    distances = np.zeros(num_points)
    for i in range(1, num_points):
        d_step = haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        distances[i] = distances[i - 1] + d_step

    print(f"Кількість вузлів: {num_points}")
    print("\nТабуляція вузлів:")
    print("№ | Latitude | Longitude | Elevation (m)")
    for i, p in enumerate(results):
        print(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}")

    print("\nТабуляція (відстань, висота):")
    print("№ | Distance (m) | Elevation (m)")
    for i in range(num_points):
        print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

    # побудова сплайнів для 21 вузла
    a_full, b_full, c_full, d_full, alpha, beta, gamma, delta = build_cubic_spline(distances, elevations)

    print("\nКоефіцієнти трьохдіагональної системи (alpha, beta, gamma, delta):")
    print("i | alpha | beta | gamma | delta")
    for i in range(len(beta)):
        print(f"{i+1:2d} | {alpha[i]:8.2f} | {beta[i]:8.2f} | {gamma[i]:8.2f} | {delta[i]:10.6f}")

    print("\nКоефіцієнти кубічних сплайнів для повної траси:")
    print("i | a_i (m) | b_i | c_i | d_i")
    for i in range(len(a_full)):
        print(f"{i+1:2d} | {a_full[i]:8.2f} | {b_full[i]:10.6f} | {c_full[i]:12.8f} | {d_full[i]:14.10f}")

    # характеристики маршруту
    total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, num_points))
    total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, num_points))

    print(f"\nЗагальна довжина маршруту (м): {distances[-1]:.2f}")
    print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
    print(f"Сумарний спуск (м): {total_descent:.2f}")

    # густа сітка для аналізу сплайнів
    xx = np.linspace(distances[0], distances[-1], 500)
    yy_full = evaluate_spline(xx, distances, a_full, b_full, c_full, d_full)

    # градієнт
    grad_full = np.gradient(yy_full, xx) * 100.0
    print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
    print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
    print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

    # фізика: витрати енергії
    mass = 80.0
    g = 9.81
    energy = mass * g * total_ascent
    print(f"Механічна робота (Дж): {energy:.2f}")
    print(f"Механічна робота (кДж): {energy / 1000.0:.2f}")
    print(f"Енергія (ккал): {energy / 4184.0:.2f}")

    # сплайни для 10 та 15 вузлів
    idx_10 = np.round(np.linspace(0, num_points - 1, 10)).astype(int)
    idx_15 = np.round(np.linspace(0, num_points - 1, 15)).astype(int)

    a_10, b_10, c_10, d_10, _, _, _, _ = build_cubic_spline(distances[idx_10], elevations[idx_10])
    a_15, b_15, c_15, d_15, _, _, _, _ = build_cubic_spline(distances[idx_15], elevations[idx_15])

    yy_10 = evaluate_spline(xx, distances[idx_10], a_10, b_10, c_10, d_10)
    yy_15 = evaluate_spline(xx, distances[idx_15], a_15, b_15, c_15, d_15)

    # похибки відносно повного сплайна
    err_10 = np.abs(yy_10 - yy_full)
    err_15 = np.abs(yy_15 - yy_full)

    print(f"\nМаксимальна похибка сплайну з 10 вузлами: {np.max(err_10):.2f} м")
    print(f"Максимальна похибка сплайну з 15 вузлами: {np.max(err_15):.2f} м")

    # інтерактивні графіки
    plt.figure(figsize=(10, 9))

    plt.subplot(3, 1, 1)
    plt.plot(distances, elevations, 'ro', markersize=5, label='GPS точки (21 вузол)')
    plt.plot(xx, yy_full, 'b-', label='Кубічний сплайн (21 вузол)')
    plt.plot(xx, yy_10, 'g--', label='Кубічний сплайн (10 вузлів)')
    plt.plot(xx, yy_15, 'm-.', label='Кубічний сплайн (15 вузлів)')
    plt.title('Профіль висоти маршруту Заросляк - Говерла')
    plt.xlabel('Відстань (м)')
    plt.ylabel('Висота (м)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(xx, grad_full, 'darkorange', label='Градієнт крутизни (%)')
    plt.axhline(15, color='red', linestyle='--', label='Крутизна > 15%')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
    plt.title('Градієнтний профіль маршруту (% нахилу)')
    plt.xlabel('Відстань (м)')
    plt.ylabel('Нахил (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(xx, err_10, 'g-', label='Похибка (10 вузлів vs 21 вузол)')
    plt.plot(xx, err_15, 'm-', label='Похибка (15 вузлів vs 21 вузол)')
    plt.title('Абсолютна похибка інтерполяції кубічними сплайнами')
    plt.xlabel('Відстань (м)')
    plt.ylabel('Похибка (м)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
