from dataclasses import dataclass
import math
from pathlib import Path

from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
TRAJECTORY_FILE = BASE_DIR / "trajectory.txt"

X0_SYSTEM = np.array([0.2, 0.2], dtype=float)
DELTA0_SYSTEM = np.array([0.2, 0.2], dtype=float)
X0_ROSEN = np.array([-1.2, 0.0], dtype=float)
DELTA0_ROSEN = np.array([0.2, 0.2], dtype=float)
Q = 2.0
P = 2.0
EPS1 = 1e-7
EPS2 = 1e-12


@dataclass
class HookeJeevesResult:
    x: np.ndarray
    value: float
    trajectory: np.ndarray
    values: np.ndarray
    steps: int
    evaluations: int
    pattern_moves: int
    converged: bool


def system_equations(x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        x1 * x1 + x2 * x2 - 4.0,
        x2 - math.exp(x1) + 1.0,
    ], dtype=float)


def system_objective(x):
    residuals = system_equations(x)
    return float(np.dot(residuals, residuals))


def rosenbrock(x):
    x1, x2 = float(x[0]), float(x[1])
    return 100.0 * (x1 * x1 - x2) ** 2 + (x1 - 1.0) ** 2


def _validate_inputs(x0, delta0, q, p, eps1, eps2, max_iter):
    x0_arr = np.asarray(x0, dtype=float)
    delta_arr = np.asarray(delta0, dtype=float)
    if x0_arr.ndim != 1 or x0_arr.size == 0:
        raise ValueError("початкова точка має бути непорожнім 1d вектором")
    if delta_arr.shape != x0_arr.shape:
        raise ValueError("розмірність кроку має збігатися з x0")
    if not (np.all(np.isfinite(x0_arr)) and np.all(np.isfinite(delta_arr))):
        raise ValueError("координати та кроки мають бути скінченними")
    if np.any(delta_arr <= 0):
        raise ValueError("початкові кроки мають бути додатними")
    if not math.isfinite(q) or q <= 1.0:
        raise ValueError("коефіцієнт q має бути більшим за 1")
    if not math.isfinite(p) or p <= 0.0:
        raise ValueError("коефіцієнт p має бути додатним")
    if not math.isfinite(eps1) or eps1 <= 0.0:
        raise ValueError("eps1 має бути додатним і скінченним")
    if not math.isfinite(eps2) or eps2 <= 0.0:
        raise ValueError("eps2 має бути додатним і скінченним")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter має бути цілим числом >= 1")
    return x0_arr, delta_arr


def _probe_axis(eval_fn, point, f_cur, axis, step):
    for sign in (1.0, -1.0):
        cand = point.copy()
        cand[axis] = point[axis] + sign * step
        fc = eval_fn(cand)
        if fc < f_cur:
            return cand, fc, True
    return point, f_cur, False


def _explore(eval_fn, base, f_base, delta, q, eps1, shrink):
    point = base.copy()
    f_cur = f_base
    step = delta.copy()
    for i in range(len(point)):
        if not shrink:
            point, f_cur, _ = _probe_axis(eval_fn, point, f_cur, i, step[i])
            continue
        while step[i] >= eps1:
            point, f_cur, improved = _probe_axis(
                eval_fn, point, f_cur, i, step[i]
            )
            if improved:
                break
            step[i] /= q
    return point, f_cur, step


def hooke_jeeves(
    func, x0, delta0, q=2.0, p=2.0, eps1=1e-7, eps2=1e-12, max_iter=5000
):
    x0_arr, delta_arr = _validate_inputs(
        x0, delta0, q, p, eps1, eps2, max_iter
    )
    evaluations = 0

    def eval_fn(pt):
        nonlocal evaluations
        evaluations += 1
        try:
            val = float(func(pt))
        except (OverflowError, ValueError):
            return math.inf
        return val if math.isfinite(val) else math.inf

    x_base = x0_arr.copy()
    f_base = eval_fn(x_base)
    if not math.isfinite(f_base):
        raise ValueError("функція в початковій точці має бути скінченною")
    delta = delta_arr.copy()
    trajectory = [x_base.copy()]
    values = [f_base]
    pattern_moves = 0
    converged = False
    iterations = 0

    while iterations < max_iter:
        iterations += 1
        x_curr, f_curr, delta = _explore(
            eval_fn, x_base, f_base, delta, q, eps1, shrink=True
        )
        if np.array_equal(x_curr, x_base):
            converged = True
            break
        trajectory.append(x_curr.copy())
        values.append(f_curr)
        if np.linalg.norm(delta) < eps1 and abs(f_curr - f_base) < eps2:
            converged = True
            break

        # пошук по зразку без зменшення поточного кроку
        while iterations < max_iter:
            iterations += 1
            x_pat = x_curr + p * (x_curr - x_base)
            f_pat = eval_fn(x_pat)
            if not math.isfinite(f_pat):
                x_base, f_base = x_curr, f_curr
                break
            x_next, f_next, _ = _explore(
                eval_fn, x_pat, f_pat, delta, q, eps1, shrink=False
            )
            if f_next < f_curr:
                pattern_moves += 1
                x_base, f_base = x_curr, f_curr
                x_curr, f_curr = x_next, f_next
                trajectory.append(x_curr.copy())
                values.append(f_curr)
            else:
                x_base, f_base = x_curr, f_curr
                break

    traj_arr = np.array(trajectory, dtype=float)
    val_arr = np.array(values, dtype=float)
    return HookeJeevesResult(
        x=traj_arr[-1].copy(),
        value=float(val_arr[-1]),
        trajectory=traj_arr,
        values=val_arr,
        steps=len(traj_arr) - 1,
        evaluations=evaluations,
        pattern_moves=pattern_moves,
        converged=converged,
    )


def save_trajectory(trajectory, values, filepath=TRAJECTORY_FILE):
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{'step':>4} {'x1':>18} {'x2':>18} {'Phi':>22}"]
    for step, (pt, val) in enumerate(zip(trajectory, values)):
        lines.append(f"{step:4d} {pt[0]:18.12f} {pt[1]:18.12f} {val:22.14e}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def experiment():
    rosen_res = hooke_jeeves(
        rosenbrock, X0_ROSEN, DELTA0_ROSEN,
        q=Q, p=P, eps1=EPS1, eps2=EPS2,
    )
    sys_res = hooke_jeeves(
        system_objective, X0_SYSTEM, DELTA0_SYSTEM,
        q=Q, p=P, eps1=EPS1, eps2=EPS2,
    )
    residuals = system_equations(sys_res.x)
    return dict(
        rosenbrock=rosen_res,
        system=sys_res,
        residuals=residuals,
    )


def print_results(result):
    rosen_res = result["rosenbrock"]
    sys_res = result["system"]
    res_vec = result["residuals"]
    print(f"q = {Q:g}; p = {P:g}; eps1 = {EPS1:.0e}; eps2 = {EPS2:.0e}")
    print(
        f"Rosenbrock: x* = ({rosen_res.x[0]:.10f}, {rosen_res.x[1]:.10f}); "
        f"f(x*) = {rosen_res.value:.6e}; steps = {rosen_res.steps}; "
        f"pattern_moves = {rosen_res.pattern_moves}; evals = {rosen_res.evaluations}"
    )
    print(
        f"System X0 = ({X0_SYSTEM[0]:g}, {X0_SYSTEM[1]:g}); "
        f"delta0 = ({DELTA0_SYSTEM[0]:g}, {DELTA0_SYSTEM[1]:g})"
    )
    print(
        f"System: x* = ({sys_res.x[0]:.12f}, {sys_res.x[1]:.12f}); "
        f"Phi(x*) = {sys_res.value:.6e}"
    )
    print(f"f1(x*) = {res_vec[0]:.6e}; f2(x*) = {res_vec[1]:.6e}")
    print(
        f"trajectory_points = {len(sys_res.trajectory)}; "
        f"steps = {sys_res.steps}; evals = {sys_res.evaluations}"
    )
    print(f"{'step':>4} {'x1':>16} {'x2':>16} {'Phi(x1, x2)':>16}")
    for step, (pt, val) in enumerate(zip(sys_res.trajectory, sys_res.values)):
        print(f"{step:4d} {pt[0]:16.10f} {pt[1]:16.10f} {val:16.6e}")


def make_plots(result):
    sys_res = result["system"]
    rosen_res = result["rosenbrock"]
    x_star = sys_res.x

    # графік кривих системи для вибору початкового наближення
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    cx, cy = 2.0 * np.cos(theta), 2.0 * np.sin(theta)
    ex1 = np.linspace(-2.4, 1.35, 400)
    ey1 = np.exp(ex1) - 1.0

    for ax in axes1:
        ax.plot(cx, cy, label="x1^2 + x2^2 = 4", linewidth=1.8)
        ax.plot(ex1, ey1, label="x2 = exp(x1) - 1", linewidth=1.8)
        ax.scatter(
            X0_SYSTEM[0], X0_SYSTEM[1], color="#2ca02c", s=65,
            marker="s", zorder=5,
            label=f"X^(0) = ({X0_SYSTEM[0]:g}, {X0_SYSTEM[1]:g})"
        )
        ax.scatter(
            x_star[0], x_star[1], color="red", s=85, marker="*",
            zorder=6, label=f"X* = ({x_star[0]:.4f}, {x_star[1]:.4f})"
        )
        ax.grid(alpha=0.3)

    axes1[0].set(
        xlim=(-2.4, 2.4), ylim=(-2.4, 2.4), aspect="equal",
        xlabel="x1", ylabel="x2", title="Криві рівнянь системи"
    )
    axes1[0].legend(fontsize=8, loc="upper left")
    axes1[1].set(
        xlim=(0.0, 1.45), ylim=(0.0, 2.1), aspect="equal",
        xlabel="x1", ylabel="x2", title="Перша чверть біля розв'язку"
    )
    axes1[1].legend(fontsize=8, loc="upper left")

    # лінії рівня цільової функції системи та траєкторія спуску
    fig2, ax2 = plt.subplots(figsize=(8.2, 5.0), layout="constrained")
    gx = np.linspace(0.05, 1.50, 260)
    gy = np.linspace(0.05, 1.95, 260)
    mx, my = np.meshgrid(gx, gy)
    mz = (mx * mx + my * my - 4.0) ** 2 + (my - np.exp(mx) + 1.0) ** 2
    mz_clipped = np.clip(mz, 1e-13, None)
    levels = np.logspace(-12, 1.3, 28)
    cf = ax2.contourf(
        mx, my, mz_clipped, levels=levels,
        norm=LogNorm(vmin=1e-12, vmax=10.0 ** 1.3),
        cmap="viridis", alpha=0.88, extend="both"
    )
    ax2.contour(mx, my, mz_clipped, levels=levels[::2], colors="white",
                linewidths=0.45, alpha=0.55)
    cbar = fig2.colorbar(cf, ax=ax2, label="Phi(x1, x2)")
    cbar.locator = LogLocator(base=10.0, numticks=8)
    cbar.update_ticks()
    traj = sys_res.trajectory
    ax2.plot(traj[:, 0], traj[:, 1], "o-", color="#ff7f0e",
             linewidth=1.8, markersize=4.5, label="Траєкторія Хука-Дживса")
    ax2.scatter(X0_SYSTEM[0], X0_SYSTEM[1], color="#2ca02c", s=70,
                marker="s", zorder=6, label="Старт X^(0)")
    ax2.scatter(x_star[0], x_star[1], color="red", s=100,
                marker="*", zorder=7, label="Розв'язок X*")
    ax2.set(xlabel="x1", ylabel="x2",
            title="Ізолінії Phi(x1, x2) та траєкторія спуску")
    ax2.legend(loc="lower left")

    # збіжність цільової функції та тест на функції розенброка
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    sys_steps = np.arange(len(sys_res.values))
    axes3[0].semilogy(sys_steps, np.clip(sys_res.values, 1e-16, None), "o-")
    axes3[0].set(xlabel="Крок траєкторії k", ylabel="Phi(X_k)",
                 title="Збіжність Phi(X_k) для системи")
    axes3[0].grid(True, which="both", alpha=0.3)

    rx = np.linspace(-1.5, 1.4, 240)
    ry = np.linspace(-0.4, 1.5, 240)
    rmx, rmy = np.meshgrid(rx, ry)
    rmz = np.clip(100.0 * (rmx * rmx - rmy) ** 2 + (rmx - 1.0) ** 2, 1e-7, None)
    rlevels = np.logspace(-6, 3.5, 22)
    axes3[1].contourf(
        rmx, rmy, rmz, levels=rlevels,
        norm=LogNorm(vmin=1e-6, vmax=10.0 ** 3.5),
        cmap="viridis", alpha=0.88, extend="both"
    )
    rtraj = rosen_res.trajectory
    axes3[1].plot(rtraj[:, 0], rtraj[:, 1], ".-", color="#ff7f0e",
                  linewidth=1.5, markersize=4.0, label="Траєкторія")
    axes3[1].scatter(X0_ROSEN[0], X0_ROSEN[1], color="#2ca02c", s=55,
                     marker="s", zorder=6, label="Старт (-1.2, 0)")
    axes3[1].scatter(1.0, 1.0, color="red", s=90, marker="*",
                     zorder=7, label="Мінімум (1, 1)")
    axes3[1].set(xlabel="x1", ylabel="x2",
                 title="Тест на функції Розенброка")
    axes3[1].legend(fontsize=8, loc="upper left")
    return [fig1, fig2, fig3]


def main():
    result = experiment()
    save_trajectory(
        result["system"].trajectory,
        result["system"].values,
        TRAJECTORY_FILE,
    )
    print_results(result)
    make_plots(result)
    plt.show()


if __name__ == "__main__":
    main()
