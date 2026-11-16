from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
import numpy as np


A, B = 0.0, 6.0
Y0 = 1.0
H_FIXED = 1e-2
ADAMS_H0 = 0.04
ADAMS_EPS = 1e-5
RK4_H0 = 0.05
RK4_EPS = 1e-8
SWEEP_STEPS = (0.2, 0.1, 0.05, 0.025, 0.01, 0.005)


def ode_rhs(x, y):
    return 2.0 * y * math.cos(2.0 * x)


def exact_solution(x):
    return math.exp(math.sin(2.0 * x))


@dataclass
class AdamsResult:
    x: np.ndarray
    y: np.ndarray
    x_pc: np.ndarray
    y_pr: np.ndarray
    y_cor: np.ndarray
    est_err: np.ndarray
    corr_iters: np.ndarray


@dataclass
class AdaptiveResult:
    x: np.ndarray
    y: np.ndarray
    h: np.ndarray
    est_err: np.ndarray
    rejects: int
    restarts: int = 0


def _validate_positive(value, name):
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} має бути додатним і скінченним")


def _validate_interval(x0, b):
    if not (math.isfinite(x0) and math.isfinite(b)) or b <= x0:
        raise ValueError("потрібно скінченні межі з x0 < b")


def _build_steps(x0, b, h):
    _validate_interval(x0, b)
    _validate_positive(h, "h")
    span = b - x0
    ratio = span / h
    nearest = round(ratio)
    if nearest >= 1 and abs(ratio - nearest) <= 1e-9:
        steps = [float(h)] * nearest
    else:
        n_full = int(span // h)
        steps = [float(h)] * n_full
        rem = span - n_full * h
        if rem > 1e-12 * h:
            steps.append(float(rem))
    return steps


def rk4_step(f, x, y, h):
    _validate_positive(h, "h")
    k1 = f(x, y)
    k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_solve(f, x0, y0, b, h):
    steps = _build_steps(x0, b, h)
    xs = [float(x0)]
    ys = [float(y0)]
    x, y = float(x0), float(y0)
    for idx, step in enumerate(steps):
        x_next = float(b) if idx == len(steps) - 1 else x + step
        y = rk4_step(f, x, y, x_next - x)
        x = x_next
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def rk4_runge_error(f, x_nodes, y_nodes, h):
    _validate_positive(h, "h")
    if len(x_nodes) != len(y_nodes) or len(x_nodes) < 2:
        raise ValueError("потрібно щонайменше два вузли однакової довжини")
    phis = []
    # оцінюєм локальну похибку рунге з кожного вузла сітки
    for i in range(len(x_nodes) - 1):
        xn = float(x_nodes[i])
        yn = float(y_nodes[i])
        step = float(x_nodes[i + 1] - xn)
        y_full = rk4_step(f, xn, yn, step)
        y_mid = rk4_step(f, xn, yn, 0.5 * step)
        y_half = rk4_step(f, xn + 0.5 * step, y_mid, 0.5 * step)
        phis.append((16.0 / 15.0) * (y_half - y_full))
    return np.array(phis, dtype=float)


def _adams2_step(f, x_next, yn, fn, f_prev, h, corr_eps, max_corr):
    y_pr = yn + 0.5 * h * (3.0 * fn - f_prev)
    y_cur = y_pr
    for it in range(1, max_corr + 1):
        y_next = yn + 0.5 * h * (f(x_next, y_cur) + fn)
        if not math.isfinite(y_next):
            raise ArithmeticError("нескінченне значення на етапі корекції")
        if abs(y_next - y_cur) <= corr_eps:
            est_err = -(y_next - y_pr) / 6.0
            return y_pr, y_next, est_err, it
        y_cur = y_next
    raise ArithmeticError("ітерації коректора не збігаються")


def adams2_pc(f, x0, y0, b, h, corr_eps=1e-12, max_corr=20):
    _validate_positive(corr_eps, "corr_eps")
    if not isinstance(max_corr, int) or isinstance(max_corr, bool) or max_corr < 1:
        raise ValueError("max_corr має бути натуральним числом")
    steps = _build_steps(x0, b, h)
    xs = [float(x0)]
    ys = [float(y0)]
    # перший крок рахуєм через рк4 як стартовий вузол для адамса
    x1 = float(b) if len(steps) == 1 else float(x0) + steps[0]
    y1 = rk4_step(f, float(x0), float(y0), x1 - float(x0))
    xs.append(x1)
    ys.append(y1)
    f_prev = f(float(x0), float(y0))
    fn = f(x1, y1)
    x_pc, y_pr_list, y_cor_list, est_list, iters_list = [], [], [], [], []
    for idx in range(1, len(steps)):
        step = steps[idx]
        x_next = float(b) if idx == len(steps) - 1 else xs[-1] + step
        if abs(step - steps[0]) > 1e-9 * steps[0]:
            y_next = rk4_step(f, xs[-1], ys[-1], x_next - xs[-1])
            xs.append(x_next)
            ys.append(y_next)
            continue
        y_pr, y_cor, est_err, it = _adams2_step(
            f, x_next, ys[-1], fn, f_prev, step, corr_eps, max_corr
        )
        xs.append(x_next)
        ys.append(y_cor)
        x_pc.append(x_next)
        y_pr_list.append(y_pr)
        y_cor_list.append(y_cor)
        est_list.append(est_err)
        iters_list.append(it)
        f_prev = fn
        fn = f(x_next, y_cor)
    return AdamsResult(
        x=np.array(xs, dtype=float),
        y=np.array(ys, dtype=float),
        x_pc=np.array(x_pc, dtype=float),
        y_pr=np.array(y_pr_list, dtype=float),
        y_cor=np.array(y_cor_list, dtype=float),
        est_err=np.array(est_list, dtype=float),
        corr_iters=np.array(iters_list, dtype=int),
    )


def adams2_adaptive(
    f, x0, y0, b, h0, eps, h_min=1e-5, h_max=0.2, corr_eps=1e-12, max_corr=20
):
    _validate_interval(x0, b)
    _validate_positive(h0, "h0")
    _validate_positive(eps, "eps")
    _validate_positive(h_min, "h_min")
    _validate_positive(h_max, "h_max")
    _validate_positive(corr_eps, "corr_eps")
    if h_max < h_min:
        raise ValueError("потрібно h_max >= h_min")
    x, y = float(x0), float(y0)
    h = min(max(float(h0), float(h_min)), float(h_max))
    xs, ys, hs, errs = [x], [y], [], []
    prev_f = None
    rejects, restarts = 0, 0
    while x < b - 1e-12:
        rem = b - x
        if rem <= h + 1e-12:
            y = rk4_step(f, x, y, rem)
            x = float(b)
            xs.append(x)
            ys.append(y)
            hs.append(rem)
            errs.append(0.0)
            break
        if prev_f is None:
            if rem < 2.0 * h - 1e-12:
                step = min(h, rem)
                y = rk4_step(f, x, y, step)
                x = float(b) if abs(x + step - b) <= 1e-12 else x + step
                xs.append(x)
                ys.append(y)
                hs.append(step)
                errs.append(0.0)
                continue
            # після зміни кроку перезапускаєм два вузли через рк4
            y1 = rk4_step(f, x, y, h)
            x1 = x + h
            f0, f1 = f(x, y), f(x1, y1)
            try:
                _, y2, est2, _ = _adams2_step(
                    f, x1 + h, y1, f1, f0, h, corr_eps, max_corr
                )
            except ArithmeticError:
                if h <= h_min + 1e-15:
                    raise
                h = max(0.5 * h, h_min)
                rejects += 1
                continue
            if abs(est2) > eps:
                if h <= h_min + 1e-15:
                    raise ArithmeticError("крок досяг мінімальної межі h_min")
                h = max(0.5 * h, h_min)
                rejects += 1
                continue
            x2 = float(b) if abs(x1 + h - b) <= 1e-12 else x1 + h
            restarts += 1
            xs.extend([x1, x2])
            ys.extend([y1, y2])
            hs.extend([h, h])
            errs.extend([0.0, abs(est2)])
            prev_f = f1
            x, y = x2, y2
            if abs(est2) < eps / 8.0 and 2.0 * h <= h_max + 1e-15:
                h = min(2.0 * h, h_max)
                prev_f = None
            continue
        fn = f(x, y)
        try:
            _, y_next, est, _ = _adams2_step(
                f, x + h, y, fn, prev_f, h, corr_eps, max_corr
            )
        except ArithmeticError:
            if h <= h_min + 1e-15:
                raise
            h = max(0.5 * h, h_min)
            prev_f = None
            rejects += 1
            continue
        if abs(est) > eps:
            if h <= h_min + 1e-15:
                raise ArithmeticError("крок досяг мінімальної межі h_min")
            h = max(0.5 * h, h_min)
            prev_f = None
            rejects += 1
            continue
        x_next = float(b) if abs(x + h - b) <= 1e-12 else x + h
        xs.append(x_next)
        ys.append(y_next)
        hs.append(h)
        errs.append(abs(est))
        prev_f = fn
        x, y = x_next, y_next
        if abs(est) < eps / 8.0 and 2.0 * h <= h_max + 1e-15:
            h = min(2.0 * h, h_max)
            prev_f = None
    return AdaptiveResult(
        x=np.array(xs, dtype=float),
        y=np.array(ys, dtype=float),
        h=np.array(hs, dtype=float),
        est_err=np.array(errs, dtype=float),
        rejects=rejects,
        restarts=restarts,
    )


def rk4_adaptive(f, x0, y0, b, h0=0.05, eps=1e-8, h_min=1e-6, h_max=0.25):
    _validate_interval(x0, b)
    _validate_positive(h0, "h0")
    _validate_positive(eps, "eps")
    _validate_positive(h_min, "h_min")
    _validate_positive(h_max, "h_max")
    if h_max < h_min:
        raise ValueError("потрібно h_max >= h_min")
    x, y = float(x0), float(y0)
    h = min(max(float(h0), float(h_min)), float(h_max))
    xs, ys, hs, errs = [x], [y], [], []
    rejects = 0
    while x < b - 1e-12:
        step = min(h, b - x)
        yh = rk4_step(f, x, y, step)
        ym = rk4_step(f, x, y, 0.5 * step)
        yh2 = rk4_step(f, x + 0.5 * step, ym, 0.5 * step)
        err = abs((16.0 / 15.0) * (yh2 - yh))
        if err > eps:
            if step <= h_min + 1e-15 or h <= h_min + 1e-15:
                raise ArithmeticError("крок досяг мінімальної межі h_min")
            h = max(0.5 * step, h_min)
            rejects += 1
            continue
        x = float(b) if abs(x + step - b) <= 1e-12 else x + step
        y = yh2
        xs.append(x)
        ys.append(y)
        hs.append(step)
        errs.append(err)
        if err < eps / 32.0 and abs(step - h) <= 1e-12 and 2.0 * h <= h_max + 1e-15:
            h = min(2.0 * h, h_max)
    return AdaptiveResult(
        x=np.array(xs, dtype=float),
        y=np.array(ys, dtype=float),
        h=np.array(hs, dtype=float),
        est_err=np.array(errs, dtype=float),
        rejects=rejects,
    )


def exact_error(x_nodes, y_nodes):
    exact = np.array([exact_solution(x) for x in x_nodes], dtype=float)
    return y_nodes - exact, exact


def experiment():
    adams_fixed = adams2_pc(ode_rhs, A, Y0, B, H_FIXED)
    adams_phi, adams_exact = exact_error(adams_fixed.x, adams_fixed.y)
    adams_pc_diff = adams_fixed.y_cor - adams_fixed.y_pr
    adams_max_est = float(np.max(np.abs(adams_fixed.est_err)))
    adams_optimal = (ADAMS_EPS / 8.0) <= adams_max_est <= ADAMS_EPS
    adams_adapt = adams2_adaptive(ode_rhs, A, Y0, B, ADAMS_H0, ADAMS_EPS)
    adams_adapt_phi, _ = exact_error(adams_adapt.x, adams_adapt.y)

    rk4_x, rk4_y = rk4_solve(ode_rhs, A, Y0, B, H_FIXED)
    rk4_phi, rk4_exact = exact_error(rk4_x, rk4_y)
    rk4_runge_phi = rk4_runge_error(ode_rhs, rk4_x, rk4_y, H_FIXED)
    rk4_max_runge = float(np.max(np.abs(rk4_runge_phi)))
    rk4_optimal = (RK4_EPS / 32.0) <= rk4_max_runge <= RK4_EPS
    rk4_h_req = H_FIXED * (RK4_EPS / rk4_max_runge) ** (1.0 / 5.0)

    rk4_sweep = []
    for h in SWEEP_STEPS:
        sx, sy = rk4_solve(ode_rhs, A, Y0, B, h)
        sphi, _ = exact_error(sx, sy)
        max_e = float(np.max(np.abs(sphi)))
        order = None
        if rk4_sweep:
            prev_h, _, prev_e, _ = rk4_sweep[-1]
            order = math.log(prev_e / max_e) / math.log(prev_h / h)
        rk4_sweep.append((h, len(sx) - 1, max_e, order))

    rk4_adapt = rk4_adaptive(ode_rhs, A, Y0, B, RK4_H0, RK4_EPS)
    rk4_adapt_phi, _ = exact_error(rk4_adapt.x, rk4_adapt.y)

    return dict(
        adams_fixed=adams_fixed,
        adams_exact=adams_exact,
        adams_phi=adams_phi,
        adams_pc_diff=adams_pc_diff,
        adams_max_est=adams_max_est,
        adams_optimal=adams_optimal,
        adams_adapt=adams_adapt,
        adams_adapt_phi=adams_adapt_phi,
        rk4_x=rk4_x,
        rk4_y=rk4_y,
        rk4_exact=rk4_exact,
        rk4_phi=rk4_phi,
        rk4_runge_phi=rk4_runge_phi,
        rk4_max_runge=rk4_max_runge,
        rk4_optimal=rk4_optimal,
        rk4_h_req=rk4_h_req,
        rk4_sweep=rk4_sweep,
        rk4_adapt=rk4_adapt,
        rk4_adapt_phi=rk4_adapt_phi,
    )


def print_results(result):
    af = result["adams_fixed"]
    aa = result["adams_adapt"]
    ra = result["rk4_adapt"]
    print(f"[a, b] = [{A:g}, {B:g}]; y(0) = {Y0:g}; h = {H_FIXED:g}")
    print(f"adams2_pc: steps = {len(af.x) - 1}; "
          f"max |y - y_ex| = {np.max(np.abs(result['adams_phi'])):.6e}")
    print(f"adams2_pc: max |-(y_cor-y_pr)/6| = {result['adams_max_est']:.6e}; "
          f"max |y_cor-y_pr| = {np.max(np.abs(result['adams_pc_diff'])):.6e}")
    print(f"adams2_pc: corr iters avg = {np.mean(af.corr_iters):.2f}; "
          f"max = {np.max(af.corr_iters)}; optimal = {result['adams_optimal']}")
    print(f"adams2_adaptive (h0={ADAMS_H0:g}, eps={ADAMS_EPS:.0e}): "
          f"steps = {len(aa.h)}; rejects = {aa.rejects}; restarts = {aa.restarts}")
    print(f"  h in [{np.min(aa.h):.4g}, {np.max(aa.h):.4g}]; "
          f"max |y - y_ex| = {np.max(np.abs(result['adams_adapt_phi'])):.6e}")
    print(f"rk4_fixed: steps = {len(result['rk4_x']) - 1}; "
          f"max |y - y_ex| = {np.max(np.abs(result['rk4_phi'])):.6e}")
    print(f"rk4_runge: max |phi_runge| = {result['rk4_max_runge']:.6e}; "
          f"optimal (eps={RK4_EPS:.0e}) = {result['rk4_optimal']}")
    print(f"  h_req (eps={RK4_EPS:.0e}) = {result['rk4_h_req']:.6e}")
    print(f"{'h':>8} {'steps':>7} {'max |y - y_ex|':>16} {'order':>9}")
    for h, steps, max_e, order in result["rk4_sweep"]:
        ord_str = "-" if order is None else f"{order:.4f}"
        print(f"{h:8.4f} {steps:7d} {max_e:16.6e} {ord_str:>9}")
    print(f"rk4_adaptive (h0={RK4_H0:g}, eps={RK4_EPS:.0e}): "
          f"steps = {len(ra.h)}; rejects = {ra.rejects}")
    print(f"  h in [{np.min(ra.h):.4g}, {np.max(ra.h):.4g}]; "
          f"max |y - y_ex| = {np.max(np.abs(result['rk4_adapt_phi'])):.6e}")


def make_plots(result):
    af = result["adams_fixed"]
    aa = result["adams_adapt"]
    ra = result["rk4_adapt"]

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")
    x_dense = np.linspace(A, B, 601)
    y_dense = np.array([exact_solution(x) for x in x_dense])
    ax1.plot(x_dense, y_dense, "k-", linewidth=1.6, label="точний y(x) = exp(sin(2x))")
    ax1.plot(aa.x[::12], aa.y[::12], "o", markersize=4, label="Адамс-2 (адапт.)")
    ax1.plot(ra.x[::6], ra.y[::6], "x", markersize=5, label="РК4 (адапт.)")
    ax1.set(xlabel="x", ylabel="y(x)", title="Точний і чисельні розв'язки задачі Коші")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.semilogy(af.x[1:], np.abs(result["adams_phi"][1:]), label="Адамс-2 (h = 0.01)")
    ax2.semilogy(result["rk4_x"][1:], np.abs(result["rk4_phi"][1:]), label="РК4 (h = 0.01)")
    ax2.set(xlabel="x", ylabel="|y_n - y(x_n)|", title="Порівняння похибок при h = 0.01")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")
    ax3.plot(af.x, result["adams_phi"], color="#1f77b4", linewidth=1.3)
    ax3.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax3.set(xlabel="x", ylabel="phi_n = y_n - y(x_n)",
            title="Адамс-2 (h = 0.01): точна похибка phi_n")
    ax3.grid(alpha=0.3)

    ax4.semilogy(af.x_pc, np.maximum(np.abs(result["adams_pc_diff"]), 1e-16),
                 color="#ff7f0e", linewidth=1.2, label="|y_cor - y_pr|")
    ax4.semilogy(af.x_pc, np.maximum(np.abs(af.est_err), 1e-16),
                 color="#2ca02c", linewidth=1.2, label="|-(1/6)(y_cor - y_pr)|")
    ax4.axhline(ADAMS_EPS, color="red", linestyle="--", linewidth=1.0,
                label=f"eps = {ADAMS_EPS:.0e}")
    ax4.axhline(ADAMS_EPS / 8.0, color="purple", linestyle=":", linewidth=1.1,
                label=f"eps/8 = {ADAMS_EPS / 8:.2e}")
    ax4.set(xlabel="x", ylabel="Оцінка похибки",
            title="Адамс-2 (h = 0.01): оцінка похибки")
    ax4.grid(True, which="both", alpha=0.3)
    ax4.legend(fontsize=7.5)

    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")
    ax5.step(aa.x[:-1], aa.h, where="post", color="#1f77b4", linewidth=1.5)
    ax5.set(xlabel="x", ylabel="Крок h(x)",
            title=f"Адаптивний Адамс-2: залежність кроку h(x) (eps={ADAMS_EPS:.0e})")
    ax5.grid(alpha=0.3)

    mask_a = aa.est_err > 0.0
    ax6.semilogy(aa.x[1:][mask_a], aa.est_err[mask_a], ".", markersize=4,
                 label="оцінка |est_err|")
    ax6.semilogy(aa.x[1:], np.maximum(np.abs(result["adams_adapt_phi"][1:]), 1e-16),
                 linewidth=1.1, label="фактична |y_n - y(x_n)|")
    ax6.axhline(ADAMS_EPS, color="red", linestyle="--", linewidth=1.0, label="eps")
    ax6.axhline(ADAMS_EPS / 8.0, color="purple", linestyle=":", linewidth=1.1, label="eps/8")
    ax6.set(xlabel="x", ylabel="Похибка",
            title="Адаптивний Адамс-2: похибки")
    ax6.grid(True, which="both", alpha=0.3)
    ax6.legend(fontsize=8)

    fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")
    ax7.plot(result["rk4_x"], result["rk4_phi"], color="#1f77b4", linewidth=1.3)
    ax7.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax7.set(xlabel="x", ylabel="phi_n = y_n - y(x_n)",
            title="РК4 (h = 0.01): похибка за точним розв'язком")
    ax7.grid(alpha=0.3)

    ax8.plot(result["rk4_x"][1:], result["rk4_runge_phi"], color="#d62728", linewidth=1.3)
    ax8.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax8.set(xlabel="x", ylabel="phi_n = (16/15)(y^(h/2) - y^h)",
            title="РК4 (h = 0.01): похибка за методом Рунге")
    ax8.grid(alpha=0.3)

    fig5, (ax9, ax10) = plt.subplots(1, 2, figsize=(10.2, 4.0), layout="constrained")
    hs = np.array([row[0] for row in result["rk4_sweep"]])
    errs = np.array([row[2] for row in result["rk4_sweep"]])
    ref = errs[0] * (hs / hs[0]) ** 4
    ax9.loglog(hs, errs, "o-", linewidth=1.5, label="max |y_n - y(x_n)|")
    ax9.loglog(hs, ref, "--", color="gray", label="лінія O(h^4)")
    ax9.set(xlabel="Крок h", ylabel="Максимальна похибка",
            title="РК4: залежність похибки від кроку h")
    ax9.grid(True, which="both", alpha=0.3)
    ax9.legend(fontsize=8)

    ax10.step(ra.x[:-1], ra.h, where="post", color="#2ca02c", linewidth=1.5)
    ax10.set(xlabel="x", ylabel="Крок h(x)",
             title=f"Адаптивний РК4: залежність кроку h(x) (eps={RK4_EPS:.0e})")
    ax10.grid(alpha=0.3)

    return [fig1, fig2, fig3, fig4, fig5]


def main():
    result = experiment()
    print_results(result)
    make_plots(result)
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
