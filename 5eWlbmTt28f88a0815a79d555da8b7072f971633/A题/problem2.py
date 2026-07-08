"""
2022 数模 A 题 · 问题 2 — 最优阻尼系数搜索
═══════════════════════════════════════════════════════════
策略:
  线性:   对数网格粗搜(40点) + golden-section 精化
  非线性: 连续二维搜索 — 粗网格找初值 + L-BFGS-B 精化 alpha
功率:   完整周期平均, 丢弃前若干周期瞬态
═══════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, minimize, differential_evolution
import pandas as pd
import time

# ═══════════════════════════════════════════════════════════
# 参数 (问题2 — 附件3第2行)
# ═══════════════════════════════════════════════════════════
m_b   = 4866.0       # 浮子质量 (kg)
r_b   = 1.0          # 浮子底面半径 (m)
m_z   = 2433.0       # 振子质量 (kg)
rho   = 1025.0       # 海水密度 (kg/m³)
g     = 9.8          # 重力加速度
k_s   = 80000.0      # 弹簧刚度 (N/m)

# 问题2 水动力参数 (附件3 "问题2" 行, 仅垂荡列)
omega = 2.2143        # 波浪圆频率 (s⁻¹)
m_add = 1165.992      # 垂荡附加质量 (kg)
C_w   = 167.8395      # 垂荡兴波阻尼系数 (N·s/m)
f_amp = 4890.0        # 垂荡激励力振幅 (N)

# 计算
K_h   = rho * g * np.pi * r_b**2   # 静水恢复力系数
T_p   = 2 * np.pi / omega           # 单周期 (s)

print("=" * 66)
print("  2022 A题 问题2 — 最优阻尼系数搜索")
print("=" * 66)
print(f"  omega={omega:.4f}  T={T_p:.3f}s  C_wave={C_w:.2f}")
print(f"  m_add={m_add:.3f}  f_amp={f_amp:.1f}  K_hydro={K_h:.1f}")
print(f"  决策变量: 线性 c∈[0,1e5] / 非线性 c∈[0,1e5], α∈[0,1]")

# ═══════════════════════════════════════════════════════════
# Simpson 积分
# ═══════════════════════════════════════════════════════════
def simpson(y, h):
    """对等间距数据做 Simpson 积分, 返回定积分值。"""
    y = np.asarray(y, dtype=float)
    n = len(y) - 1
    if n < 1:
        return 0.0

    tail = 0.0
    if n % 2 == 1:  # 奇数个区间不能全用 Simpson, 最后一段用梯形修正
        tail = (y[-1] + y[-2]) * h / 2
        y = y[:-1]
        n -= 1

    if n == 0:
        return tail

    # Simpson 核心
    s = y[0] + y[n] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2])
    return s * h / 3 + tail


# ═══════════════════════════════════════════════════════════
# ODE 系统
# ═══════════════════════════════════════════════════════════
def damping_force(c, alpha, rv):
    """
    PTO 阻尼力。

    alpha < 0 表示常量阻尼系数情形: F = c * v。
    alpha >= 0 表示题目中的非线性阻尼系数: c_d(v)=c*|v|^alpha,
    因而阻尼力 F = c*|v|^alpha*v。
    """
    if alpha < 0:
        return c * rv
    return c * abs(rv) ** alpha * rv


def damping_power(c, alpha, rv):
    """阻尼器瞬时输出功率。"""
    if alpha < 0:
        return c * rv ** 2
    return c * abs(rv) ** (alpha + 2)


def ode(t, x, c, alpha):
    """
    状态 [z_b, zb_dot, z_z, zz_dot].
    alpha < 0 时表示线性阻尼 (F=c·v_rel).
    """
    zb, zd, zz, zzd = x
    rv = zd - zzd                             # 相对速度
    fd = damping_force(c, alpha, rv)           # 阻尼力
    fP = k_s * (zb - zz) + fd                 # PTO总力
    fe = f_amp * np.cos(omega * t)            # 波浪激励力
    zb_dd = (fe - C_w * zd - K_h * zb - fP) / (m_b + m_add)
    zz_dd = fP / m_z
    return [zd, zb_dd, zzd, zz_dd]


# ═══════════════════════════════════════════════════════════
# 功率评估函数: 给定 (c, α) → P_avg
# ═══════════════════════════════════════════════════════════
def avg_power(c, alpha, n_total=40, n_drop=10, dt=0.2):
    """
    求解 ODE, 丢弃前 n_drop 个周期, Simpson 积分求稳态平均功率.
    alpha < 0 → 线性阻尼.
    """
    t_max = n_total * T_p
    t_eval = np.arange(0, t_max, dt)

    sol = solve_ivp(ode, (0, t_max), [0., 0., 0., 0.],
                    t_eval=t_eval, method='RK45',
                    args=(c, alpha), rtol=1e-6, atol=1e-8)
    if not sol.success:
        return -np.inf

    vr = sol.y[1] - sol.y[3]                            # 相对速度
    P_inst = damping_power(c, alpha, vr)                  # 瞬时功率

    n_skip = int(n_drop * T_p / dt)
    P_steady = P_inst[n_skip:]
    t_steady = sol.t[n_skip:]

    # Simpson 积分 → 时间平均
    integral = simpson(P_steady, dt)
    T_steady = t_steady[-1] - t_steady[0]
    return integral / T_steady


def avg_power_periodic(c, alpha, n_total=36, n_drop=18, samples_per_period=100,
                       rtol=1e-5, atol=1e-7):
    """
    用完整周期采样估计稳态平均功率，避免 dt 与周期不整除造成的截断偏差。

    优化阶段只采样丢弃瞬态后的完整周期；最终复核时可提高 n_total 和
    samples_per_period。c=0 时功率恒为 0，直接返回。
    """
    if c <= 0:
        return 0.0

    t_max = n_total * T_p
    t_start = n_drop * T_p
    n_eval = (n_total - n_drop) * samples_per_period + 1
    t_eval = np.linspace(t_start, t_max, n_eval)

    sol = solve_ivp(
        ode, (0, t_max), [0., 0., 0., 0.],
        t_eval=t_eval, method='RK45', args=(c, alpha),
        rtol=rtol, atol=atol, max_step=T_p / 25
    )
    if not sol.success or sol.y.shape[1] < 2:
        return -np.inf

    vr = sol.y[1] - sol.y[3]
    P_inst = damping_power(c, alpha, vr)
    return np.trapezoid(P_inst, sol.t) / (sol.t[-1] - sol.t[0])


def rhs_array(t, x, c, alpha):
    """优化阶段用的数组版 ODE 右端，避免 solve_ivp 在 alpha=0 边界反复缩步。"""
    zb, zd, zz, zzd = x
    rv = zd - zzd
    fd = damping_force(c, alpha, rv)
    fP = k_s * (zb - zz) + fd
    fe = f_amp * np.cos(omega * t)
    zb_dd = (fe - C_w * zd - K_h * zb - fP) / (m_b + m_add)
    zz_dd = fP / m_z
    return np.array([zd, zb_dd, zzd, zz_dd], dtype=float)


def rk4_power(c, alpha, n_total=40, n_drop=20, steps_per_period=120):
    """
    固定步长 RK4 估计稳态平均功率。

    这是优化阶段的快速目标函数。它牺牲少量自适应精度，换来在 alpha=0
    这类不光滑边界上的稳定运行。平均区间严格取完整周期。
    """
    if c <= 0:
        return 0.0

    dt = T_p / steps_per_period
    total_steps = int(n_total * steps_per_period)
    drop_steps = int(n_drop * steps_per_period)
    x = np.zeros(4, dtype=float)
    t = 0.0
    power_sum = 0.0
    count = 0

    for step in range(total_steps):
        k1 = rhs_array(t, x, c, alpha)
        k2 = rhs_array(t + 0.5 * dt, x + 0.5 * dt * k1, c, alpha)
        k3 = rhs_array(t + 0.5 * dt, x + 0.5 * dt * k2, c, alpha)
        k4 = rhs_array(t + dt, x + dt * k3, c, alpha)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt

        if step >= drop_steps:
            rv = x[1] - x[3]
            power_sum += damping_power(c, alpha, rv)
            count += 1

    return power_sum / count if count else -np.inf


def rk4_full_solve(c, alpha, n_periods=40, output_dt=0.2, steps_per_period=180):
    """
    固定步长 RK4 输出题目要求的时间序列。

    对 alpha=0 边界解，比自适应 solve_ivp 更可控。输出时刻仍为 0.2 s
    间隔，状态使用相邻 RK4 节点线性插值。
    """
    t_max = n_periods * T_p
    t_eval = np.arange(0, t_max, output_dt)
    dt = T_p / steps_per_period
    x = np.zeros(4, dtype=float)
    t = 0.0
    out = []
    next_i = 0

    while next_i < len(t_eval) and abs(t_eval[next_i]) < 1e-12:
        out.append(x.copy())
        next_i += 1

    while t < t_max and next_i < len(t_eval):
        t_prev = t
        x_prev = x.copy()

        h = min(dt, t_max - t)
        k1 = rhs_array(t, x, c, alpha)
        k2 = rhs_array(t + 0.5 * h, x + 0.5 * h * k1, c, alpha)
        k3 = rhs_array(t + 0.5 * h, x + 0.5 * h * k2, c, alpha)
        k4 = rhs_array(t + h, x + h * k3, c, alpha)
        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h

        while next_i < len(t_eval) and t_eval[next_i] <= t + 1e-12:
            w = (t_eval[next_i] - t_prev) / (t - t_prev)
            out.append(x_prev + w * (x - x_prev))
            next_i += 1

    y = np.array(out)
    z_b, zb_dot, z_z, zz_dot = y.T
    vr = zb_dot - zz_dot
    p_inst = damping_power(c, alpha, vr)

    n_skip = int(20 * T_p / output_dt)
    P_avg = np.trapezoid(p_inst[n_skip:], t_eval[n_skip:]) / (t_eval[-1] - t_eval[n_skip])

    df = pd.DataFrame({
        '时间 t (s)':          t_eval,
        '浮子位移 z_b (m)':     z_b,
        '浮子速度 zb_dot (m/s)': zb_dot,
        '振子位移 z_z (m)':     z_z,
        '振子速度 zz_dot (m/s)': zz_dot,
        '相对速度 v_rel (m/s)':  vr,
        '瞬时功率 P (W)':       p_inst
    })
    return df, P_avg


# ═══════════════════════════════════════════════════════════
# 1D 搜索: 对数网格 + golden-section (线性 & 剖面法共用)
# ═══════════════════════════════════════════════════════════
def search_1d(alpha, c_range=(1, 100000), n_grid=40, label="",
              n_total=40, n_drop=10, dt=0.2):
    """
    对固定 alpha, 一维搜索最优 c.
    返回: (c_opt, P_opt, c_grid, P_grid)
    """
    cg = np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), n_grid)
    pg = np.array([avg_power(c, alpha, n_total, n_drop, dt) for c in cg])
    ib = np.argmax(pg)
    c_best = cg[ib]

    # 精化
    lo, hi = max(c_best * 0.3, c_range[0]), min(c_best * 3, c_range[1])
    res = minimize_scalar(
        lambda c: -avg_power(c, alpha, n_total, n_drop, dt),
        bounds=(lo, hi), method='bounded',
        options={'xatol': 1.0, 'maxiter': 40}
    )
    c_opt, P_opt = res.x, -res.fun
    return c_opt, P_opt, cg, pg


def search_nonlinear_continuous(c_range=(1e-3, 100000), alpha_range=(0.0, 1.0),
                                n_c=15, n_alpha=11):
    """
    连续优化非线性阻尼参数。

    先在 (log10(c), alpha) 上做小规模粗网格，找到可靠初值；
    再用 L-BFGS-B 在连续区间内精化。返回最优 c、alpha、功率和粗搜记录。
    """
    log_bounds = (np.log10(c_range[0]), np.log10(c_range[1]))
    log_grid = np.linspace(log_bounds[0], log_bounds[1], n_c)
    alpha_grid = np.linspace(alpha_range[0], alpha_range[1], n_alpha)

    records = []
    best_x = None
    best_p = -np.inf

    for a in alpha_grid:
        for log_c in log_grid:
            c = 10 ** log_c
            p = avg_power_periodic(c, a)
            records.append((c, a, p))
            if p > best_p:
                best_p = p
                best_x = np.array([log_c, a])

    def objective(x):
        log_c, alpha = x
        c = 10 ** log_c
        p = avg_power_periodic(c, alpha)
        if not np.isfinite(p):
            return 1e30
        return -p

    res = minimize(
        objective, best_x, method='L-BFGS-B',
        bounds=[log_bounds, alpha_range],
        options={'maxiter': 45, 'ftol': 1e-5}
    )

    log_c_opt, alpha_opt = res.x
    c_opt = 10 ** log_c_opt
    P_opt = -res.fun
    return c_opt, alpha_opt, P_opt, records, res


def search_nonlinear_surface(c_range=(1e-2, 100000), alpha_range=(0.0, 1.0),
                             n_c=36, n_alpha=31):
    """
    非线性阻尼的粗扫描 + 下山法 + 差分进化交叉验证。

    变量采用 (log10(c), alpha)，功率用固定步长 RK4 估计。粗扫描先确认
    曲面形态和边界候选；随后 Powell 在粗搜最优点附近精化；最后用
    differential_evolution 做一次全局黑箱优化校验。
    """
    log_bounds = (np.log10(c_range[0]), np.log10(c_range[1]))
    log_grid = np.linspace(log_bounds[0], log_bounds[1], n_c)
    alpha_grid = np.linspace(alpha_range[0], alpha_range[1], n_alpha)

    records = []
    best = (-np.inf, None, None)

    for a in alpha_grid:
        row_best = (-np.inf, None)
        for log_c in log_grid:
            c = 10 ** log_c
            p_val = rk4_power(c, a, n_total=32, n_drop=16, steps_per_period=100)
            records.append((c, a, p_val))
            if p_val > best[0]:
                best = (p_val, c, a)
            if p_val > row_best[0]:
                row_best = (p_val, c)
        print(f"  粗扫 α={a:5.3f}: best P={row_best[0]:8.3f} W, c={row_best[1]:10.2f}")

    def objective(x, n_total=36, n_drop=18, steps_per_period=120):
        log_c, alpha = x
        if not (log_bounds[0] <= log_c <= log_bounds[1] and alpha_range[0] <= alpha <= alpha_range[1]):
            return 1e30
        c = 10 ** log_c
        p_val = rk4_power(c, alpha, n_total=n_total, n_drop=n_drop,
                          steps_per_period=steps_per_period)
        return 1e30 if not np.isfinite(p_val) else -p_val

    x0 = np.array([np.log10(best[1]), best[2]])

    powell = minimize(
        objective, x0, method='Powell',
        bounds=[log_bounds, alpha_range],
        options={'maxiter': 60, 'xtol': 1e-3, 'ftol': 1e-3}
    )

    de = differential_evolution(
        lambda x: objective(x, n_total=28, n_drop=14, steps_per_period=90),
        bounds=[log_bounds, alpha_range],
        maxiter=18, popsize=8, tol=1e-3, polish=False, seed=2022,
        updating='immediate', workers=1
    )

    candidates = [
        ('coarse', x0, best[0]),
        ('powell', powell.x, -powell.fun),
        ('de', de.x, -de.fun),
    ]

    refined = []
    for name, x, _ in candidates:
        c = 10 ** x[0]
        a = float(x[1])
        p_val = rk4_power(c, a, n_total=60, n_drop=30, steps_per_period=160)
        refined.append((p_val, c, a, name))

    P_opt, c_opt, a_opt, source = max(refined, key=lambda item: item[0])
    meta = {
        'records': records,
        'powell': powell,
        'de': de,
        'refined': refined,
        'source': source,
    }
    return c_opt, a_opt, P_opt, meta


# ═══════════════════════════════════════════════════════════
# 完整求解 (最优参数下输出时间序列)
# ═══════════════════════════════════════════════════════════
def full_solve(c, alpha, n_periods=40, dt=0.2, label=""):
    """最优 c 下的 40 周期完整结果, 返回 DataFrame 和稳态功率."""
    t_max = n_periods * T_p
    t_eval = np.arange(0, t_max, dt)

    sol = solve_ivp(ode, (0, t_max), [0., 0., 0., 0.],
                    t_eval=t_eval, method='RK45',
                    args=(c, alpha), rtol=1e-8, atol=1e-10)

    z_b, zb_dot, z_z, zz_dot = sol.y
    vr = zb_dot - zz_dot
    fd = damping_force(c, alpha, vr)
    P_inst = fd * vr

    # 稳态功率 (丢弃前10周期)
    n_skip = int(10 * T_p / dt)
    integral = simpson(P_inst[n_skip:], dt)
    T_steady = sol.t[-1] - sol.t[n_skip]
    P_avg = integral / T_steady

    df = pd.DataFrame({
        '时间 t (s)':          sol.t,
        '浮子位移 z_b (m)':     z_b,
        '浮子速度 zb_dot (m/s)': zb_dot,
        '振子位移 z_z (m)':     z_z,
        '振子速度 zz_dot (m/s)': zz_dot,
        '相对速度 v_rel (m/s)':  vr,
        '瞬时功率 P (W)':       P_inst
    })

    return df, P_avg


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0_total = time.time()

    # ─── 线性阻尼 ─────────────────────────────────────────
    print("\n" + "─" * 66)
    print("  情况1: 线性阻尼  对数网格 + golden-section")
    print("─" * 66)

    t0 = time.time()
    c1, P1, cg1, pg1 = search_1d(
        alpha=-1.0, c_range=(1, 100000), n_grid=40,
        label="线性", n_total=40, n_drop=10
    )
    t1 = time.time()

    # 精确评估
    P1_fine = avg_power(c1, -1.0, n_total=80, n_drop=20, dt=0.1)
    print(f"  粗搜最佳 c  ≈ {cg1[np.argmax(pg1)]:.1f}  N·s/m")
    print(f"  精化后  c   = {c1:.2f}  N·s/m")
    print(f"  最大功率    = {P1_fine:.2f}  W  (精确评估)")
    print(f"  c/C_wave     = {c1/C_w:.2f}")
    print(f"  用时: {t1-t0:.1f}s")

    # ─── 非线性阻尼: 连续二维优化 ───────────────────────────
    print("\n" + "─" * 66)
    print("  情况2: 非线性阻尼  粗扫描曲面 + Powell下山法 + 差分进化")
    print("─" * 66)

    t0 = time.time()
    c2_opt, a2_opt, P2_fine, nl_meta = search_nonlinear_surface(
        c_range=(1e-2, 100000), alpha_range=(0.0, 1.0),
        n_c=36, n_alpha=31
    )
    t1 = time.time()

    print(f"  粗扫评估点数: {len(nl_meta['records'])}")
    print(f"  Powell成功: {nl_meta['powell'].success}, 差分进化迭代: {nl_meta['de'].nit}")
    print(f"  最优来源: {nl_meta['source']}")
    print(f"  非线性优化用时: {t1-t0:.1f}s")
    print(f"\n  全局候选最优: c={c2_opt:.2f}  α={a2_opt:.4f}  P={P2_fine:.2f} W (RK4复核)")

    # ─── 完整求解 + Excel 输出 ─────────────────────────────
    print("\n" + "─" * 66)
    print("  输出时间序列 (40周期, Δt=0.2s)")
    print("─" * 66)

    df1, P1s = full_solve(c1, -1.0, label="线性")
    df1.to_excel('result2-1.xlsx', index=False)
    print(f"  result2-1.xlsx  — {len(df1)} 行, P_steady={P1s:.2f}W")

    df2, P2s = rk4_full_solve(c2_opt, a2_opt)
    df2.to_excel('result2-2.xlsx', index=False)
    print(f"  result2-2.xlsx  — {len(df2)} 行, P_steady={P2s:.2f}W")

    # ─── 对比 c=10000 ─────────────────────────────────────
    print("\n" + "─" * 66)
    print("  与基准 c=10000 对比")
    print("─" * 66)
    p10k_lin = avg_power(10000, -1.0)
    p10k_nl  = avg_power(10000, 0.5)
    print(f"  线性   c=10000 → {p10k_lin:.2f}W   最优 c={c1:.1f} → {P1_fine:.2f}W   +{(P1_fine/p10k_lin-1)*100:.1f}%")
    print(f"  非线性 c=10000 → {p10k_nl:.2f}W   最优 c={c2_opt:.1f},α={a2_opt:.1f} → {P2_fine:.2f}W   +{(P2_fine/p10k_nl-1)*100:.1f}%")

    # ─── 特定时刻输出 ─────────────────────────────────────
    print("\n" + "─" * 66)
    print("  特定时刻数据 (论文要求: 10s, 20s, 40s, 60s, 100s)")
    print("─" * 66)

    for name, df, c_val in [('线性', df1, c1), ('非线性', df2, c2_opt)]:
        print(f"\n  [{name}]  c={c_val:.1f}" +
              (f"  α={a2_opt:.2f}" if name == '非线性' else ""))
        hdr = f"  {'t(s)':>8} {'z_b(m)':>10} {'ż_b(m/s)':>12} {'z_z(m)':>10} {'ż_z(m/s)':>12} {'P(W)':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr)-2))
        for tt in [10, 20, 40, 60, 100]:
            i = (df['时间 t (s)'] - tt).abs().idxmin()
            r = df.iloc[i]
            print(f"  {tt:>8} {r['浮子位移 z_b (m)']:>10.4f} "
                  f"{r['浮子速度 zb_dot (m/s)']:>12.4f} "
                  f"{r['振子位移 z_z (m)']:>10.4f} "
                  f"{r['振子速度 zz_dot (m/s)']:>12.4f} "
                  f"{r['瞬时功率 P (W)']:>10.2f}")

    # ─── 非线性粗搜最佳点摘要 ─────────────────────────────
    print("\n" + "─" * 66)
    print("  非线性连续优化粗搜 Top 8")
    print("─" * 66)
    print(f"  {'alpha':>8} {'c':>12} {'P':>10}")
    for c, a, p in sorted(nl_meta['records'], key=lambda item: item[2], reverse=True)[:8]:
        print(f"  {a:>8.3f} {c:>12.1f} {p:>10.2f}")

    # ─── 完成 ─────────────────────────────────────────────
    t_total = time.time() - t0_total
    print(f"\n{'='*66}")
    print(f"  全部完成, 总用时 {t_total:.1f}s")
    print(f"  输出: result2-1.xlsx, result2-2.xlsx")
    print(f"{'='*66}")
