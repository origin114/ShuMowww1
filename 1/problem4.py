"""
2022 数模 A 题 · 问题 4

基于问题 3 的四自由度非线性几何耦合模型，同时优化:
  c_linear:  垂荡线性阻尼系数
  c_rotary:  纵摇旋转阻尼系数

目标函数为稳态平均输出功率:
  P = mean(c_linear * zz_dot^2 + c_rotary * theta_z_dot^2)

求解流程:
  1. 二维粗网格扫描，判断峰区；
  2. 以粗扫描最优点为初值，用 Powell 无导数方法局部精修；
  3. 对最优参数做高精度长时间复算，并导出结果。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# ============================================================
# 参数定义
# ============================================================

# 附件 4: 结构与环境参数
m_b = 4866.0
r_b = 1.0
h_cyl = 3.0
h_cone = 0.8
m_z = 2433.0
r_z = 0.5
h_z = 0.5
rho = 1025.0
g = 9.8
k_linear = 80000.0
k_torsion = 250000.0
K_pitch = 8890.7
l0 = 0.5

# 附件 3: 问题 4 水动力参数
omega = 1.9806
m_add = 1091.099
J_add = 7142.493
C_heave = 528.5018
C_pitch = 1655.909
f_amp = 1760.0
L_amp = 2140.0

# 派生参数
K_heave = rho * g * np.pi * r_b**2
T_period = 2 * np.pi / omega
l_eq = l0 - m_z * g / k_linear


def estimate_float_pitch_inertia_aproblem() -> float:
    """AProblem 附录 B.3.1 使用的浮子纵摇转动惯量近似。"""
    R = r_b
    Hcone = h_cone
    Hclnd = h_cyl
    L = np.sqrt(R**2 + Hcone**2)
    area = np.pi * R * L + 2 * np.pi * R * Hclnd
    mcone = np.pi * R * L * m_b / area
    mclnd = 2 * np.pi * R * Hclnd * m_b / area
    return (
        mclnd * R**2 / 2
        + mclnd * Hclnd**2 / 3
        + mcone * R**2 / 4
        + mcone * Hcone**2 / 6
    )


I_b = estimate_float_pitch_inertia_aproblem()
I_z = (1 / 12) * m_z * (3 * r_z**2 + h_z**2)


@dataclass(frozen=True)
class PowerResult:
    c_linear: float
    c_rotary: float
    p_total: float
    p_linear: float
    p_rotary: float


def make_ode(c_linear: float, c_rotary: float):
    def ode_system(t, x):
        z_z, z_b, th_z, th_b, zz_dot, zb_dot, thz_dot, thb_dot = x

        angle = th_z + th_b
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)

        thb_ddot = (
            -K_pitch * th_b
            - C_pitch * thb_dot
            + c_rotary * thz_dot
            + k_torsion * th_z
            + L_amp * np.cos(omega * t)
        ) / (I_b + J_add)

        I_z_total = I_z + m_z * z_z**2
        thz_ddot = (
            -c_rotary * thz_dot
            - k_torsion * th_z
            + m_z * g * z_z * sin_a
        ) / I_z_total - thb_ddot

        m_eff = m_b + m_add + m_z * sin_a**2
        zb_ddot = (
            k_linear * (z_z - l0 + m_z * g / k_linear) * cos_a
            + c_linear * zz_dot * cos_a
            - C_heave * zb_dot
            + f_amp * np.cos(omega * t)
            - K_heave * z_b
            + sin_a
            * (
                m_z
                * (
                    z_z * thz_ddot
                    + 2 * zz_dot * thz_dot
                    + z_z * thb_ddot
                    + 2 * zz_dot * thb_dot
                )
                - m_z * g * sin_a
            )
        ) / m_eff

        zz_ddot = (
            -k_linear * (z_z - l0)
            - c_linear * zz_dot
            - m_z * g * cos_a
            - m_z
            * (
                -z_z * thz_dot**2
                + zb_ddot * cos_a
                - z_z * thb_dot**2
                - 2 * z_z * thb_dot * thz_dot
            )
        ) / m_z

        return np.array(
            [zz_dot, zb_dot, thz_dot, thb_dot, zz_ddot, zb_ddot, thz_ddot, thb_ddot]
        )

    return ode_system


def solve_response(
    c_linear: float,
    c_rotary: float,
    n_periods: int,
    dt: float,
    rtol: float,
    atol: float,
):
    t_max = n_periods * T_period
    t_eval = np.arange(0.0, t_max, dt)
    if t_eval[-1] < t_max:
        t_eval = np.append(t_eval, t_max)
    x0 = np.array([l_eq, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    sol = solve_ivp(
        make_ode(c_linear, c_rotary),
        (0.0, t_max),
        x0,
        t_eval=t_eval,
        method="Radau",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def average_power(
    c_linear: float,
    c_rotary: float,
    n_periods: int = 45,
    drop_periods: int = 25,
    dt: float = 0.35,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> PowerResult:
    if not (0.0 <= c_linear <= 100000.0 and 0.0 <= c_rotary <= 100000.0):
        return PowerResult(c_linear, c_rotary, -np.inf, -np.inf, -np.inf)

    try:
        sol = solve_response(c_linear, c_rotary, n_periods, dt, rtol, atol)
    except RuntimeError:
        return PowerResult(c_linear, c_rotary, -np.inf, -np.inf, -np.inf)

    steady_start = drop_periods * T_period
    mask = sol.t >= steady_start
    if mask.sum() < 3:
        return PowerResult(c_linear, c_rotary, -np.inf, -np.inf, -np.inf)

    t = sol.t[mask]
    zz_dot = sol.y[4, mask]
    thz_dot = sol.y[6, mask]

    p_linear_inst = c_linear * zz_dot**2
    p_rotary_inst = c_rotary * thz_dot**2
    duration = t[-1] - t[0]

    p_linear = np.trapezoid(p_linear_inst, t) / duration
    p_rotary = np.trapezoid(p_rotary_inst, t) / duration
    return PowerResult(c_linear, c_rotary, p_linear + p_rotary, p_linear, p_rotary)


def coarse_scan() -> pd.DataFrame:
    values = np.linspace(0.0, 100000.0, 11)
    rows = []
    total = len(values) ** 2
    count = 0
    t0 = time.time()
    for c1 in values:
        for c2 in values:
            count += 1
            result = average_power(c1, c2)
            rows.append(result.__dict__)
            if count % 10 == 0 or count == total:
                elapsed = time.time() - t0
                print(f"粗扫描 {count:3d}/{total}, 当前最优 {max(r['p_total'] for r in rows):.6f} W, 用时 {elapsed:.1f}s")
    df = pd.DataFrame(rows).sort_values("p_total", ascending=False).reset_index(drop=True)
    return df


def refine_from(best: PowerResult) -> PowerResult:
    cache: dict[tuple[int, int], PowerResult] = {}

    def objective(x):
        c1, c2 = float(x[0]), float(x[1])
        if c1 < 0 or c1 > 100000 or c2 < 0 or c2 > 100000:
            return 1e9
        key = (round(c1), round(c2))
        if key not in cache:
            cache[key] = average_power(c1, c2)
        return -cache[key].p_total

    opt = minimize(
        objective,
        x0=np.array([best.c_linear, best.c_rotary]),
        method="Powell",
        bounds=[(0.0, 100000.0), (0.0, 100000.0)],
        options={"xtol": 50.0, "ftol": 1e-4, "maxiter": 35, "disp": True},
    )
    c1, c2 = opt.x
    return average_power(c1, c2)


def final_recompute(c_linear: float, c_rotary: float):
    sol = solve_response(
        c_linear,
        c_rotary,
        n_periods=45,
        dt=0.2,
        rtol=1e-8,
        atol=1e-10,
    )
    steady_start = 25 * T_period
    mask = sol.t >= steady_start
    t = sol.t[mask]
    zz_dot = sol.y[4, mask]
    thz_dot = sol.y[6, mask]
    p_linear_inst = c_linear * zz_dot**2
    p_rotary_inst = c_rotary * thz_dot**2
    duration = t[-1] - t[0]
    p_linear = np.trapezoid(p_linear_inst, t) / duration
    p_rotary = np.trapezoid(p_rotary_inst, t) / duration

    df = pd.DataFrame(
        {
            "时间 t (s)": sol.t,
            "振子垂荡位移 z_z (m)": sol.y[0],
            "浮子垂荡位移 z_b (m)": sol.y[1],
            "振子纵摇角位移 theta_z (rad)": sol.y[2],
            "浮子纵摇角位移 theta_b (rad)": sol.y[3],
            "振子垂荡速度 zz_dot (m/s)": sol.y[4],
            "浮子垂荡速度 zb_dot (m/s)": sol.y[5],
            "振子纵摇角速度 theta_z_dot (rad/s)": sol.y[6],
            "浮子纵摇角速度 theta_b_dot (rad/s)": sol.y[7],
            "垂荡瞬时输出功率 (W)": c_linear * sol.y[4] ** 2,
            "纵摇瞬时输出功率 (W)": c_rotary * sol.y[6] ** 2,
            "总瞬时输出功率 (W)": c_linear * sol.y[4] ** 2 + c_rotary * sol.y[6] ** 2,
        }
    )

    summary = pd.DataFrame(
        [
            {"指标": "最优垂荡阻尼系数 c_linear", "数值": c_linear, "单位": "N·s/m"},
            {"指标": "最优旋转阻尼系数 c_rotary", "数值": c_rotary, "单位": "N·m·s"},
            {"指标": "稳态平均总功率", "数值": p_linear + p_rotary, "单位": "W"},
            {"指标": "稳态平均垂荡功率", "数值": p_linear, "单位": "W"},
            {"指标": "稳态平均纵摇功率", "数值": p_rotary, "单位": "W"},
            {"指标": "稳态平均起始时刻", "数值": steady_start, "单位": "s"},
        ]
    )
    return df, summary


def main():
    print("=" * 72)
    print("2022 A题 问题4: 垂荡阻尼与旋转阻尼联合优化")
    print("=" * 72)
    print(f"波浪周期 T = {T_period:.6f} s")
    print(f"搜索范围: c_linear, c_rotary ∈ [0, 100000]")

    scan = coarse_scan()
    scan.to_excel("result4_scan.xlsx", index=False)
    best_grid = PowerResult(**scan.iloc[0].to_dict())
    print("\n粗扫描前 5 个候选:")
    print(scan.head(5).to_string(index=False))

    refined = refine_from(best_grid)
    print("\n局部优化候选:")
    print(refined)

    response, summary = final_recompute(refined.c_linear, refined.c_rotary)
    response.to_excel("result4_timeseries.xlsx", index=False)
    summary.to_excel("result4_summary.xlsx", index=False)

    print("\n高精度复算结果:")
    print(summary.to_string(index=False))
    print("\n已保存 result4_scan.xlsx, result4_timeseries.xlsx, result4_summary.xlsx")


if __name__ == "__main__":
    main()
