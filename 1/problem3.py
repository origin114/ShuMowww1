"""
2022 数模 A 题 · 问题 3

四自由度非线性几何耦合模型。

符号约定:
  z_b, zb_dot             浮子相对静水平衡位置的垂荡位移与速度
  z_z, zz_dot             振子沿中轴方向的位置与速度
  theta_b, theta_b_dot    浮子纵摇角位移与角速度
  theta_z, theta_z_dot    振子纵摇角位移与角速度

初始条件按题面“初始时刻浮子和振子平衡于静水中”确定：
  浮子全局位移与角位移为 0；
  振子沿中轴的位置取弹簧静平衡长度 l_eq = l0 - m_z*g/k；
  相对速度与相对角速度为 0。


输出:
  result3.xlsx
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# ============================================================
# 参数定义
# ============================================================

# 附件4：结构与环境参数
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

# 题目给定：问题3阻尼
c_linear = 10000.0
c_rotary = 1000.0

# 附件3：问题3水动力参数
omega = 1.7152
m_add = 1028.876
J_add = 7001.914
C_heave = 683.4558
C_pitch = 654.3383
f_amp = 3640.0
L_amp = 1690.0

# 派生参数
K_heave = rho * g * np.pi * r_b**2
T_period = 2 * np.pi / omega
l_eq = l0 - m_z * g / k_linear


def estimate_float_pitch_inertia(n=4000):
    """
    估算浮子绕横向纵摇轴的转动惯量。

    假设浮子为均匀薄壳，由圆柱侧面、圆锥侧面和顶面圆盘组成。
    先按表面积分配质量，再计算绕整体质心处横向轴的转动惯量：
        I = ∫[(z-z_cm)^2 + r(z)^2/2] dm
    """
    R = r_b
    h1 = h_cyl
    h2 = h_cone
    l_cone = np.sqrt(R**2 + h2**2)

    area_cyl = 2 * np.pi * R * h1
    area_cone = np.pi * R * l_cone
    area_top = np.pi * R**2
    area_total = area_cyl + area_cone + area_top
    sigma = m_b / area_total

    masses = []
    z_moments = []
    inertias_origin = []

    # 圆柱侧面，z 从 0 到 h1
    z = np.linspace(0, h1, n)
    dz = z[1] - z[0]
    dm = sigma * 2 * np.pi * R * dz
    masses.append(dm * len(z))
    z_moments.append(np.sum(dm * z))
    inertias_origin.append(np.sum(dm * (z**2 + R**2 / 2)))

    # 圆锥侧面，z 从 h1 到 h1+h2，半径线性减小到 0
    z = np.linspace(h1, h1 + h2, n)
    dz = z[1] - z[0]
    r = R * (1 - (z - h1) / h2)
    ds_dz = np.sqrt(1 + (R / h2) ** 2)
    dm = sigma * 2 * np.pi * r * ds_dz * dz
    masses.append(np.sum(dm))
    z_moments.append(np.sum(dm * z))
    inertias_origin.append(np.sum(dm * (z**2 + r**2 / 2)))

    # 顶面圆盘，位于 z=0
    r = np.linspace(0, R, n)
    dr = r[1] - r[0]
    dm = sigma * 2 * np.pi * r * dr
    z = 0.0
    masses.append(np.sum(dm))
    z_moments.append(0.0)
    inertias_origin.append(np.sum(dm * (z**2 + r**2 / 2)))

    mass_total = np.sum(masses)
    z_cm = np.sum(z_moments) / mass_total
    I_origin = np.sum(inertias_origin)
    return I_origin - mass_total * z_cm**2


# AProblem.pdf 附录 B.3.1 按圆柱侧面和圆锥侧面面积分配浮子壳体质量，
# 由此得到与参考结果一致的浮子纵摇转动惯量。
def estimate_float_pitch_inertia_aproblem():
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

# 振子近似为实心圆柱，绕横向中心轴转动
I_z = (1 / 12) * m_z * (3 * r_z**2 + h_z**2)


# ============================================================
# 四自由度 ODE
# ============================================================

def ode_system(t, x):
    """
    状态变量:
      x = [z_z, z_b, theta_z, theta_b, zz_dot, zb_dot, theta_z_dot, theta_b_dot]

    注意这里为了贴合 AProblem 附录代码，状态顺序采用“振子在前、浮子在后”。
    z_z 是振子沿中轴方向的位置，z_b 是浮子相对静水平衡位置的垂荡位移。
    """
    z_z, z_b, th_z, th_b, zz_dot, zb_dot, thz_dot, thb_dot = x

    angle = th_z + th_b
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)

    # 浮子纵摇角加速度。
    thb_ddot = (
        -K_pitch * th_b
        - C_pitch * thb_dot
        + c_rotary * thz_dot
        + k_torsion * th_z
        + L_amp * np.cos(omega * t)
    ) / (I_b + J_add)

    # 振子相对转轴的转动惯量随滑移位置变化。
    I_z_total = I_z + m_z * z_z**2

    # 振子相对纵摇角加速度。
    thz_ddot = (
        -c_rotary * thz_dot
        - k_torsion * th_z
        + m_z * g * z_z * sin_a
    ) / I_z_total - thb_ddot

    # 浮子垂荡方向的等效质量项。
    m_eff = m_b + m_add + m_z * sin_a**2

    # 浮子垂荡加速度。该式来自 AProblem.pdf 附录 B.3.1 的 calcdiff34.py。
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

    # 振子沿中轴方向的滑移加速度。
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

    return np.array([zz_dot, zb_dot, thz_dot, thb_dot, zz_ddot, zb_ddot, thz_ddot, thb_ddot])


def solve_problem3(n_periods=40, dt=0.2):
    t_max = n_periods * T_period
    t_eval = np.arange(0, t_max, dt)
    # 题面给定“初始时刻浮子和振子平衡于静水中”。
    # 振子沿中轴方向的静平衡位置由弹簧原长和重力压缩量给出。
    x0 = np.array([l_eq, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    sol = solve_ivp(
        ode_system,
        (0, t_max),
        x0,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    df = pd.DataFrame({
        "时间 t (s)": sol.t,
        "浮子垂荡位移 z_b (m)": sol.y[1],
        "浮子垂荡速度 zb_dot (m/s)": sol.y[5],
        "浮子纵摇角位移 theta_b (rad)": sol.y[3],
        "浮子纵摇角速度 theta_b_dot (rad/s)": sol.y[7],
        "振子垂荡位移 z_z (m)": sol.y[0],
        "振子垂荡速度 zz_dot (m/s)": sol.y[4],
        "振子纵摇角位移 theta_z (rad)": sol.y[2],
        "振子纵摇角速度 theta_z_dot (rad/s)": sol.y[6],
    })
    return df


def print_check_times(df):
    check_times = [10, 20, 40, 60, 100]
    print("\n特定时刻结果")
    print("-" * 96)
    header = (
        f"{'t(s)':>7} {'z_b':>10} {'v_b':>10} {'theta_b':>12} {'omega_b':>12} "
        f"{'z_z':>10} {'v_z':>10} {'theta_z':>12} {'omega_z':>12}"
    )
    print(header)
    for tt in check_times:
        i = (df["时间 t (s)"] - tt).abs().idxmin()
        row = df.iloc[i]
        print(
            f"{tt:7.1f} "
            f"{row['浮子垂荡位移 z_b (m)']:10.6f} "
            f"{row['浮子垂荡速度 zb_dot (m/s)']:10.6f} "
            f"{row['浮子纵摇角位移 theta_b (rad)']:12.6f} "
            f"{row['浮子纵摇角速度 theta_b_dot (rad/s)']:12.6f} "
            f"{row['振子垂荡位移 z_z (m)']:10.6f} "
            f"{row['振子垂荡速度 zz_dot (m/s)']:10.6f} "
            f"{row['振子纵摇角位移 theta_z (rad)']:12.6f} "
            f"{row['振子纵摇角速度 theta_z_dot (rad/s)']:12.6f}"
        )


if __name__ == "__main__":
    print("=" * 72)
    print("2022 A题 问题3：垂荡 + 纵摇四自由度响应")
    print("=" * 72)
    print(f"波浪周期 T = {T_period:.4f} s, 计算时长 = {40*T_period:.2f} s")
    print(f"静水恢复力系数 K_heave = {K_heave:.3f} N/m")
    print(f"振子静平衡长度 l_eq = {l_eq:.6f} m")
    print(f"浮子纵摇转动惯量 I_b ≈ {I_b:.3f} kg·m^2")
    print(f"振子纵摇转动惯量 I_z ≈ {I_z:.3f} kg·m^2")

    result = solve_problem3()
    result.to_excel("result3.xlsx", index=False)
    print(f"\n已保存 result3.xlsx，共 {len(result)} 行。")
    print_check_times(result)
