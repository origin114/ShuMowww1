import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

# ============================================================
# 参数定义（全部使用标准单位：kg, m, s, N）
# ============================================================

# --- 浮子参数 ---
m_b = 4866.0            # 浮子质量 (kg)
r = 1.0                 # 浮子底面半径 (m)

# --- 振子参数 ---
m_z = 2433.0            # 振子质量 (kg)

# --- 环境参数 ---
rho = 1025.0            # 海水密度 (kg/m³)
g = 9.8                 # 重力加速度 (m/s²)

# --- PTO 参数 ---
k_spring = 80000.0      # 弹簧刚度 (N/m)

# --- 水动力参数（附件3，问题1行，只取垂荡列）---
omega = 1.4005          # 波浪圆频率 (s⁻¹)
m_add = 1335.535        # 垂荡附加质量 (kg)
C_wave = 656.3616       # 垂荡兴波阻尼系数 (N·s/m)
f_amp = 6250.0          # 垂荡激励力振幅 (N)

# --- 计算静水恢复力系数 ---
K_hydro = rho * g * np.pi * r**2   # ≈ 31559.7 N/m

print(f"静水恢复力系数 K_hydro = {K_hydro:.2f} N/m")
print(f"波浪周期 T = {2*np.pi/omega:.3f} s")

# ============================================================
# ODE 右端函数
# ============================================================

def ode_system(t, x, c_damp, nonlinear=False, alpha=0.5):
    """
    计算状态变量的导数 ẋ = f(t, x)

    参数:
        t: 当前时间
        x: 状态向量 [z_b, ż_b, z_z, ż_z]
        c_damp: 阻尼系数
        nonlinear: True=非线性阻尼(情况2), False=线性阻尼(情况1)
        alpha: 非线性阻尼的幂指数(情况2用)
    """
    z_b, zb_dot, z_z, zz_dot = x

    # 相对位移
    rel_disp = z_b - z_z

    # 相对速度
    rel_vel = zb_dot - zz_dot

    # 阻尼力（线性 vs 非线性）
    if nonlinear:
        F_damp = c_damp * np.abs(rel_vel) ** alpha * np.sign(rel_vel)
    else:
        F_damp = c_damp * rel_vel

    # PTO 总力 = 弹簧力 + 阻尼力
    F_PTO = k_spring * rel_disp + F_damp

    # === 浮子加速度 ===
    # 方程: (m_b + m_add)·z̈_b = f·cos(ωt) − C_波·ż_b − K_静水·z_b − F_PTO
    F_excite = f_amp * np.cos(omega * t)          # 波浪激励力
    F_total_buoy = F_excite - C_wave * zb_dot - K_hydro * z_b - F_PTO
    zb_ddot = F_total_buoy / (m_b + m_add)

    # === 振子加速度 ===
    # 方程: m_z·z̈_z = F_PTO
    zz_ddot = F_PTO / m_z

    return [zb_dot, zb_ddot, zz_dot, zz_ddot]


# ============================================================
# 求解函数
# ============================================================

def solve_problem(c_damp, nonlinear=False, alpha=0.5):
    """
    求解 40 个周期的运动，返回结果 DataFrame。
    """
    T_period = 2 * np.pi / omega          # 单周期 ≈ 4.487 s
    t_max = 40 * T_period                 # 最大时间 ≈ 179.5 s
    t_eval = np.arange(0, t_max, 0.2)     # 步长 0.2 s

    # 初始条件 [z_b, ż_b, z_z, ż_z] = [0, 0, 0, 0]
    x0 = [0.0, 0.0, 0.0, 0.0]

    # 调用 ODE 求解器
    sol = solve_ivp(
        ode_system,
        t_span=(0, t_max),
        y0=x0,
        t_eval=t_eval,
        method='RK45',
        args=(c_damp, nonlinear, alpha),
        rtol=1e-8,
        atol=1e-10
    )

    # 整理结果
    df = pd.DataFrame({
        '时间 t (s)': sol.t,
        '浮子位移 z_b (m)': sol.y[0],
        '浮子速度 ż_b (m/s)': sol.y[1],
        '振子位移 z_z (m)': sol.y[2],
        '振子速度 ż_z (m/s)': sol.y[3]
    })

    # 计算相对速度
    df['相对速度 v_rel (m/s)'] = df['浮子速度 ż_b (m/s)'] - df['振子速度 ż_z (m/s)']

    return df


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    # --- 情况 1：线性阻尼，c = 10000 ---
    print("正在计算 情况1（线性阻尼）...")
    df1 = solve_problem(c_damp=10000.0, nonlinear=False)
    df1.to_excel('result1-1.xlsx', index=False)
    print(f"  完成！共 {len(df1)} 个数据点，已保存到 result1-1.xlsx")

    # --- 情况 2：非线性阻尼，c = 10000, α = 0.5 ---
    print("正在计算 情况2（非线性阻尼）...")
    df2 = solve_problem(c_damp=10000.0, nonlinear=True, alpha=0.5)
    df2.to_excel('result1-2.xlsx', index=False)
    print(f"  完成！共 {len(df2)} 个数据点，已保存到 result1-2.xlsx")

    # --- 输出特定时刻的结果（论文中需要）---
    check_times = [10, 20, 40, 60, 100]
    print("\n" + "=" * 80)
    print("特定时刻结果汇总")
    print("=" * 80)

    for label, df in [("情况1（线性阻尼）", df1), ("情况2（非线性阻尼）", df2)]:
        print(f"\n--- {label} ---")
        print(f"{'时刻(s)':>8} {'浮子位移(m)':>14} {'浮子速度(m/s)':>14} "
              f"{'振子位移(m)':>14} {'振子速度(m/s)':>14}")
        print("-" * 70)
        for t_target in check_times:
            idx = (df['时间 t (s)'] - t_target).abs().idxmin()
            row = df.iloc[idx]
            print(f"{t_target:>8} {row['浮子位移 z_b (m)']:>14.6f} "
                  f"{row['浮子速度 ż_b (m/s)']:>14.6f} "
                  f"{row['振子位移 z_z (m)']:>14.6f} "
                  f"{row['振子速度 ż_z (m/s)']:>14.6f}")
