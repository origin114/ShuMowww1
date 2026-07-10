import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import expm


# Parameters are kept identical to problem1.py.
m_b = 4866.0
m_z = 2433.0
r = 1.0
rho = 1025.0
g = 9.8
k_spring = 80000.0

omega = 1.4005
m_add = 1335.535
C_wave = 656.3616
f_amp = 6250.0
c_damp = 10000.0

K_hydro = rho * g * np.pi * r**2
T_period = 2 * np.pi / omega
t_max = 40 * T_period
t_eval = np.arange(0, t_max, 0.2)


def state_matrices():
    """State vector x = [z_b, z_z, z_b_dot, z_z_dot]^T."""
    mb_eff = m_b + m_add
    a = np.zeros((4, 4), dtype=float)
    b = np.zeros(4, dtype=float)

    a[0, 2] = 1.0
    a[1, 3] = 1.0

    # (m_b+m_a) z_b'' = F cos(wt) - C_h z_b' - K_h z_b
    #                    - k(z_b-z_z) - c(z_b'-z_z')
    a[2, 0] = -(K_hydro + k_spring) / mb_eff
    a[2, 1] = k_spring / mb_eff
    a[2, 2] = -(C_wave + c_damp) / mb_eff
    a[2, 3] = c_damp / mb_eff
    b[2] = f_amp / mb_eff

    # m_z z_z'' = k(z_b-z_z) + c(z_b'-z_z')
    a[3, 0] = k_spring / m_z
    a[3, 1] = -k_spring / m_z
    a[3, 2] = c_damp / m_z
    a[3, 3] = -c_damp / m_z
    return a, b


def ode_linear(t, x_problem1_order):
    """Use problem1.py state order: [z_b, z_b_dot, z_z, z_z_dot]."""
    z_b, zb_dot, z_z, zz_dot = x_problem1_order
    rel_disp = z_b - z_z
    rel_vel = zb_dot - zz_dot
    f_pto = k_spring * rel_disp + c_damp * rel_vel
    zb_ddot = (f_amp * np.cos(omega * t) - C_wave * zb_dot - K_hydro * z_b - f_pto) / (m_b + m_add)
    zz_ddot = f_pto / m_z
    return [zb_dot, zb_ddot, zz_dot, zz_ddot]


def numerical_solution():
    sol = solve_ivp(
        ode_linear,
        t_span=(0.0, t_max),
        y0=[0.0, 0.0, 0.0, 0.0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def analytic_solution(times):
    a, b = state_matrices()
    q = np.linalg.solve(1j * omega * np.eye(4) - a, b)
    x0 = np.zeros(4)
    x_ss_0 = np.real(q)
    exact = []
    steady = []
    for t in times:
        x_steady = np.real(q * np.exp(1j * omega * t))
        x_exact = x_steady + expm(a * t).dot(x0 - x_ss_0)
        exact.append(x_exact)
        steady.append(x_steady)
    return np.asarray(exact), np.asarray(steady), q


def main():
    sol = numerical_solution()
    exact, steady, q = analytic_solution(sol.t)

    # Reorder numerical solution to [z_b, z_z, z_b_dot, z_z_dot].
    num = np.vstack([sol.y[0], sol.y[2], sol.y[1], sol.y[3]]).T
    err_exact = exact - num
    err_steady = steady - num

    detail = pd.DataFrame(
        {
            "t (s)": sol.t,
            "numeric z_b (m)": num[:, 0],
            "analytic exact z_b (m)": exact[:, 0],
            "steady z_b (m)": steady[:, 0],
            "numeric z_z (m)": num[:, 1],
            "analytic exact z_z (m)": exact[:, 1],
            "steady z_z (m)": steady[:, 1],
            "numeric z_b_dot (m/s)": num[:, 2],
            "analytic exact z_b_dot (m/s)": exact[:, 2],
            "numeric z_z_dot (m/s)": num[:, 3],
            "analytic exact z_z_dot (m/s)": exact[:, 3],
            "exact error z_b (m)": err_exact[:, 0],
            "exact error z_z (m)": err_exact[:, 1],
            "steady error z_b (m)": err_steady[:, 0],
            "steady error z_z (m)": err_steady[:, 1],
        }
    )

    steady_start = t_max - 10 * T_period
    mask_steady = sol.t >= steady_start
    summary = pd.DataFrame(
        [
            ["K_hydro (N/m)", K_hydro],
            ["period T (s)", T_period],
            ["|Z_b| steady displacement amplitude (m)", abs(q[0])],
            ["|Z_z| steady displacement amplitude (m)", abs(q[1])],
            ["phase Z_b (rad)", np.angle(q[0])],
            ["phase Z_z (rad)", np.angle(q[1])],
            ["max abs exact error z_b all times (m)", np.max(np.abs(err_exact[:, 0]))],
            ["max abs exact error z_z all times (m)", np.max(np.abs(err_exact[:, 1]))],
            ["max abs exact error z_b last 10 periods (m)", np.max(np.abs(err_exact[mask_steady, 0]))],
            ["max abs exact error z_z last 10 periods (m)", np.max(np.abs(err_exact[mask_steady, 1]))],
            ["max abs steady-vs-numeric z_b last 10 periods (m)", np.max(np.abs(err_steady[mask_steady, 0]))],
            ["max abs steady-vs-numeric z_z last 10 periods (m)", np.max(np.abs(err_steady[mask_steady, 1]))],
            ["max |numeric z_b| last 10 periods (m)", np.max(np.abs(num[mask_steady, 0]))],
            ["max |numeric z_z| last 10 periods (m)", np.max(np.abs(num[mask_steady, 1]))],
        ],
        columns=["metric", "value"],
    )

    samples = []
    for target in [10, 20, 40, 60, 100]:
        idx = int(np.argmin(np.abs(sol.t - target)))
        samples.append(
            {
                "target t (s)": target,
                "actual t (s)": sol.t[idx],
                "numeric z_b (m)": num[idx, 0],
                "analytic exact z_b (m)": exact[idx, 0],
                "numeric z_z (m)": num[idx, 1],
                "analytic exact z_z (m)": exact[idx, 1],
                "abs error z_b (m)": abs(err_exact[idx, 0]),
                "abs error z_z (m)": abs(err_exact[idx, 1]),
            }
        )
    samples = pd.DataFrame(samples)

    with pd.ExcelWriter("result1-analytic-check.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        samples.to_excel(writer, sheet_name="sample_times", index=False)
        detail.to_excel(writer, sheet_name="timeseries", index=False)

    print(summary.to_string(index=False))
    print("\nSaved result1-analytic-check.xlsx")


if __name__ == "__main__":
    main()
