import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq


# Parameters are identical to the linear part of problem2.py.
m_b = 4866.0
r_b = 1.0
m_z = 2433.0
rho = 1025.0
g = 9.8
k_s = 80000.0

omega = 2.2143
m_add = 1165.992
C_w = 167.8395
f_amp = 4890.0

K_h = rho * g * np.pi * r_b**2


def complex_amplitude(c):
    """Return [Z_b, Z_z] for the steady response under linear PTO damping."""
    m = np.array([[m_b + m_add, 0.0], [0.0, m_z]], dtype=complex)
    cmat = np.array([[C_w + c, -c], [-c, c]], dtype=complex)
    kmat = np.array([[K_h + k_s, -k_s], [-k_s, k_s]], dtype=complex)
    force = np.array([f_amp, 0.0], dtype=complex)
    dynamic_stiffness = kmat - omega**2 * m + 1j * omega * cmat
    return np.linalg.solve(dynamic_stiffness, force)


def analytic_power(c):
    """Steady average output power for linear damping."""
    z = complex_amplitude(c)
    z_rel = z[0] - z[1]
    return 0.5 * c * omega**2 * abs(z_rel) ** 2


def derivative_power(c):
    """Numerical derivative only for reporting the stationary condition."""
    h = max(1e-3, c * 1e-6)
    return (analytic_power(c + h) - analytic_power(c - h)) / (2 * h)


def main():
    opt = minimize_scalar(lambda c: -analytic_power(c), bounds=(1e-6, 1e5), method="bounded", options={"xatol": 1e-9})
    c_opt = float(opt.x)
    p_opt = analytic_power(c_opt)
    z_opt = complex_amplitude(c_opt)

    # A dense scan is saved for plotting or independent checking.
    c_grid = np.r_[np.linspace(1e-6, 1000, 120), np.linspace(1000, 100000, 900)]
    p_grid = np.array([analytic_power(c) for c in c_grid])
    scan = pd.DataFrame({"c (N s/m)": c_grid, "analytic steady power (W)": p_grid})

    # Bracket where the power remains within 0.1 percent of the analytic maximum.
    threshold = 0.999 * p_opt
    left = brentq(lambda x: analytic_power(x) - threshold, 1e-6, c_opt)
    right = brentq(lambda x: analytic_power(x) - threshold, c_opt, 1e5)

    reference_c = 37420.87
    reference_p = analytic_power(reference_c)

    summary = pd.DataFrame(
        [
            ["analytic optimal c (N s/m)", c_opt],
            ["analytic optimal power (W)", p_opt],
            ["|Z_b| at optimum (m)", abs(z_opt[0])],
            ["|Z_z| at optimum (m)", abs(z_opt[1])],
            ["|Z_b-Z_z| at optimum (m)", abs(z_opt[0] - z_opt[1])],
            ["phase Z_b at optimum (rad)", np.angle(z_opt[0])],
            ["phase Z_z at optimum (rad)", np.angle(z_opt[1])],
            ["dP/dc at optimum approx", derivative_power(c_opt)],
            ["power at numerical c=37420.87 (W)", reference_p],
            ["difference vs analytic optimum (W)", p_opt - reference_p],
            ["relative difference vs analytic optimum", (p_opt - reference_p) / p_opt],
            ["0.1 percent plateau left c (N s/m)", left],
            ["0.1 percent plateau right c (N s/m)", right],
        ],
        columns=["metric", "value"],
    )

    with pd.ExcelWriter("result2-linear-analytic.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        scan.to_excel(writer, sheet_name="power_curve", index=False)

    print(summary.to_string(index=False))
    print("\nSaved result2-linear-analytic.xlsx")


if __name__ == "__main__":
    main()
