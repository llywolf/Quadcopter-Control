import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def bspline_conversion_matrices(n, d, knot):
    """
    Python version of bsplineConversionMatrices.m

    Parameters
    ----------
    n : int
        Number such that there are n+1 control points.
    d : int
        Degree of the splines.
    knot : array-like
        Knot vector.

    Returns
    -------
    M : list of np.ndarray
        List of cumulative conversion matrices.
    Sd : list
        Empty list, kept for compatibility with MATLAB output.
    """
    knot = np.asarray(knot, dtype=float)
    
    expected_len = n + d + 2
    if len(knot) != expected_len:
        raise ValueError(
            f"Knot vector must have length {expected_len} for n={n}, d={d}, but got {len(knot)}"
        )
    
    M = []
    tmp = np.eye(n + 1)
    Sd = []

    for r in range(1, d + 1):
        Mr = np.zeros((n + r, n + 1 + r))

        for i in range(1, n + r + 1):
            row = i - 1

            if knot[i + d - r - 1] == knot[i - 1]:
                Mr[row, i - 1] = 0.0
            else:
                Mr[row, i - 1] = (d - r) / (knot[i + d - r - 1] - knot[i - 1])

            if knot[i + d - r] == knot[i]:
                Mr[row, i] = 0.0
            else:
                Mr[row, i] = -(d - r) / (knot[i + d - r] - knot[i])

        tmp = tmp @ Mr
        M.append(tmp.copy())

    return M, Sd


def heaviside_casadi(z):
    """CasADi-compatible Heaviside."""
    return ca.if_else(z < 0, 0, 1)


def bsplines_casadi(n, deg, knot=None):
    """
    Python version of bsplines_casadi.m

    Parameters
    ----------
    n : int
        Number such that there are n+1 control points.
    deg : int
        Degree of the splines.
    knot : array-like, optional
        Knot vector.

    Returns
    -------
    b : list[list[casadi.Function]]
        B-spline basis functions.
    knot : np.ndarray
        Final knot vector used.
    x : casadi.SX
        Symbolic variable.
    """
    if knot is None:
        knot = np.array([0.0, 1.0], dtype=float)
    else:
        knot = np.asarray(knot, dtype=float)

    if len(knot) != n + deg + 2:
        knot = np.concatenate([
            np.ones(deg - 1) * np.min(knot),
            np.linspace(np.min(knot), np.max(knot), n - deg + 3),
            np.ones(deg - 1) * np.max(knot)
        ])

    m = len(knot) - 1
    x = ca.SX.sym("x", 1)

    b = []

    k = 1
    b1 = []
    for i in range(m):
        expr = heaviside_casadi(x - knot[i]) - heaviside_casadi(x - knot[i + 1])
        b1.append(ca.Function(f"f_{k}_{i+1}", [x], [expr]))
    b.append(b1)

    for k in range(2, deg + 1):
        bk = []
        for i in range(m - k + 1):
            left_den = knot[i + k - 1] - knot[i]
            right_den = knot[i + k] - knot[i + 1]

            left_term = ca.if_else(
                left_den != 0,
                b[k - 2][i](x) * (x - knot[i]) / left_den,
                0
            )

            right_term = ca.if_else(
                right_den != 0,
                b[k - 2][i + 1](x) * (knot[i + k] - x) / right_den,
                0
            )

            bk.append(ca.Function(f"f_{k}_{i+1}", [x], [left_term + right_term]))
        b.append(bk)

    return b, knot, x


def simulate_dynamics():
    """
    Python version of dinamica.m
    """
    l = 2.0
    alpha = 1.0
    beta = 0.1
    c = 0.2

    x = ca.MX.sym("x")
    y = ca.MX.sym("y")
    eta = ca.MX.sym("eta")
    v = ca.MX.sym("v")

    x_state = ca.vertcat(x, y, eta, v)

    theta = ca.MX.sym("theta")
    delta = ca.MX.sym("delta")
    u = ca.vertcat(theta, delta)

    xdot_state = ca.vertcat(
        v * ca.cos(eta),
        v * ca.sin(eta),
        (v * ca.tan(delta)) / l,
        -c * v + alpha * theta - beta
    )

    dae = {
        "x": x_state,
        "p": u,
        "ode": xdot_state
    }

    dt = 0.1
    options = {"tf": dt}
    integrator_cv = ca.integrator("integrator_cv", "rk", dae, options)

    x0 = np.array([2.0, 1.0, -0.2, 2.0])
    u_val = np.array([-0.3, 0.2])

    T = 10.0
    N = int(T / dt)

    trajectory = np.zeros((4, N + 1))
    trajectory[:, 0] = x0

    for k in range(N):
        result = integrator_cv(x0=trajectory[:, k], p=u_val)
        trajectory[:, k + 1] = np.array(result["xf"]).squeeze()

    x_vals = trajectory[0, :]
    y_vals = trajectory[1, :]

    plt.figure()
    plt.plot(x_vals, y_vals, "r-", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Vehicle Trajectory using CasADi Integrator")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

    return trajectory


if __name__ == "__main__":
    # Example usage for conversion matrices
    n = 4
    d = 3
    knot = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float)

    M, Sd = bspline_conversion_matrices(n, d, knot)
    print("B-spline conversion matrices:")
    for idx, mat in enumerate(M, start=1):
        print(f"M[{idx}] =\n{mat}\n")

    # Example usage for B-spline basis generation
    b, knot_used, x_sym = bsplines_casadi(n=4, deg=3, knot=[0, 1])
    print("Generated knot vector:", knot_used)
    print("Number of basis-function levels:", len(b))

    # Run dynamics simulation
    simulate_dynamics()