import numpy as np
import casadi as ca


def heaviside_casadi(z):
    """CasADi-compatible Heaviside step."""
    return ca.if_else(z < 0, 0, 1)


def bsplines_casadi(n, deg, knot=None):
    """
    Python version of bsplines_casadi.m

    Parameters
    ----------
    n : int
        Number such that there are n+1 control points.
    deg : int
        Degree/order parameter used in the original MATLAB code.
    knot : array-like, optional
        Knot vector. If None, defaults to [0, 1].

    Returns
    -------
    b : list of list of casadi.Function
        Nested list of B-spline basis functions.
        b[k-1][i-1] corresponds to MATLAB b{k}{i}
    knot : np.ndarray
        Final knot vector used.
    x : casadi.SX
        Symbolic variable.
    """
    if knot is None:
        knot = np.array([0.0, 1.0], dtype=float)
    else:
        knot = np.asarray(knot, dtype=float)

    if len(knot) != n + 1 + deg:
        kmin = np.min(knot)
        kmax = np.max(knot)
        knot = np.concatenate([
            np.ones(deg - 1) * kmin,
            np.linspace(kmin, kmax, n - deg + 3),
            np.ones(deg - 1) * kmax
        ])

    m = len(knot) - 1
    x = ca.SX.sym("x", 1)

    b = []

    # Order 1 basis functions
    b1 = []
    for i in range(m):
        expr = heaviside_casadi(x - knot[i]) - heaviside_casadi(x - knot[i + 1])
        f = ca.Function(f"f_1_{i+1}", [x], [expr])
        b1.append(f)
    b.append(b1)

    # Recursive Cox-de Boor relation
    for k in range(2, deg + 1):
        bk = []
        for i in range(m - k + 1):
            denom1 = knot[i + k - 1] - knot[i]
            denom2 = knot[i + k] - knot[i + 1]

            term1 = ca.if_else(
                denom1 != 0,
                b[k - 2][i](x) * (x - knot[i]) / denom1,
                0
            )

            term2 = ca.if_else(
                denom2 != 0,
                b[k - 2][i + 1](x) * (knot[i + k] - x) / denom2,
                0
            )

            f = ca.Function(f"f_{k}_{i+1}", [x], [term1 + term2])
            bk.append(f)
        b.append(bk)

    return b, knot, x