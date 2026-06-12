import numpy as np


def bspline_conversion_matrices(n, d, knot):
    """
    Python version of bsplineConversionMatrices.m

    Parameters
    ----------
    n : int
        Number such that there are n+1 control points.
    d : int
        Spline degree.
    knot : array-like
        Knot vector.

    Returns
    -------
    M : list of np.ndarray
        List of conversion matrices M[r-1], for r = 1..d
    Sd : list
        Kept for compatibility with MATLAB code (empty list).
    """
    knot = np.asarray(knot, dtype=float)
    M = []
    tmp = np.eye(n + 1)
    Sd = []

    for r in range(1, d + 1):
        Mr = np.zeros((n + r, n + 1 + r))

        for i in range(1, n + r + 1):  # MATLAB: 1 : n+r
            # Convert MATLAB 1-based indexing to Python 0-based indexing
            row = i - 1

            denom1 = knot[i + d - r - 1] - knot[i - 1]
            if knot[i + d - r - 1] == knot[i - 1]:
                Mr[row, i - 1] = 0.0
            else:
                Mr[row, i - 1] = (d - r) / denom1

            denom2 = knot[i + d - r] - knot[i]
            if knot[i + d - r] == knot[i]:
                Mr[row, i] = 0.0
            else:
                Mr[row, i] = -(d - r) / denom2

        tmp = tmp @ Mr
        M.append(tmp.copy())

    return M, Sd