import json
import numpy as np

with open("bspline_poly.json", "r") as f:
    data = json.load(f)

intervals = data["intervals"]
coeffs_x = data["coefficients_x"]
coeffs_y = data["coefficients_y"]


def poly_to_python_expr(coeffs, var="u"):
    """
    coeffs are descending powers:
    [a_n, a_{n-1}, ..., a_0]
    """
    degree = len(coeffs) - 1
    terms = []

    for i, c in enumerate(coeffs):
        power = degree - i

        if abs(c) < 1e-12:
            continue

        c_str = f"{c:.12g}"

        if power == 0:
            terms.append(f"({c_str})")
        elif power == 1:
            terms.append(f"({c_str})*{var}")
        else:
            terms.append(f"({c_str})*{var}**{power}")

    return " + ".join(terms) if terms else "0.0"


def derivative_coeffs(coeffs):
    degree = len(coeffs) - 1
    d = []

    for i, c in enumerate(coeffs[:-1]):
        power = degree - i
        d.append(c * power)

    return d


print("def eval_spline_manual(s):")
print("    # Hardcoded piecewise polynomial exported from the B-spline")
print("")

for i, ((a, b), cx, cy) in enumerate(zip(intervals, coeffs_x, coeffs_y)):
    cx = np.array(cx, dtype=float)
    cy = np.array(cy, dtype=float)

    dcx = derivative_coeffs(cx)
    dcy = derivative_coeffs(cy)

    condition = "if" if i == 0 else "elif"

    print(f"    {condition} s <= {b:.12g}:")
    print(f"        u = s - ({a:.12g})")
    print(f"        x = {poly_to_python_expr(cx)}")
    print(f"        y = {poly_to_python_expr(cy)}")
    print(f"        dx = {poly_to_python_expr(dcx)}")
    print(f"        dy = {poly_to_python_expr(dcy)}")
    print("")

print("    else:")
print(f"        u = s - ({intervals[-1][0]:.12g})")
print(f"        x = {poly_to_python_expr(coeffs_x[-1])}")
print(f"        y = {poly_to_python_expr(coeffs_y[-1])}")
print(f"        dx = {poly_to_python_expr(derivative_coeffs(np.array(coeffs_x[-1], dtype=float)))}")
print(f"        dy = {poly_to_python_expr(derivative_coeffs(np.array(coeffs_y[-1], dtype=float)))}")
print("")
print("    return np.array([x, y]), np.array([dx, dy])")