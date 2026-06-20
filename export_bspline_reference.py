import json
import numpy as np


INPUT_JSON = "bspline_poly.json"
OUTPUT_FILE = "bspline_reference.py"


with open(INPUT_JSON, "r") as f:
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


lines = []

lines.append("import numpy as np")
lines.append("")
lines.append("")
lines.append("s_min = 0.0")
lines.append("s_max = 1.0")
lines.append("")
lines.append("")
lines.append("def eval_spline_manual(s):")
lines.append("    \"\"\"")
lines.append("    Hardcoded piecewise polynomial exported from the B-spline.")
lines.append("")
lines.append("    Returns")
lines.append("    -------")
lines.append("    position : np.ndarray")
lines.append("        [x, y]")
lines.append("")
lines.append("    derivative : np.ndarray")
lines.append("        [dx/ds, dy/ds]")
lines.append("    \"\"\"")
lines.append("")

for i, ((a, b), cx, cy) in enumerate(zip(intervals, coeffs_x, coeffs_y)):
    cx = np.array(cx, dtype=float)
    cy = np.array(cy, dtype=float)

    dcx = derivative_coeffs(cx)
    dcy = derivative_coeffs(cy)

    condition = "if" if i == 0 else "elif"

    lines.append(f"    {condition} s <= {b:.12g}:")
    lines.append(f"        u = s - ({a:.12g})")
    lines.append(f"        x = {poly_to_python_expr(cx)}")
    lines.append(f"        y = {poly_to_python_expr(cy)}")
    lines.append(f"        dx = {poly_to_python_expr(dcx)}")
    lines.append(f"        dy = {poly_to_python_expr(dcy)}")
    lines.append("")

lines.append("    else:")
lines.append(f"        u = s - ({intervals[-1][0]:.12g})")
lines.append(f"        x = {poly_to_python_expr(coeffs_x[-1])}")
lines.append(f"        y = {poly_to_python_expr(coeffs_y[-1])}")
lines.append(
    f"        dx = {poly_to_python_expr(derivative_coeffs(np.array(coeffs_x[-1], dtype=float)))}"
)
lines.append(
    f"        dy = {poly_to_python_expr(derivative_coeffs(np.array(coeffs_y[-1], dtype=float)))}"
)
lines.append("")
lines.append("    return np.array([x, y]), np.array([dx, dy])")
lines.append("")
lines.append("")
lines.append("def sample_reference_curve(z_value=-10.0, num_points=500):")
lines.append("    s_vals = np.linspace(s_min, s_max, num_points)")
lines.append("    ref = np.zeros((num_points, 3))")
lines.append("")
lines.append("    for i, s in enumerate(s_vals):")
lines.append("        xy, _ = eval_spline_manual(s)")
lines.append("        ref[i, 0] = xy[0]")
lines.append("        ref[i, 1] = xy[1]")
lines.append("        ref[i, 2] = z_value")
lines.append("")
lines.append("    return ref")
lines.append("")


with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(lines))

print(f"Exported B-spline reference to: {OUTPUT_FILE}")