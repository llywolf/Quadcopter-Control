import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from rrt_algorithms_develop.rrt_algorithms_develop.rrt_algorithms.utilities.geometry import es_points_along_line
from splines_py.bsplines import bsplines_casadi

class PolytopeSearchSpace:
    def __init__(self, dimension_lengths, obstacles):
        self.dimension_lengths = np.array(dimension_lengths)
        self.dimensions = len(dimension_lengths)
        self.obstacles = obstacles

    def obstacle_free(self, x):
        x_vec = np.array(x).reshape(2, 1)
        for obs in self.obstacles:
            if np.all(obs["A"] @ x_vec <= obs["b"] + 1e-5):
                return False  # Collision!
        return True  # Free!

    def sample(self):
        x = np.random.uniform(
            self.dimension_lengths[:, 0], self.dimension_lengths[:, 1]
        )
        return tuple(x)

    def sample_free(self):
        while True:
            x = self.sample()
            if self.obstacle_free(x):
                return x

    def collision_free(self, start, end, r):
        points = es_points_along_line(start, end, r)
        return all(map(self.obstacle_free, points))


def generate_bspline_trajectory(path, degree=3, num_samples=100):
    """
    Takes a list of RRT waypoints and generates a smooth B-spline trajectory
    using the CasADi basis functions.
    """
    if path is None or len(path) < 2:
        return path

    P = np.array(path)
    num_ctrl_pts = len(P)
    n = num_ctrl_pts - 1

    if n < degree:
        degree = n

    # Generate CasADi Bspline basis functions
    b_levels, knot, x_sym = bsplines_casadi(n=n, deg=degree, knot=[0.0, 1.0])
    basis_funcs = b_levels[-1]

    t_vals = np.linspace(float(np.min(knot)), float(np.max(knot)) - 1e-5, num_samples)

    trajectory = np.zeros((num_samples, 2))

    for j, t in enumerate(t_vals):
        x_val, y_val = 0.0, 0.0
        for i in range(num_ctrl_pts):

            weight = float(basis_funcs[i](t))
            x_val += P[i, 0] * weight
            y_val += P[i, 1] * weight
        trajectory[j] = [x_val, y_val]

    # Snap the very last point exactly to the goal to fix the epsilon gap
    trajectory[-1] = P[-1]

    return trajectory


def optimize_control_points(path, safe_radius, degree=3, smooth_weight=50.0):
    """
    Optimizes control points using L1 relaxation for waypoint adherence.
    - Uses 'delta' to allow minimal, uniform deviation from RRT points.
    - Uses 'epsilon' to bound the control polygon for safety.
    """
    W = np.array(path).reshape(-1, 2) # RRT waypoints as a NumPy array
    num_waypoints = W.shape[0]

    # Increase degrees of freedom (Knot Refinement principle)
    # n = N-1 (number of control points minus 1)
    N = num_waypoints + degree

    # Generate Basis Matrices
    # We evaluate the basis functions at fixed time steps for each waypoint
    b_levels, knot, _ = bsplines_casadi(n=N - 1, deg=degree, knot=[0.0, 1.0])
    basis_funcs = b_levels[-1]
    tw = np.linspace(0.0, 1.0 - 1e-5, num_waypoints)

    solver = ca.Opti()

    P = solver.variable(N, 2)  # Control Points (N x 2)
    delta = solver.variable(1)  # L1 relaxation factor (allowed deviation)

    for j in range(num_waypoints):
        # Calculate curve position at time tw[j]
        basis_values = ca.vcat([basis_funcs[i](tw[j]) for i in range(N)])
        zj_x = ca.dot(basis_values, P[:, 0])
        zj_y = ca.dot(basis_values, P[:, 1])

        if j == 0 or j == num_waypoints - 1:
            # Hard constraints for Start and Goal
            solver.subject_to(zj_x == W[j, 0])
            solver.subject_to(zj_y == W[j, 1])
        else:
            # L1 Relaxation: Allowed deviation of 'delta' in each coordinate
            solver.subject_to(zj_x <= W[j, 0] + delta)
            solver.subject_to(zj_x >= W[j, 0] - delta)
            solver.subject_to(zj_y <= W[j, 1] + delta)
            solver.subject_to(zj_y >= W[j, 1] - delta)

    solver.subject_to(delta >= 0)
    # Limit maximum allowed relaxation to keep it close to the corridor
    solver.subject_to(delta <= safe_radius)

    # COST: Minimizing the acceleration of the control polygon
    accel_cost = 0
    for i in range(1, N - 1):
        accel_cost += ca.sumsqr(P[i - 1, :] - 2 * P[i, :] + P[i + 1, :])

    solver.minimize(smooth_weight * accel_cost + 1000 * delta**2)

    solver.solver("ipopt", {"print_time": False, "ipopt": {"print_level": 0}})

    try:
        sol = solver.solve()
        return sol.value(P), sol.value(delta)
    except Exception as e:
        print(f"Relaxed optimization failed: {e}")
        return W, 0.0