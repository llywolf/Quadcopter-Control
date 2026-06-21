import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parent
        / "rrt_algorithms_develop"
        / "rrt_algorithms_develop"
    )
)

from matplotlib.pylab import array
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
import roblib as rl
import json
from scipy.interpolate import BSpline, PPoly
from splines_py.bsplines import bsplines_casadi

from rrt_algorithms_develop.rrt_algorithms_develop.rrt_algorithms.rrt.rrt import RRT
from trajectory_generator import (
    optimize_control_points,
    generate_bspline_trajectory,
    PolytopeSearchSpace,
)
from nmpc import FullStateNMPC

from rand_position_polytopes_generation import DEFAULT_TEMPLATE_SHAPES, randomize_obstacle_positions
from scenario_obstacles import save_obstacle_vertices
from initial_obstacles import get_initial_obstacle_shapes
import initial_reference

from metrics import print_tracking_metrics

# Scenario selection
USE_RANDOM_REFERENCE = True
USE_RANDOM_OBSTACLES = True

# set seed for reproducibility
np.random.seed(123456)

#! PARAMETERS THAT CAN BE TUNED FOR TRAJECTORY GENERATION
OBSTACLE_INFLATION_RADIUS = 0.3  # How much to inflate obstacles for safety margin
RRT_SAMPLING_RADIUS = 0.1  # Radius for RRT collision checking
RRT_Q = 1.5  # RRT growth factor (how far to extend towards sampled point)
RRT_MAX_SAMPLES = 2500  # Max number of samples for RRT
RRT_PRC = 0.15  # Probability of directly sampling the goal in RRT

BSPLINE_DEGREE = 5  # Degree of the B-spline for trajectory generation
BSPLINE_SMOOTH_WEIGHT = 15.0  # Weight for smoothness in control point optimization
BSPLINE_SAFE_RADIUS = (
    0.2  # Safety radius for control point optimization (!must be < inflation radius)
)


def create_convex_polytope(points):
    """
    Forms a convex hull from 2D points and returns A, b, and ordered Vertices.
    """
    points = np.array(points)
    hull = ConvexHull(points)

    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1].reshape(-1, 1)
    V = points[hull.vertices]

    return A, b, V

def inflate_obstacle(A, b, V_original, inflation_radius):
    """
    Inflates a convex polytope by shifting inequalities outward,
    then calculates the new vertices for visualization.
    """
    norms = np.linalg.norm(A, axis=1).reshape(-1, 1)
    A_inflated = A
    b_inflated = b + inflation_radius * norms

    # Find new vertices using Halfspace Intersection
    interior_point = np.mean(V_original, axis=0)

    # scipy expects the format: A*x + c <= 0.
    # Since our format is A*x <= b, we rearrange to A*x - b <= 0.
    halfspaces = np.hstack((A_inflated, -b_inflated))

    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        V_inflated = hs.intersections

        # Sort the new vertices counter-clockwise so roblib can draw them correctly
        hull = ConvexHull(V_inflated)
        V_inflated_ordered = V_inflated[hull.vertices]
    except Exception as e:
        print(f"Warning: Could not compute inflated vertices: {e}")
        V_inflated_ordered = V_original

    return A_inflated, b_inflated, V_inflated_ordered

# Dynamics integration using RK2 for the quadrotor
def clock_quadri_rk2(p_in, R_in, vr_in, wr_in, w_in, B_mat, I, g, m, dt):
        def dynamics(p_loc, R_loc, vr_loc, wr_loc, w_loc):
            w2_loc = w_loc * np.abs(w_loc)
            tau_loc = B_mat @ w2_loc.flatten()
            p_dot = R_loc @ vr_loc
            vr_dot = -rl.adjoint(wr_loc) @ vr_loc + np.linalg.inv(R_loc) @ np.array([[0],[0],[g]]) + np.array([[0],[0],[-tau_loc[0]/m]])
            wr_dot = np.linalg.inv(I) @ (-rl.adjoint(wr_loc) @ I @ wr_loc + tau_loc[1:4].reshape(3,1))
            return p_dot, vr_dot, wr_dot

        k1_p, k1_vr, k1_wr = dynamics(p_in, R_in, vr_in, wr_in, w_in)
        p_mid = p_in + (2/3) * dt * k1_p
        vr_mid = vr_in + (2/3) * dt * k1_vr
        wr_mid = wr_in + (2/3) * dt * k1_wr
        R_mid = R_in @ rl.expw((2/3) * dt * wr_in)

        k2_p, k2_vr, k2_wr = dynamics(p_mid, R_mid, vr_mid, wr_mid, w_in)
        p_out = p_in + dt * (0.25 * k1_p + 0.75 * k2_p)
        vr_out = vr_in + dt * (0.25 * k1_vr + 0.75 * k2_vr)
        wr_out = wr_in + dt * (0.25 * k1_wr + 0.75 * k2_wr)
        wr_eff = 0.25 * wr_in + 0.75 * wr_mid
        R_out = R_in @ rl.expw(dt * wr_eff)
        return p_out, R_out, vr_out, wr_out
    
def draw_obstacles_3d(ax, obstacles):
        for obs in obstacles:
            V = obs["vertices"]
            V_closed = np.vstack((V, V[0]))
            ax.plot(V_closed[:, 0], V_closed[:, 1], 8, color='red', alpha=0.6)
            ax.plot(V_closed[:, 0], V_closed[:, 1], 12, color='red', alpha=0.6)
            for v in V: 
                ax.plot([v[0], v[0]], [v[1], v[1]], [8, 12], color='red', alpha=0.6)


# save bspline polynomial

def export_bspline_as_polynomial(control_points, degree, filename="bspline_poly.json"):
    P = np.asarray(control_points, dtype=float)
    N = P.shape[0]

    degree = min(degree, N - 1)

    _, knot, _ = bsplines_casadi(
        n=N - 1,
        deg=degree,
        knot=[0.0, 1.0]
    )

    knot = np.asarray(knot, dtype=float).flatten()

    sx = BSpline(knot, P[:, 0], degree, extrapolate=False)
    sy = BSpline(knot, P[:, 1], degree, extrapolate=False)

    px = PPoly.from_spline(sx)
    py = PPoly.from_spline(sy)

    intervals = []
    coeffs_x = []
    coeffs_y = []

    for i in range(len(px.x) - 1):
        a = float(px.x[i])
        b = float(px.x[i + 1])

        if b <= a:
            continue

        intervals.append([a, b])
        coeffs_x.append(px.c[:, i].tolist())
        coeffs_y.append(py.c[:, i].tolist())

    data = {
        "degree": degree,
        "intervals": intervals,
        "coefficients_x": coeffs_x,
        "coefficients_y": coeffs_y
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Exported spline polynomial to {filename}")


def get_ref_window(ref_trajectory, i, Npred):
    required_cols = Npred + 1
    ncols = ref_trajectory.shape[1]

    # If i is already past the reference length, hold the final reference
    if i >= ncols:
        last_col = ref_trajectory[:, -1:].copy()
        return np.repeat(last_col, required_cols, axis=1)

    # Normal case
    X_ref_window = ref_trajectory[:, i : i + required_cols]

    # If the window is too short, pad with the final reference column
    if X_ref_window.shape[1] < required_cols:
        missing_cols = required_cols - X_ref_window.shape[1]
        last_col = ref_trajectory[:, -1:].copy()
        padding = np.repeat(last_col, missing_cols, axis=1)
        X_ref_window = np.hstack((X_ref_window, padding))

    return X_ref_window

def sample_reference_xy(reference_module, num_samples=200):
    s_vals = np.linspace(reference_module.s_min, reference_module.s_max, num_samples)
    xy = np.zeros((num_samples, 2))

    for i, s in enumerate(s_vals):
        pos, _ = reference_module.eval_spline_manual(s)
        xy[i, :] = pos

    return xy

# main simulation
def main():
    bounds = {"xmin": 0, "xmax": 20, "ymin": 0, "ymax": 20}
    bspline_traj = None

    if USE_RANDOM_REFERENCE:
        start = np.array([[2], [2]])
        goal = np.array([[18], [18]])
    else:
        bspline_traj = sample_reference_xy(
            initial_reference,
            num_samples=200
        )

        start = bspline_traj[0].reshape(2, 1)
        goal = bspline_traj[-1].reshape(2, 1)


    if USE_RANDOM_OBSTACLES:
        base_obstacles = []
        shapes = randomize_obstacle_positions(
            DEFAULT_TEMPLATE_SHAPES,
            bounds,
            start,
            goal,
            start_goal_clearance=1.0,
            obstacle_gap=0.3,
            seed=None,
        )
    else:
        base_obstacles = []
        shapes = get_initial_obstacle_shapes()
        
    save_obstacle_vertices(shapes)

    # * Create Convex Polytopes from the defined obstacle
    for shape in shapes:
        A, b, V = create_convex_polytope(shape)
        base_obstacles.append({"A": A, "b": b, "vertices": V})

    # * Inflate Obstacles for trajectory optimization safety margin
    inflated_obstacles = []
    inflation_radius = OBSTACLE_INFLATION_RADIUS

    for obs in base_obstacles:
        A_inf, b_inf, V_inf = inflate_obstacle(
            obs["A"], obs["b"], obs["vertices"], inflation_radius
        )
        inflated_obstacles.append({"A": A_inf, "b": b_inf, "vertices": V_inf})

    rrt = None
    path = None
    optimized_control_points = None

    # * Create Search Space for RRT with Inflated Obstacles
    dim_lengths = [(bounds["xmin"], bounds["xmax"]), (bounds["ymin"], bounds["ymax"])]
    search_space = PolytopeSearchSpace(dim_lengths, inflated_obstacles)

    rrt = RRT(
        X=search_space,
        q=RRT_Q,
        x_init=tuple(start.flatten()),
        x_goal=tuple(goal.flatten()),
        max_samples=RRT_MAX_SAMPLES,
        r=RRT_SAMPLING_RADIUS,
        prc=RRT_PRC,
    )

    print(">> Computing RRT Path")
    raw_path = rrt.rrt_search()

    path = raw_path

    if path is None:
        raise ValueError("RRT failed to find a path!!!")
    else:
        print(f">> Path found successfully! Waypoints reduced to: {len(path)}")


    if USE_RANDOM_REFERENCE:
        print(">> Optimizing RRT Control Points with CasADi")

        optimized_control_points, delta = optimize_control_points(
            path,
            safe_radius=BSPLINE_SAFE_RADIUS,
            degree=BSPLINE_DEGREE,
            smooth_weight=BSPLINE_SMOOTH_WEIGHT,
        )

        print(f">> Optimal relaxation delta: {delta:.6f}")

        export_bspline_as_polynomial(
            optimized_control_points,
            BSPLINE_DEGREE,
            "bspline_poly.json"
        )

        print("Generating B-spline trajectory...")
        bspline_traj = generate_bspline_trajectory(
            optimized_control_points,
            degree=BSPLINE_DEGREE,
            num_samples=200
        )

    else:
        print(">> Using initial_reference.py trajectory")
        optimized_control_points = None
        
    if bspline_traj is None:
        raise RuntimeError("B-spline trajectory was not initialized.")
    
    # environment plots

    def draw_2d_obstacles(ax, base_obstacles, inflated_obstacles=None):
        for obs in base_obstacles:
            V = obs["vertices"]
            V_closed = np.vstack((V, V[0]))
            ax.plot(V_closed[:, 0], V_closed[:, 1], color="red", linewidth=2)
            ax.fill(V_closed[:, 0], V_closed[:, 1], color="red", alpha=0.25)

        if inflated_obstacles is not None:
            for obs in inflated_obstacles:
                V = obs["vertices"]
                V_closed = np.vstack((V, V[0]))
                ax.plot(V_closed[:, 0], V_closed[:, 1], color="orange", linewidth=1.5)
                ax.fill(V_closed[:, 0], V_closed[:, 1], color="orange", alpha=0.2)


    def setup_2d_axis(ax, bounds, title):
        ax.set_xlim(bounds["xmin"], bounds["xmax"])
        ax.set_ylim(bounds["ymin"], bounds["ymax"])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(title)


    def draw_start_goal(ax, start, goal):
        ax.scatter(start[0, 0], start[1, 0], color="blue", s=80, label="Start", zorder=5)
        ax.scatter(goal[0, 0], goal[1, 0], color="green", s=80, label="Goal", zorder=5)

        ax.text(start[0, 0], start[1, 0] - 0.7, "Start", color="blue", ha="center")
        ax.text(goal[0, 0], goal[1, 0] + 0.7, "Goal", color="green", ha="center")


    # Figure 1: Start, Goal and Obstacles
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    setup_2d_axis(ax1, bounds, "Start, Goal and Obstacles")
    draw_2d_obstacles(ax1, base_obstacles)
    draw_start_goal(ax1, start, goal)
    ax1.legend()
    fig1.canvas.draw()
    plt.show(block=False)
    plt.pause(0.1)


    # Figure 2: RRT Tree and RRT Path
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    setup_2d_axis(ax2, bounds, "RRT Tree and Final RRT Path")
    draw_2d_obstacles(ax2, base_obstacles, inflated_obstacles)

    # Draw RRT exploration tree
    E = rrt.trees[0].E
    for child, parent in E.items():
        if parent is not None:
            child = np.array(child)
            parent = np.array(parent)
            ax2.plot(
                [parent[0], child[0]],
                [parent[1], child[1]],
                color="gray",
                linewidth=0.8,
                alpha=0.7,
            )

    # Draw final RRT path
    if path is not None:
        path_arr = np.array(path)
        ax2.plot(
            path_arr[:, 0],
            path_arr[:, 1],
            color="green",
            linewidth=2.5,
            label="Final RRT Path",
        )
        ax2.scatter(path_arr[:, 0], path_arr[:, 1], color="green", s=20)

    draw_start_goal(ax2, start, goal)
    ax2.legend()
    fig2.canvas.draw()
    plt.show(block=False)
    plt.pause(0.1)


    # Figure 3: RRT Control Polygon and B-Spline Trajectory
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    setup_2d_axis(ax3, bounds, "B-Spline Trajectory")
    draw_2d_obstacles(ax3, base_obstacles, inflated_obstacles)

    # Draw optimized control polygon / RRT control points
    if optimized_control_points is not None:
        cp = np.array(optimized_control_points)
        ax3.plot(
            cp[:, 0],
            cp[:, 1],
            color="pink",
            linewidth=2,
            marker="o",
            label="Optimized Control Points",
        )

    # Draw B-spline trajectory
    if bspline_traj is not None:
        ax3.plot(
            bspline_traj[:, 0],
            bspline_traj[:, 1],
            color="magenta",
            linewidth=3,
            label="B-Spline Trajectory",
        )

    draw_start_goal(ax3, start, goal)
    ax3.legend()
    fig3.canvas.draw()
    plt.show(block=False)
    plt.pause(0.1)
    
    # Preparing reference trajectory for NMPC
    dt_nmpc = 0.05
    Npred = 15

    # Make sure path starts/ends exactly where expected
    bspline_traj[0] = start.flatten()
    bspline_traj[-1] = goal.flatten()

    # Simple speed easing for quadcopter:
    # slow start -> faster middle -> slow end
    N_traj = len(bspline_traj)
    s = np.linspace(0.0, 1.0, N_traj)

    # Smoothstep profile
    s_smooth = 3*s**2 - 2*s**3

    # Resample trajectory using the smooth progress variable
    idx_old = np.linspace(0.0, 1.0, N_traj)

    bspline_smooth = np.zeros_like(bspline_traj)
    bspline_smooth[:, 0] = np.interp(s_smooth, idx_old, bspline_traj[:, 0])
    bspline_smooth[:, 1] = np.interp(s_smooth, idx_old, bspline_traj[:, 1])

    # Padding
    start_pad_steps = Npred
    end_pad_steps = Npred + 60

    total_steps = start_pad_steps + N_traj + end_pad_steps
    ref_trajectory = np.zeros((12, total_steps))

    # Start hover
    ref_trajectory[0:2, :start_pad_steps] = start.reshape(2, 1)
    ref_trajectory[2, :start_pad_steps] = -10.0

    # Main trajectory
    traj_start = start_pad_steps
    traj_end = traj_start + N_traj

    ref_trajectory[0:2, traj_start:traj_end] = bspline_smooth.T
    ref_trajectory[2, traj_start:traj_end] = -10.0

    # End hover
    ref_trajectory[0:2, traj_end:] = goal.reshape(2, 1)
    ref_trajectory[2, traj_end:] = -10.0

    # Velocity calculation
    ref_trajectory[3:6, :-1] = (
        ref_trajectory[0:3, 1:] - ref_trajectory[0:3, :-1]
    ) / dt_nmpc

    # Force hover velocity at the very end
    ref_trajectory[3:6, -1] = 0.0
    ref_trajectory[9:12, :] = 0.0
    N_traj = total_steps
    

    print(">> Executing Trajectory Tracking Simulation...")
    
    # Drone Physics Parameters
    # m, g, b, d, l = 10.0, 9.81, 2.0, 1.0, 1.0
    # I = np.array([[10,0,0], [0,10,0], [0,0,20]])
    m, g, b, d, l = 1.3269, 9.81, 3.15e-5, 1, 0.25
    I = array([[0.01295, 0, 0], [0, 0.01244, 0], [0, 0, 0.01571]])  
    B_mat = np.array([[b, b, b, b],
                      [-b*l, 0, b*l, 0],
                      [0, -b*l, 0, b*l],
                      [-d, d, -d, d]])

    # Initial State
    p = ref_trajectory[0:3, 0].reshape(3, 1)
    R = np.eye(3)
    vr = np.zeros((3, 1))
    wr = np.zeros((3, 1))
    alpha_rotors = np.zeros((4, 1))

    w_max = 700.0 # 10 for old params
    # Initialize Full-State NMPC
    nmpc = FullStateNMPC(dt=dt_nmpc, Npred=Npred, m=m, g=g, obstacles=inflated_obstacles, B_mat=B_mat, I_mat=I, w_max = w_max)
    
    # Physics Integration settings
    dt_phys = 0.01  
    steps_per_nmpc = int(dt_nmpc / dt_phys) 

    # uncomment for iterative 3d plot
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection='3d')
    

    traj_history = []
    
    #  Data loggers for analysis plots
    time_history = []
    error_history = []
    command_history = []
    
    metric_pos_history = []
    metric_ref_history = []
    
    

    target_goal_3d = np.array([[goal[0,0]], [goal[1,0]], [-10.0]])
    i = 0
    while np.linalg.norm(p - target_goal_3d) > 0.4:
        
        phi, theta, psi = rl.eulermat2angles(R)
        v_world = R @ vr
        omega_body = wr.flatten()
        
        current_state = np.array([
            p[0,0], p[1,0], p[2,0], v_world[0,0], v_world[1,0], v_world[2,0],
            phi, theta, psi, omega_body[0], omega_body[1], omega_body[2]
        ])
        
        X_ref_window = get_ref_window(ref_trajectory, i, Npred)
        
        w_opt = nmpc.get_control(current_state, X_ref_window)

        for _ in range(steps_per_nmpc):
            p, R, vr, wr = clock_quadri_rk2(p, R, vr, wr, w_opt, B_mat, I, g, m, dt_phys)
            traj_history.append(p.flatten())
            alpha_rotors += dt_phys * 30 * w_opt.reshape(-1, 1)
            
        #Record data for analysis
        time_history.append(i * dt_nmpc)
        ref_idx = min(i, ref_trajectory.shape[1] - 1)
        ref_p = ref_trajectory[0:3, ref_idx].reshape(3, 1)
        
        error_history.append((p - ref_p).flatten())
        command_history.append(w_opt.flatten())
        
        metric_pos_history.append(p.flatten())
        metric_ref_history.append(ref_p.flatten())
        

        # Visualization (uncomment for iterative 3D plot)
        rl.clean3D(ax3d, 0, 20, 0, 20, 0, 20)
        draw_obstacles_3d(ax3d, base_obstacles)

        ax3d.plot(ref_trajectory[0, :N_traj], ref_trajectory[1, :N_traj], -ref_trajectory[2, :N_traj], 
                  'g--', linewidth=2, label='B-Spline Reference')
        
        hist = np.array(traj_history)
        ax3d.plot(hist[:, 0], hist[:, 1], -hist[:, 2], 
                  'b-', linewidth=2, label='Actual Flight Path')

        # draw quadrotor in unmirrored X coordinates, while keeping correct NED height
        M = np.diag([-1.0, 1.0, 1.0])
        R_draw = M @ R @ M
        
        p_draw = p.copy()
        p_draw[0, 0] = -p_draw[0, 0]

        rl.draw_quadrotor3D(ax3d, p_draw, R_draw, alpha_rotors, l=0.4, mirror=-1)

        # Highlight the physical Goal 
        ax3d.scatter(goal[0,0], goal[1,0], 10, color='green', s=100, label='Goal')
        

        ax3d.set_title(f"Full-State NMPC")
        ax3d.set_xlabel('X [m]')
        ax3d.set_ylabel('Y [m]')
        ax3d.set_zlabel('-Z (Altitude) [m]')
        
        rl.pause(0.1)
        i += 1

    print(">> Tracking Complete! Goal Reached.")
    plt.ioff()
    plt.show()
    
    
    
    # Error and Command Evolution Plots
    time_arr = np.array(time_history)
    error_arr = np.array(error_history)
    cmd_arr = np.array(command_history)
    hist = np.array(traj_history)
    
    # Final 3D plot drawn once
    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection='3d')

    rl.clean3D(ax3d, 0, 20, 0, 20, 0, 20)
    draw_obstacles_3d(ax3d, base_obstacles)

    ax3d.plot(
        ref_trajectory[0, :N_traj],
        ref_trajectory[1, :N_traj],
        -ref_trajectory[2, :N_traj],
        'g--',
        linewidth=2,
        label='B-Spline Reference'
    )

    ax3d.plot(
        hist[:, 0],
        hist[:, 1],
        -hist[:, 2],
        'b-',
        linewidth=2,
        label='Actual Flight Path'
    )

    # Draw final quadrotor pose
    M = np.diag([-1.0, 1.0, 1.0])
    R_draw = M @ R @ M

    p_draw = p.copy()
    p_draw[0, 0] = -p_draw[0, 0]

    rl.draw_quadrotor3D(
        ax3d,
        p_draw,
        R_draw,
        alpha_rotors,
        l=0.4,
        mirror=-1
    )

    ax3d.scatter(
        start[0, 0],
        start[1, 0],
        10,
        color='blue',
        s=80,
        label='Start'
    )

    ax3d.scatter(
        goal[0, 0],
        goal[1, 0],
        10,
        color='green',
        s=100,
        label='Goal'
    )

    ax3d.set_title("Full-State NMPC")
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('-Z (Altitude) [m]')
    ax3d.legend()
    
    # 2D trajectory comparison with obstacles
    plt.figure(figsize=(8, 8))
    ax2d = plt.gca()

    draw_2d_obstacles(ax2d, base_obstacles)

    plt.plot(
        ref_trajectory[0, :N_traj],
        ref_trajectory[1, :N_traj],
        "g--",
        linewidth=2,
        label="B-spline reference"
    )

    # hist = np.array(traj_history)
    plt.plot(
        hist[:, 0],
        hist[:, 1],
        "b-",
        linewidth=2,
        label="Flatness actual trajectory"
    )

    plt.scatter(ref_trajectory[0, 0], ref_trajectory[1, 0], color='blue', s=80, label='Start')
    plt.scatter(ref_trajectory[0, -1], ref_trajectory[1, -1], color='green', s=100, label='Goal')

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("2D trajectory comparison with obstacles")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    # Trahectory Tracking 3D Error Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_arr, np.linalg.norm(error_arr, axis=1), 'r-', label='Position Tracking Error (m)')
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error (m)')
    plt.title('Position Tracking Error Over Time')
    plt.grid()
    plt.legend()
    plt.show()
    
    # Control Command Evolution Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 12), sharex=True)
    rotor_labels = ['Rotor 1', 'Rotor 2', 'Rotor 3', 'Rotor 4']
    for i in range(4):
        axs[i//2, i%2].plot(time_arr, cmd_arr[:, i], label=f'{rotor_labels[i]} Command')
        axs[i//2, i%2].set_ylabel('Command (rad/s)')
        axs[i//2, i%2].set_title(f'{rotor_labels[i]} Command Evolution')
        # plot lines with control limits
        
        axs[i//2, i%2].grid()
        axs[i//2, i%2].legend()
    axs[1, 0].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    
    metric_pos_arr = np.array(metric_pos_history)
    metric_ref_arr = np.array(metric_ref_history)

    metrics = print_tracking_metrics(
        "NMPC",
        metric_pos_arr,
        metric_ref_arr,
        cmd_arr,
        w_min=0.0,
        w_max=w_max
    )

if __name__ == "__main__":
    main()
