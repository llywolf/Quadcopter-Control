from roblib import *  # Ensure this is in your Python path or in the same directory
import time
import csv
import numpy as np

from metrics import print_tracking_metrics

from scenario_obstacles import load_base_obstacles
from initial_obstacles import load_initial_base_obstacles

from reference_selector import load_reference

# USE_RANDOM_REFERENCE = True
USE_RANDOM_REFERENCE = False

reference = load_reference(
    use_random_reference=USE_RANDOM_REFERENCE
)

s_min = reference.s_min
s_max = reference.s_max
eval_spline_manual = reference.eval_spline_manual
sample_reference_curve = reference.sample_reference_curve

# USE_RANDOM_OBSTACLES = True
USE_RANDOM_OBSTACLES = False

if USE_RANDOM_OBSTACLES:
    base_obstacles = load_base_obstacles()
else:
    base_obstacles = load_initial_base_obstacles()

ZD = -10.0
s_current = s_min

ref_curve = sample_reference_curve(
    z_value=ZD,
    num_points=500
)

LOOKAHEAD_S = 0.015
SEARCH_WINDOW_S = 0.12
K_PATH = 2
VD = 1.5

traj_history = []
time_history = []
error_history = []
command_history = []
ref_history = []
ref_time_history = []
control_times = []

def draw_reference_and_actual(ax, path):
    ax.plot(
        ref_curve[:, 0],
        ref_curve[:, 1],
        -ref_curve[:, 2],
        'g--',
        linewidth=2,
        label='B-Spline Reference'
    )

    if len(path) > 1:
        hist = np.array(path)
        ax.plot(
            hist[:, 0],
            hist[:, 1],
            -hist[:, 2],
            'b-',
            linewidth=2,
            label='Actual Flight Path'
        )

    ax.scatter(
        ref_curve[0, 0],
        ref_curve[0, 1],
        -ref_curve[0, 2],
        color='blue',
        s=80,
        label='Start'
    )

    ax.scatter(
        ref_curve[-1, 0],
        ref_curve[-1, 1],
        -ref_curve[-1, 2],
        color='green',
        s=100,
        label='Goal'
    )

# Initialize the figure and 3D plot
fig = figure()
ax = fig.add_subplot(111, projection = '3d')

# Define physical parameters for the quadrotor
# m, g, b, d, l = 10, 9.81, 2, 1, 1
# I = array([[10, 0, 0], [0, 10, 0], [0, 0, 20]])

m, g, b, d, l = 1.3269, 9.81, 3.15e-5, 1, 0.25
I = array([[0.01295, 0, 0], [0, 0.01244, 0], [0, 0, 0.01571]])

dt = 0.01
B = array([[b, b, b, b], [-b * l, 0, b * l, 0], [0, -b * l, 0, b * l], [-d, d, -d, d]])

# to convert spline parameter s to time
T_REF = 30.0       # time used to move from start to goal
T_HOVER = 5.0      # extra time holding the final goal position

time_ref = np.arange(0, T_REF + T_HOVER, dt)

Nref = int(T_REF / dt)
Nhover = len(time_ref) - Nref
Nsim = len(time_ref)

def build_time_indexed_reference(Nref, Nhover, z_value):
    s_values = np.linspace(s_min, s_max, Nref)

    ref_pos_move = np.zeros((Nref, 3))
    ref_tangent_move = np.zeros((Nref, 2))

    for i, s in enumerate(s_values):
        xy, tangent = eval_spline_manual(s)

        ref_pos_move[i, 0] = xy[0]
        ref_pos_move[i, 1] = xy[1]
        ref_pos_move[i, 2] = z_value

        ref_tangent_move[i, 0] = tangent[0]
        ref_tangent_move[i, 1] = tangent[1]

    # Hold final position for a few seconds
    final_pos = ref_pos_move[-1].copy()
    final_tangent = ref_tangent_move[-1].copy()

    ref_pos_hover = np.tile(final_pos, (Nhover, 1))
    ref_tangent_hover = np.tile(final_tangent, (Nhover, 1))

    ref_pos = np.vstack((ref_pos_move, ref_pos_hover))
    ref_tangent = np.vstack((ref_tangent_move, ref_tangent_hover))

    ref_vel = np.gradient(ref_pos, dt, axis=0)

    return ref_pos, ref_vel, ref_tangent


ref_pos, ref_vel, ref_tangent = build_time_indexed_reference(
    Nref,
    Nhover,
    ZD
)

def draw_2d_obstacles(ax, obstacles):
    for obs in obstacles:
        V = obs["vertices"]
        V_closed = np.vstack((V, V[0]))

        ax.plot(
            V_closed[:, 0],
            V_closed[:, 1],
            color="red",
            linewidth=2
        )

        ax.fill(
            V_closed[:, 0],
            V_closed[:, 1],
            color="red",
            alpha=0.25
        )


def draw_obstacles_3d(ax, obstacles):
    for obs in obstacles:
        V = obs["vertices"]
        V_closed = np.vstack((V, V[0]))

        # Top and bottom edges
        ax.plot(V_closed[:, 0], V_closed[:, 1], 8, color="red", alpha=0.6)
        ax.plot(V_closed[:, 0], V_closed[:, 1], 12, color="red", alpha=0.6)

        # Vertical edges
        for v in V:
            ax.plot(
                [v[0], v[0]],
                [v[1], v[1]],
                [8, 12],
                color="red",
                alpha=0.6
            )

# Initialize list to store the path of the drone
path = []


# Define the clock function to update the quadrotor state
def clock_quadri_rk2(p_in, R_in, vr_in, wr_in, w_in, B_mat, I, g, m, dt):
        def dynamics(p_loc, R_loc, vr_loc, wr_loc, w_loc):
            w2_loc = w_loc * np.abs(w_loc)
            tau_loc = B_mat @ w2_loc.flatten()
            p_dot = R_loc @ vr_loc
            vr_dot = -adjoint(wr_loc) @ vr_loc + np.linalg.inv(R_loc) @ np.array([[0],[0],[g]]) + np.array([[0],[0],[-tau_loc[0]/m]])
            wr_dot = np.linalg.inv(I) @ (-adjoint(wr_loc) @ I @ wr_loc + tau_loc[1:4].reshape(3,1))
            return p_dot, vr_dot, wr_dot

        k1_p, k1_vr, k1_wr = dynamics(p_in, R_in, vr_in, wr_in, w_in)
        p_mid = p_in + (2/3) * dt * k1_p
        vr_mid = vr_in + (2/3) * dt * k1_vr
        wr_mid = wr_in + (2/3) * dt * k1_wr
        R_mid = R_in @ expw((2/3) * dt * wr_in)

        k2_p, k2_vr, k2_wr = dynamics(p_mid, R_mid, vr_mid, wr_mid, w_in)
        p_out = p_in + dt * (0.25 * k1_p + 0.75 * k2_p)
        vr_out = vr_in + dt * (0.25 * k1_vr + 0.75 * k2_vr)
        wr_out = wr_in + dt * (0.25 * k1_wr + 0.75 * k2_wr)
        wr_eff = 0.25 * wr_in + 0.75 * wr_mid
        R_out = R_in @ expw(dt * wr_eff)
        return p_out, R_out, vr_out, wr_out

def f_vdp(x):
    global s_current

    xy = x.flatten()

    s_search = np.linspace(
        s_current,
        min(s_current + SEARCH_WINDOW_S, s_max),
        100
    )

    positions = np.array([eval_spline_manual(s)[0] for s in s_search])
    distances = np.linalg.norm(positions - xy, axis=1)

    best_idx = np.argmin(distances)
    s_current = s_search[best_idx]

    s_target = min(s_current + LOOKAHEAD_S, s_max)

    target_pos, tangent = eval_spline_manual(s_target)

    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > 1e-6:
        tangent_dir = tangent / tangent_norm
    else:
        tangent_dir = np.array([1.0, 0.0])

    correction = K_PATH * (target_pos - xy)

    fd = tangent_dir + correction

    fd_norm = np.linalg.norm(fd)
    if fd_norm > 1e-6:
        fd = fd / fd_norm

    vdp0 = fd[0]
    vdp1 = fd[1]

    return array([[vdp0], [vdp1]])


# gains
KP_td0 = 40 # 300
KP_td13 = 30 # 100 for old params
KP_angles = 5 # 5 for old params
KP_vdp = 11 # 80 for old params

clip_value = 35 # 250 for old params

# Define the control function to calculate rotor speeds
def control(X):
    X = X.flatten()
    x, y, z, φ, θ, ψ = list(X[0:6])
    vr = X[6:9].reshape(3, 1)
    wr = X[9:12].reshape(3, 1)
    E = eulermat(φ, θ, ψ)
    dp = E @ vr

    zd = ZD
    vd = VD
    fd = f_vdp(array([[x], [y]]))
    
    # Desired states
    ez = z - zd
    vz = vr[2, 0]

    td0 = m * g + KP_td0 * tanh(0.5 * ez) + KP_vdp * vz
    td0 = np.clip(td0, 0.0, clip_value) # Desired thrust or related control input

    φd = 0.5 * tanh(10 * sawtooth(angle(fd) - angle(dp)))  # Desired roll angle
    
    fd_norm = norm(fd)
    if fd_norm > 1e-6:
        fd_dir = fd / fd_norm
    else:
        fd_dir = array([[1.0], [0.0]])

    v_xy = dp[0:2]
    v_along_path = float((fd_dir.T @ v_xy).item())

    θd = -0.35 * tanh(vd - v_along_path)  # Desired pitch angle
    
    ψd = angle(fd)  # Desired yaw angle
    # ψd = angle(dp)
    # Inverse of Block 3
    wrd = KP_angles * inv(eulerderivative(φ, θ, ψ)) @ array([[float(sawtooth(φd - φ).item())],
                                                     [float(sawtooth(θd - θ).item())],
                                                     [float(sawtooth(ψd - ψ).item())]], dtype = float)

    # Inverse of Block 2
    td13 = I @ ((KP_td13 * (wrd - wr)) + adjoint(wr) @ I @ wr)

    # Inverse of Block 1
    W2 = inv(B) @ vstack(([td0], td13))
    w = sqrt(abs(W2)) * sign(W2)

    return w


# Initialize state variables
# start_xy, _ = eval_spline_poly(s_min) 
start_xy, _ = eval_spline_manual(s_min)
p = array([[start_xy[0]], [start_xy[1]], [ZD]]) # Position: x, y, z (front, right, down)
_, start_tangent = eval_spline_manual(s_min)
start_psi = np.arctan2(start_tangent[1], start_tangent[0])
R = eulermat(0, 0, start_psi)
vr = array([[1], [1], [0]])  # Initial linear velocity
wr = array([[0], [0], [0]])  # Initial angular velocity
α = array([[0, 0, 0, 0]]).T  # Initial angles for the rotor blades

w_prev = None

# DU_MAX = 5      # maximum command change per step
# U_MAX = 10.0       # optional signed command magnitude limit

DU_MAX = 50      # maximum command change per step
U_MAX = 900.0       # optional signed command magnitude limit


# # Simulation loop - 3d plot for each step
# for t in arange(0, 30, dt):
#     X = hstack((p.flatten(), eulermat2angles(R), vr.flatten(), wr.flatten())).reshape(-1, 1)
#     w_cmd = control(X)

#     # Initialize limiter from the first real command
#     if w_prev is None:
#         w = w_cmd.copy()
#     else:
#         # Limit command effort: bounded command variation
#         dw = np.clip(w_cmd - w_prev, -DU_MAX, DU_MAX)
#         w = w_prev + dw

#     # Optional signed magnitude bound, keeps original signed backstepping structure
#     w = np.clip(w, -U_MAX, U_MAX)

#     w_prev = w.copy()

#     p, R, vr, wr = clock_quadri_rk2(p, R, vr, wr, w, B, I, g, m, dt)

#     # Store the position in the path list
#     path.append(p.flatten())
    
#     traj_history.append(p.flatten())
#     time_history.append(t)
#     command_history.append(w.flatten())

#     # ref_xy, _ = eval_spline_poly(s_current)
#     ref_xy, _ = eval_spline_manual(s_current)
#     ref_p = array([[ref_xy[0]], [ref_xy[1]], [ZD]])
#     error_history.append((p - ref_p).flatten())

#     clean3D(ax, 0, 20, 0, 20, 0, 20)
#     draw_obstacles_3d(ax, base_obstacles)
#     draw_reference_and_actual(ax, path)
#     M = np.diag([-1.0, 1.0, 1.0])
#     R_draw = M @ R @ M
    
#     p_draw = p.copy()
#     p_draw[0, 0] = -p_draw[0, 0]

#     draw_quadrotor3D(ax, p_draw, R_draw, α, l=0.4, mirror=-1)
#     α = α + dt * 30 * w
    
#     ax.set_title("Backstepping Controller")
#     ax.set_xlabel("X [m]")
#     ax.set_ylabel("Y [m]")
#     ax.set_zlabel("-Z (Altitude) [m]")
#     ax.legend()
    
#     pause(0.1)
    
#     goal_p = array([[ref_curve[-1, 0]], [ref_curve[-1, 1]], [ZD]])

#     if norm(p - goal_p) < 0.1:
#         print("Goal reached.")
#         break

# Simulation loop - no plotting inside
for k,t in enumerate(time_ref):
    X = hstack((p.flatten(), eulermat2angles(R), vr.flatten(), wr.flatten())).reshape(-1, 1)
    
    t0 = time.perf_counter()
    w_cmd = control(X)
    control_times.append(time.perf_counter() - t0)

    if w_prev is None:
        w = w_cmd.copy()
    else:
        dw = np.clip(w_cmd - w_prev, -DU_MAX, DU_MAX)
        w = w_prev + dw

    w = np.clip(w, -U_MAX, U_MAX)
    w_prev = w.copy()

    p, R, vr, wr = clock_quadri_rk2(p, R, vr, wr, w, B, I, g, m, dt)

    path.append(p.flatten())
    traj_history.append(p.flatten())
    time_history.append(t)
    command_history.append(w.flatten())

    ref_xy, _ = eval_spline_manual(s_current)
    ref_p = array([[ref_xy[0]], [ref_xy[1]], [ZD]])

    ref_history.append(ref_p.flatten())
    error_history.append((p - ref_p).flatten())

    # ref_time_history.append(ref_pos[k])

    α = α + dt * 30 * w

    goal_p = array([[ref_curve[-1, 0]], [ref_curve[-1, 1]], [ZD]])

    if norm(p - goal_p) < 0.1:
        print("Goal reached.")
        break

path = array(path)
time_arr = array(time_history)
error_arr = array(error_history)
cmd_arr = array(command_history)

# Final 3D plot drawn once
# fig = figure()
# ax = fig.add_subplot(111, projection='3d')

clean3D(ax, 0, 20, 0, 20, 0, 20)
draw_obstacles_3d(ax, base_obstacles)
draw_reference_and_actual(ax, path)

M = np.diag([-1.0, 1.0, 1.0])
R_draw = M @ R @ M

p_draw = p.copy()
p_draw[0, 0] = -p_draw[0, 0]

draw_quadrotor3D(ax, p_draw, R_draw, α, l=0.4, mirror=-1)

ax.set_title("Backstepping Controller")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("-Z (Altitude) [m]")
ax.legend()

show()

# 2D trajectory comparison
figure(figsize=(8, 8))
ax2d = gca()
draw_2d_obstacles(ax2d, base_obstacles)
plot(ref_curve[:, 0], ref_curve[:, 1], 'g--', linewidth=2, label='B-Spline Reference')
plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Actual Flight Path')
scatter(ref_curve[0, 0], ref_curve[0, 1], color='blue', s=80, label='Start')
scatter(ref_curve[-1, 0], ref_curve[-1, 1], color='green', s=100, label='Goal')
xlabel('X [m]')
ylabel('Y [m]')
title('2D Trajectory Comparison')
axis('equal')
grid(True)
legend()
show()

# Position tracking error
figure(figsize=(10, 6))
plot(
    time_arr,
    np.linalg.norm(error_arr, axis=1),
    'r-',
    label='Position Tracking Error (m)'
)
xlabel('Time (s)')
ylabel('Tracking Error (m)')
title('Position Tracking Error Over Time')
grid(True)
legend()
show()

# Control command evolution
fig_cmd, axs = subplots(2, 2, figsize=(10, 12), sharex=True)

rotor_labels = ['Rotor 1', 'Rotor 2', 'Rotor 3', 'Rotor 4']

for i in range(4):
    axs[i // 2, i % 2].plot(
        time_arr,
        cmd_arr[:, i],
        label=f'{rotor_labels[i]} Command'
    )

    axs[i // 2, i % 2].set_ylabel('Command (rad/s)')
    axs[i // 2, i % 2].set_title(f'{rotor_labels[i]} Command Evolution')
    axs[i // 2, i % 2].grid()
    axs[i // 2, i % 2].legend()

axs[1, 0].set_xlabel('Time (s)')
axs[1, 1].set_xlabel('Time (s)')
tight_layout()
show()

metrics = print_tracking_metrics("Backstepping", path, ref_history, cmd_arr)
control_times = np.array(control_times)

mean_time_ms = np.mean(control_times) * 1000
max_time_ms = np.max(control_times) * 1000
min_time_ms = np.min(control_times) * 1000
total_control_time_s = np.sum(control_times)

print("\nComputation time results:")
print(f"Mean time per call: {mean_time_ms:.4f} ms")
print(f"Max time per call: {max_time_ms:.4f} ms")
print(f"Min time per call: {min_time_ms:.4f} ms")
print(f"Total computation time: {total_control_time_s:.4f} s")
