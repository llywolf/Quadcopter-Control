import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.io import loadmat
import time as timer
import csv
import numpy as np

from roblib import clean3D

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

ZD = 10.0

@dataclass
class QuadcopterParams:
    # nano-quad parameters (Crazyflie 2.1)
    # m: float = 0.028
    # g: float = 9.81

    # Jx: float = 1.4e-5
    # Jy: float = 1.4e-5
    # Jz: float = 2.2e-5
    
    # custom drone parameters 
    # m: float = 10.0
    # g: float = 9.81

    # Jx: float = 10.0
    # Jy: float = 10.0
    # Jz: float = 20.0
    
    # new params
    m: float = 1.3269
    g: float = 9.81

    Jx: float = 0.01295
    Jy: float = 0.01244
    Jz: float = 0.01571

    @property
    def J(self):
        return np.diag([self.Jx, self.Jy, self.Jz])
    
    
def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
    
# attitude controller

def attitude_controller(ref, angles, angle_rates, Kp, Kd):
    """
    ref = [
        phi_d, theta_d, psi_d,
        phi_dot_d, theta_dot_d, psi_dot_d,
        phi_ddot_d, theta_ddot_d, psi_ddot_d
    ]

    angles = [phi, theta, psi]
    angle_rates = [phi_dot, theta_dot, psi_dot]
    """

    ref = np.asarray(ref, dtype=float)
    angles = np.asarray(angles, dtype=float)
    angle_rates = np.asarray(angle_rates, dtype=float)

    Kp = np.asarray(Kp, dtype=float)
    Kd = np.asarray(Kd, dtype=float)

    if Kp.ndim == 1:
        Kp = np.diag(Kp)

    if Kd.ndim == 1:
        Kd = np.diag(Kd)

    eta_d = ref[0:3]
    eta_dot_d = ref[3:6]
    eta_ddot_d = ref[6:9]

    e_eta = eta_d - angles
    e_eta[2] = wrap_to_pi(e_eta[2])
    e_eta_dot = eta_dot_d - angle_rates

    sigma = eta_ddot_d + Kd @ e_eta_dot + Kp @ e_eta

    return sigma


# low-level torque mapping

def attitude_mapping(sigma, angles, angle_rates, params):
    sigma = np.asarray(sigma, dtype=float)
    angles = np.asarray(angles, dtype=float)
    angle_rates = np.asarray(angle_rates, dtype=float)

    phi = angles[0]
    theta = angles[1]

    phi_dot = angle_rates[0]
    theta_dot = angle_rates[1]

    J = params.J

    W = np.array([
        [1.0, 0.0, -np.sin(theta)],
        [0.0, np.cos(phi), np.sin(phi) * np.cos(theta)],
        [0.0, -np.sin(phi), np.cos(phi) * np.cos(theta)]
    ], dtype=float)

    W_dot = np.array([
        [0.0, 0.0, -theta_dot * np.cos(theta)],

        [
            0.0,
            -phi_dot * np.sin(phi),
            phi_dot * np.cos(phi) * np.cos(theta)
            - theta_dot * np.sin(theta) * np.sin(phi)
        ],

        [
            0.0,
            -phi_dot * np.cos(phi),
            -phi_dot * np.sin(phi) * np.cos(theta)
            - theta_dot * np.sin(theta) * np.cos(phi)
        ]
    ], dtype=float)

    omega = W @ angle_rates

    tau = (
        J @ W @ sigma
        + J @ W_dot @ angle_rates
        + np.cross(omega, J @ omega)
    )

    return tau


# low-level controller subsystem

def low_level_attitude_control(ref, angles, angle_rates, Kp_att, Kd_att, params):
    sigma = attitude_controller(
        ref=ref,
        angles=angles,
        angle_rates=angle_rates,
        Kp=Kp_att,
        Kd=Kd_att
    )

    tau = attitude_mapping(
        sigma=sigma,
        angles=angles,
        angle_rates=angle_rates,
        params=params
    )
    
    tau = np.clip(tau, -MAX_TAU, MAX_TAU)

    return tau, sigma



# high-level position controller

def position_controller(ref, position, velocity, Kp, Kd):
    """
    ref = [
        x_ref, y_ref, z_ref,
        x_dot_ref, y_dot_ref, z_dot_ref,
        x_ddot_ref, y_ddot_ref, z_ddot_ref
    ]

    position = [x, y, z]
    velocity = [x_dot, y_dot, z_dot]
    """

    ref = np.asarray(ref, dtype=float)
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    Kp = np.asarray(Kp, dtype=float)
    Kd = np.asarray(Kd, dtype=float)

    if Kp.ndim == 1:
        Kp = np.diag(Kp)

    if Kd.ndim == 1:
        Kd = np.diag(Kd)

    pos_ref = ref[0:3]
    vel_ref = ref[3:6]
    acc_ref = ref[6:9]

    e_pos = pos_ref - position
    e_vel = vel_ref - velocity

    virtual_input = acc_ref + Kd @ e_vel + Kp @ e_pos

    return virtual_input


# high-level feedback-linearization mapping


def position_mapping(virtual_input, psi, params):
    """
    Maps virtual input v = [u1, u2, u3]
    to real thrust T and desired control angles [phi_d, theta_d].

    Inputs:
        virtual_input = [u1, u2, u3]
        psi           = current yaw angle
        params        = QuadcopterParams object

    Outputs:
        T
        control_angles = [phi_d, theta_d]
    """

    virtual_input = np.asarray(virtual_input, dtype=float)

    u1 = virtual_input[0]
    u2 = virtual_input[1]
    u3 = virtual_input[2]

    m = params.m
    g = params.g

    T_raw = m * np.sqrt(u1**2 + u2**2 + (u3 + g)**2)

    asin_arg = (u1 * np.sin(psi) - u2 * np.cos(psi)) * m / max(T_raw, 1e-9)
    asin_arg = np.clip(asin_arg, -1.0, 1.0)

    phi_d = np.arcsin(asin_arg)

    theta_d = np.arctan2(
        u1 * np.cos(psi) + u2 * np.sin(psi),
        u3 + g
    )

    phi_d = np.clip(phi_d, -MAX_TILT, MAX_TILT)
    theta_d = np.clip(theta_d, -MAX_TILT, MAX_TILT)

    T_min = MIN_THRUST_FACTOR * m * g
    T_max = MAX_THRUST_FACTOR * m * g

    T = np.clip(T_raw, T_min, T_max)

    control_angles = np.array([phi_d, theta_d], dtype=float)

    return T, control_angles


# complete high-level position control subsystem

def high_level_position_control(ref, position, velocity, angles, Kp_pos, Kd_pos, params):
    """
    High-level controller.

    Inputs:
        ref:
            [
                x_ref, y_ref, z_ref,
                x_dot_ref, y_dot_ref, z_dot_ref,
                x_ddot_ref, y_ddot_ref, z_ddot_ref
            ]

        position:
            [x, y, z]

        velocity:
            [x_dot, y_dot, z_dot]

        angles:
            [phi, theta, psi]

        Kp_pos:
            position proportional gains

        Kd_pos:
            position derivative gains

        params:
            quadcopter parameters

    Outputs:
        T:
            thrust

        control_angles:
            [phi_d, theta_d]

        virtual_input:
            [v1, v2, v3]
    """

    psi = angles[2]

    virtual_input = position_controller(
        ref=ref,
        position=position,
        velocity=velocity,
        Kp=Kp_pos,
        Kd=Kd_pos
    )

    T, control_angles = position_mapping(
        virtual_input=virtual_input,
        psi=psi,
        params=params
    )

    return T, control_angles, virtual_input

    


# translation dynamics

def translation_acceleration(T, angles, params):
    """
    Computes translational acceleration.

    Inputs:
        T      : thrust [N]
        angles : [phi, theta, psi] [rad]

    Output:
        Acc : [xdd, ydd, zdd]
    """

    phi, theta, psi = angles

    m = params.m
    g = params.g

    xdd = (T / m) * (
        np.cos(phi) * np.sin(theta) * np.cos(psi)
        + np.sin(phi) * np.sin(psi)
    )

    ydd = (T / m) * (
        np.cos(phi) * np.sin(theta) * np.sin(psi)
        - np.sin(phi) * np.cos(psi)
    )

    zdd = (T / m) * (
        np.cos(phi) * np.cos(theta)
    ) - g

    return np.array([xdd, ydd, zdd], dtype=float)


# rotation dynamics

def rotation_acceleration(tau, angles, angle_rates, params):
    """
    Computes angular acceleration.

    Inputs:
        tau         : [tau_phi, tau_theta, tau_psi] [Nm]
        angles      : [phi, theta, psi] [rad]
        angle_rates : [phid, thetad, psid] [rad/s]

    Output:
        angular_acc : [phidd, thetadd, psidd] [rad/s^2]
    """

    tau = np.asarray(tau, dtype=float)
    angles = np.asarray(angles, dtype=float)
    angle_rates = np.asarray(angle_rates, dtype=float)

    phi = angles[0]
    theta = angles[1]

    phid = angle_rates[0]
    thetad = angle_rates[1]

    J = params.J

    W = np.array([
        [1.0, 0.0, -np.sin(theta)],
        [0.0, np.cos(phi), np.sin(phi) * np.cos(theta)],
        [0.0, -np.sin(phi), np.cos(phi) * np.cos(theta)]
    ], dtype=float)

    Wd = np.array([
        [0.0, 0.0, -thetad * np.cos(theta)],

        [
            0.0,
            -phid * np.sin(phi),
            phid * np.cos(phi) * np.cos(theta)
            - thetad * np.sin(theta) * np.sin(phi)
        ],

        [
            0.0,
            -phid * np.cos(phi),
            -phid * np.sin(phi) * np.cos(theta)
            - thetad * np.sin(theta) * np.cos(phi)
        ]
    ], dtype=float)

    eta_dot = angle_rates

    omega = W @ eta_dot

    rhs = (
        tau
        - J @ Wd @ eta_dot
        - np.cross(omega, J @ omega)
    )

    # MATLAB inv(J*W)*rhs, 
    angular_acc = np.linalg.solve(J @ W, rhs)

    return angular_acc


# quadcopter dynamics

def quadcopter_dynamics(t, state, control_input, params):
    """
    Quadcopter model

    State vector:
        state = [
            x, y, z,
            xd, yd, zd,
            phi, theta, psi,
            phid, thetad, psid
        ]

    Control input:
        control_input = [
            T,
            tau_phi, tau_theta, tau_psi
        ]

    Output:
        state_dot with same structure as state
    """

    state = np.asarray(state, dtype=float)
    control_input = np.asarray(control_input, dtype=float)

    # State extraction
    position = state[0:3]
    velocity = state[3:6]

    angles = state[6:9]
    angle_rates = state[9:12]

    # Input extraction
    T = control_input[0]
    tau = control_input[1:4]

    # Accelerations
    linear_acc = translation_acceleration(T, angles, params)
    angular_acc = rotation_acceleration(tau, angles, angle_rates, params)

    # State derivative
    state_dot = np.zeros(12)

    # Position derivative = velocity
    state_dot[0:3] = velocity

    # Velocity derivative = translational acceleration
    state_dot[3:6] = linear_acc

    # Angle derivative = angle rates
    state_dot[6:9] = angle_rates

    # Angle-rate derivative = angular acceleration
    state_dot[9:12] = angular_acc

    return state_dot


# rk4 integrator for simulation

def rk4_step(f, t, state, dt, control_input, params):
    k1 = f(t, state, control_input, params)
    k2 = f(t + dt / 2, state + dt * k1 / 2, control_input, params)
    k3 = f(t + dt / 2, state + dt * k2 / 2, control_input, params)
    k4 = f(t + dt, state + dt * k3, control_input, params)

    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


B_SPLINE_T_FINAL = 30.0
HOVER_TIME = 3.0
PATH_TIME = B_SPLINE_T_FINAL - HOVER_TIME
B_SPLINE_EPS = 1e-4

MAX_TILT = np.deg2rad(25.0)
MAX_THRUST_FACTOR = 2.2
MIN_THRUST_FACTOR = 0.2

MAX_TAU = np.array([0.35, 0.35, 0.20])


def smooth_progress(t):
    """
    Smooth progress from 0 to 1.

    s_dot = 0 at start and end
    s_ddot = 0 at start and end
    Prevents sudden acceleration near the goal.
    """

    r = np.clip(t / PATH_TIME, 0.0, 1.0)

    s = 10.0*r**3 - 15.0*r**4 + 6.0*r**5
    dsdt = (30.0*r**2 - 60.0*r**3 + 30.0*r**4) / PATH_TIME
    d2sdt2 = (60.0*r - 180.0*r**2 + 120.0*r**3) / PATH_TIME**2

    return s, dsdt, d2sdt2


def final_goal_position():
    xy_goal, _ = eval_spline_manual(s_max)

    return np.array([
        xy_goal[0],
        xy_goal[1],
        ZD
    ], dtype=float)


def hover_reference():
    goal_pos = final_goal_position()

    vel_ref = np.zeros(3)
    acc_ref = np.zeros(3)

    return np.hstack((goal_pos, vel_ref, acc_ref))

# control system simulation

def reference_trajectory(t):
    """
    B-spline reference followed during PATH_TIME seconds.
    Last HOVER_TIME seconds are a hover reference:
    fixed position, zero velocity, zero acceleration.
    """

    if t >= PATH_TIME:
        return hover_reference()

    s, dsdt, d2sdt2 = smooth_progress(t)

    xy, dxy_ds = eval_spline_manual(s)

    s_plus = min(s + B_SPLINE_EPS, s_max)
    s_minus = max(s - B_SPLINE_EPS, s_min)

    _, dxy_ds_plus = eval_spline_manual(s_plus)
    _, dxy_ds_minus = eval_spline_manual(s_minus)

    if s_plus == s_minus:
        d2xy_ds2 = np.array([0.0, 0.0])
    else:
        d2xy_ds2 = (dxy_ds_plus - dxy_ds_minus) / (s_plus - s_minus)

    pos_ref = np.array([
        xy[0],
        xy[1],
        ZD
    ])

    vel_ref = np.array([
        dxy_ds[0] * dsdt,
        dxy_ds[1] * dsdt,
        0.0
    ])

    acc_ref = np.array([
        d2xy_ds2[0] * dsdt**2 + dxy_ds[0] * d2sdt2,
        d2xy_ds2[1] * dsdt**2 + dxy_ds[1] * d2sdt2,
        0.0
    ])

    return np.hstack((pos_ref, vel_ref, acc_ref))
    
# def reference_trajectory(t):
    # '''
    # step reference trajectory
    
    # '''
    # ref_pos = np.zeros(9)

    # for i in range(9):
    #     ref_pos[i] = np.interp(t, t_ref, pos_ref_data[:, i])

    # return ref_pos


def yaw_from_path(s):
    ds = 5e-3

    s_a = max(s - ds, s_min)
    s_b = min(s + ds, s_max)

    p_a, _ = eval_spline_manual(s_a)
    p_b, _ = eval_spline_manual(s_b)

    direction = p_b - p_a

    if np.linalg.norm(direction) < 1e-8:
        return 0.0

    return np.arctan2(direction[1], direction[0])


def yaw_reference(t):
    """
    Desired yaw generated from path direction.

    During hover,  yaw is constant.
    """

    if t >= PATH_TIME:
        s = max(s_max - 5e-3, s_min)
    else:
        s, _, _ = smooth_progress(t)

    psi_d = yaw_from_path(s)

    psi_dot_d = 0.0
    psi_ddot_d = 0.0

    return psi_d, psi_dot_d, psi_ddot_d

# def yaw_reference(t):
#     '''
#     yaw for step reference 
    
#     '''
    
#     psi_d = np.interp(t, t_ref, psi_ref_data)
#     psi_dot_d = np.interp(t, t_ref, psi_dot_ref_data)
#     psi_ddot_d = np.interp(t, t_ref, psi_ddot_ref_data)

#     return psi_d, psi_dot_d, psi_ddot_d


# step reference trajectory data 
trajectory = loadmat("trajectory_1.mat")

desired_position = trajectory["desired_position"]
desired_psi = trajectory["desired_psi"]

t_ref = desired_position[:, 0]

pos_ref_data = desired_position[:, 1:10]
psi_ref_data = desired_psi[:, 1]

psi_dot_ref_data = np.gradient(psi_ref_data, t_ref)
psi_ddot_ref_data = np.gradient(psi_dot_ref_data, t_ref)

def draw_2d_obstacles(ax, obstacles):
    for obs in obstacles:
        V = obs["vertices"]
        V_closed = np.vstack((V, V[0]))

        ax.plot(V_closed[:, 0], V_closed[:, 1], color="red", linewidth=2)
        ax.fill(V_closed[:, 0], V_closed[:, 1], color="red", alpha=0.25)


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


def thrust_torque_to_motor_commands(T, tau):
    """
    Converts [T, tau_phi, tau_theta, tau_psi]
    into equivalent rotor commands for plotting.

    """
    # for old params
    # b = 2.0
    # d = 1.0
    # l = 1.0
    
    # for new params
    b = 3.15e-5
    d = 1
    l = 0.25

    B = np.array([
        [b,      b,      b,      b],
        [-b*l,   0.0,    b*l,    0.0],
        [0.0,   -b*l,    0.0,    b*l],
        [-d,     d,     -d,      d]
    ], dtype=float)

    u = np.array([
        T,
        tau[0],
        tau[1],
        tau[2]
    ], dtype=float)

    # Equivalent to W2 = inv(B) * [T; tau]
    motor_squared = np.linalg.solve(B, u)

    # w2 = w * abs(w)
    # motor_commands = np.sqrt(np.abs(motor_squared)) * np.sign(motor_squared)
    # for w >= 0
    motor_commands = np.sqrt(np.maximum(motor_squared, 0.0))

    return motor_commands

def simulate_complete_control_system():
    params = QuadcopterParams()

    dt = 0.001
    # t_final = t_ref[-1]
    t_final = B_SPLINE_T_FINAL
    time = np.arange(0.0, t_final + dt, dt)
    
    #rotor plotting storage
    motor_commands = np.zeros((len(time), 4))

    # State:
    # [x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot]
    state = np.zeros(12)

    states = np.zeros((len(time), 12))
    inputs = np.zeros((len(time), 4))

    refs_pos = np.zeros((len(time), 9))
    refs_att = np.zeros((len(time), 9))

    virtual_inputs = np.zeros((len(time), 3))
    sigmas = np.zeros((len(time), 3))
    control_times = []
    
     # initial position
    initial_ref = reference_trajectory(0.0)
    state[0:3] = initial_ref[0:3]

    psi_d0, _, _ = yaw_reference(0.0)
    state[8] = psi_d0

    # Position controller gains
    # These act on:
    # x_ddot = v1, y_ddot = v2, z_ddot = v3
    Kp_pos = np.array([0.8, 0.8, 0.8]) * 3.5 # 5
    Kd_pos = np.array([1, 1, 1]) * 3.5 # 5

    # Attitude controller gains
    # These act on:
    # phi_ddot = sigma1, theta_ddot = sigma2, psi_ddot = sigma3
    # Attitude loop should be faster than position loop.
    Kp_att = np.array([10.0, 10.0, 10.0]) * 100 # 50
    Kd_att = np.array([10.0, 10.0, 10.0]) * 20 # 10

    for k, t in enumerate(time):
        position = state[0:3]
        velocity = state[3:6]
        angles = state[6:9]
        angle_rates = state[9:12]

        # High-level position controller

        ref_pos = reference_trajectory(t)

        t0 = timer.perf_counter()
        T, control_angles, virtual_input = high_level_position_control(
            ref=ref_pos,
            position=position,
            velocity=velocity,
            angles=angles,
            Kp_pos=Kp_pos,
            Kd_pos=Kd_pos,
            params=params
        )

        phi_d = control_angles[0]
        theta_d = control_angles[1]

        # User-defined desired yaw

        psi_d, psi_dot_d, psi_ddot_d = yaw_reference(t)

        # The high-level controller computes only phi_d and theta_d.
        # The desired psi is set by the user and passed directly here.
        ref_attitude = np.array([
            phi_d, theta_d, psi_d,
            0.0, 0.0, psi_dot_d,
            0.0, 0.0, psi_ddot_d
        ])

        # Low-level attitude controller

        tau, sigma = low_level_attitude_control(
            ref=ref_attitude,
            angles=angles,
            angle_rates=angle_rates,
            Kp_att=Kp_att,
            Kd_att=Kd_att,
            params=params
        )
        
        motor_cmd = thrust_torque_to_motor_commands(T, tau)

        control_input = np.array([
            T,
            tau[0],
            tau[1],
            tau[2]
        ])
        control_times.append(timer.perf_counter() - t0)

        # Save data
        states[k, :] = state
        inputs[k, :] = control_input
        refs_pos[k, :] = ref_pos
        refs_att[k, :] = ref_attitude
        virtual_inputs[k, :] = virtual_input
        sigmas[k, :] = sigma
        motor_commands[k, :] = motor_cmd

        # Integrate model
        state = rk4_step(
            quadcopter_dynamics,
            t,
            state,
            dt,
            control_input,
            params
        )

    return time, states, inputs, refs_pos, refs_att, virtual_inputs, sigmas, motor_commands, control_times


# Plot complete control system results

def plot_complete_control_results(time, states, inputs, refs_pos, refs_att, virtual_inputs, sigmas, motor_commands):
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]

    x_ref = refs_pos[:, 0]
    y_ref = refs_pos[:, 1]
    z_ref = refs_pos[:, 2]

    phi = states[:, 6]
    theta = states[:, 7]
    psi = states[:, 8]

    phi_ref = refs_att[:, 0]
    theta_ref = refs_att[:, 1]
    psi_ref = refs_att[:, 2]

    T = inputs[:, 0]
    tau_phi = inputs[:, 1]
    tau_theta = inputs[:, 2]
    tau_psi = inputs[:, 3]

    # Position tracking

    plt.figure()
    plt.plot(time, x, label="x")
    plt.plot(time, x_ref, "--", label="x_ref")
    plt.xlabel("Time [s]")
    plt.ylabel("x [m]")
    plt.title("X position tracking")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(time, y, label="y")
    plt.plot(time, y_ref, "--", label="y_ref")
    plt.xlabel("Time [s]")
    plt.ylabel("y [m]")
    plt.title("Y position tracking")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(time, z, label="z")
    plt.plot(time, z_ref, "--", label="z_ref")
    plt.xlabel("Time [s]")
    plt.ylabel("z [m]")
    plt.title("Z position tracking")
    plt.grid(True)
    plt.legend()

    # Attitude tracking

    plt.figure()
    plt.plot(time, np.rad2deg(phi), label="phi")
    plt.plot(time, np.rad2deg(phi_ref), "--", label="phi_ref")
    plt.xlabel("Time [s]")
    plt.ylabel("Roll angle [deg]")
    plt.title("Roll tracking")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(time, np.rad2deg(theta), label="theta")
    plt.plot(time, np.rad2deg(theta_ref), "--", label="theta_ref")
    plt.xlabel("Time [s]")
    plt.ylabel("Pitch angle [deg]")
    plt.title("Pitch tracking")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(time, np.rad2deg(psi), label="psi")
    plt.plot(time, np.rad2deg(psi_ref), "--", label="psi_ref")
    plt.xlabel("Time [s]")
    plt.ylabel("Yaw angle [deg]")
    plt.title("Yaw tracking")
    plt.grid(True)
    plt.legend()

    # Inputs

    plt.figure()
    plt.plot(time, T, label="T")
    plt.xlabel("Time [s]")
    plt.ylabel("Thrust [N]")
    plt.title("Thrust input")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(time, tau_phi, label="tau_phi")
    plt.plot(time, tau_theta, label="tau_theta")
    plt.plot(time, tau_psi, label="tau_psi")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.title("Torque inputs")
    plt.grid(True)
    plt.legend()

    # Virtual inputs

    plt.figure()
    plt.plot(time, virtual_inputs[:, 0], label="v1")
    plt.plot(time, virtual_inputs[:, 1], label="v2")
    plt.plot(time, virtual_inputs[:, 2], label="v3")
    plt.xlabel("Time [s]")
    plt.ylabel("Virtual acceleration [m/s²]")
    plt.title("High-level virtual control inputs")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(time, sigmas[:, 0], label="sigma_phi")
    plt.plot(time, sigmas[:, 1], label="sigma_theta")
    plt.plot(time, sigmas[:, 2], label="sigma_psi")
    plt.xlabel("Time [s]")
    plt.ylabel("Virtual angular acceleration [rad/s²]")
    plt.title("Low-level virtual control inputs")
    plt.grid(True)
    plt.legend()

    # 3D trajectory

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    clean3D(ax, 0, 20, 0, 20, 0, 20)
    ax.plot(x, y, z,"b-" ,label="trajectory")
    # ax.plot(x_ref, y_ref, z_ref, "g--", label="reference")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Flatness PID controller")
    ax.legend()
    ax.grid(True)
    
    #obstacle plotting
    draw_obstacles_3d(ax, base_obstacles)

    ref_curve = sample_reference_curve(
        z_value=ZD,
        num_points=500
    )
    ax.plot(
        ref_curve[:, 0],
        ref_curve[:, 1],
        ref_curve[:, 2],
        "g--",
        linewidth=2,
        label="B-spline reference"
    )

    ax.scatter(ref_curve[0, 0], ref_curve[0, 1], ref_curve[0, 2], color='blue', s=80, label="Start")
    ax.scatter(ref_curve[-1, 0], ref_curve[-1, 1], ref_curve[-1, 2], color='green', s=100, label="Goal")
    
    ref_curve = sample_reference_curve(
        z_value=ZD,
        num_points=500
    )

    # 2D trajectory comparison with obstacles
    plt.figure(figsize=(8, 8))
    ax2d = plt.gca()

    draw_2d_obstacles(ax2d, base_obstacles)

    plt.plot(
        ref_curve[:, 0],
        ref_curve[:, 1],
        "g--",
        linewidth=2,
        label="B-spline reference"
    )

    plt.plot(
        states[:, 0],
        states[:, 1],
        "b-",
        linewidth=2,
        label="Flatness actual trajectory"
    )

    plt.scatter(ref_curve[0, 0], ref_curve[0, 1], color='blue', s=80, label='Start')
    plt.scatter(ref_curve[-1, 0], ref_curve[-1, 1], color='green', s=100, label='Goal')

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("2D trajectory comparison with obstacles")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    
    
    # position tracking error

    pos_error = states[:, 0:3] - refs_pos[:, 0:3]
    tracking_error = np.linalg.norm(pos_error, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(time, tracking_error, color='red', label="Position tracking error")
    plt.xlabel("Time [s]")
    plt.ylabel("Tracking error [m]")
    plt.title("Position Tracking Error Over Time")
    plt.grid(True)
    plt.legend()
    
    # motor command evolution

    fig_cmd, axs = plt.subplots(2, 2, figsize=(10, 12), sharex=True)

    rotor_labels = ["Rotor 1", "Rotor 2", "Rotor 3", "Rotor 4"]

    for i in range(4):
        axs[i // 2, i % 2].plot(
            time,
            motor_commands[:, i],
            label=f"{rotor_labels[i]} command"
        )

        axs[i // 2, i % 2].set_ylabel("Command (rad/s)")
        axs[i // 2, i % 2].set_title(f"{rotor_labels[i]} Command Evolution")
        axs[i // 2, i % 2].grid(True)
        axs[i // 2, i % 2].legend()
        # plot lines with control limits
        

    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 1].set_xlabel("Time [s]")

    fig_cmd.suptitle("Equivalent Rotor Commands from Thrust and Torques")
    fig_cmd.tight_layout()

    plt.show()

if __name__ == "__main__":
    time, states, inputs, refs_pos, refs_att, virtual_inputs, sigmas, motor_commands, control_times = simulate_complete_control_system()

    plot_complete_control_results(
        time,
        states,
        inputs,
        refs_pos,
        refs_att,
        virtual_inputs,
        sigmas,
        motor_commands
    )
    
    metrics = print_tracking_metrics("Flatness", states[:, 0:3], refs_pos[:, 0:3], motor_commands)
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