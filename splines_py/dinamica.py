import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


# Parameters
l = 2.0
alpha = 1.0
beta = 0.1
c = 0.2

# Symbolic state variables
x = ca.MX.sym("x")
y = ca.MX.sym("y")
eta = ca.MX.sym("eta")
v = ca.MX.sym("v")

x_state = ca.vertcat(x, y, eta, v)

# Inputs
theta = ca.MX.sym("theta")   # throttle
delta = ca.MX.sym("delta")   # steering
u = ca.vertcat(theta, delta)

# Dynamics
xdot_state = ca.vertcat(
    v * ca.cos(eta),
    v * ca.sin(eta),
    (v * ca.tan(delta)) / l,
    -c * v + alpha * theta - beta
)

# DAE structure for integrator
dae = {
    "x": x_state,
    "p": u,
    "ode": xdot_state
}

# Integration step
dt = 0.1

# Create integrator
options = {"tf": dt}
integrator_cv = ca.integrator("integrator_cv", "cvodes", dae, options)

# Initial state
x0 = np.array([2.0, 1.0, -0.2, 2.0])

# Inputs
u_val = np.array([-0.3, 0.2])   # throttle, steering

# Simulation data
T = 10.0
N = int(T / dt)

trajectory = np.zeros((4, N + 1))
trajectory[:, 0] = x0

# Simulation loop
for k in range(N):
    result = integrator_cv(x0=trajectory[:, k], p=u_val)
    trajectory[:, k + 1] = np.array(result["xf"]).squeeze()

# Plot x-y trajectory
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