# Quadcopter Trajectory Tracking and Control

This project implements and compares different control strategies for quadcopter trajectory tracking. The trajectory is generated using RRT path planning followed by B-spline smoothing, and the resulting reference is tracked using three control approaches:

1. Nonlinear Model Predictive Control
2. Backstepping control
3. Differential flatness with PID control

---

## Prerequisites

Before running the code, install the required Python libraries using:

```bash
pip install -r requirements.txt
```

---

## Open-Source Libraries for Trajectory Generation

The RRT and B-spline trajectory generation code is adapted from open-source implementations.

The trajectory generation method used in `quadcopter_nmpc.py` is adapted from `trajectory_generator.py`.

For the other controller files, the trajectory is implemented as a hardcoded polynomial reference extracted from the previously generated B-spline trajectory.

---

## Quadcopter Dynamics

For the NMPC and Backstepping controllers, the quadcopter model is formulated as a causal chain. This means that the complete quadcopter behaviour is obtained by connecting several dynamic blocks in sequence, where each block describes an isolated part of the system dynamics.

The Differential Flatness model uses a different analytical modelling approach. Instead of directly following the same causal chain structure, it exploits the flat outputs of the quadcopter to generate the required thrust and attitude references from the desired trajectory.

All three approaches describe the coupling between translational and rotational dynamics in order to control the position of the quadcopter. The actuator inputs are represented through the four rotor speeds, or through quantities directly related to them.

---

## Control

### Nonlinear Model Predictive Control

The NMPC controller solves an optimization problem over a finite prediction horizon. The objective is to minimize the trajectory tracking error, the deviation from the reference input, the terminal error, and the variation of the control commands.

The NMPC problem is formulated as:

```math
\begin{aligned}
\min_{\{u_k\}_{k=0}^{N-1}} \quad
J =
&\sum_{k=0}^{N-1}
\Big[
(x_k - x_{\mathrm{ref},k})^T Q (x_k - x_{\mathrm{ref},k}) \\
&\quad +
(u_k - u_{\mathrm{ref},k})^T R (u_k - u_{\mathrm{ref},k}) \\
&\quad +
(u_k - u_{k-1})^T S (u_k - u_{k-1})
\Big] \\
&+
(x_N - x_{\mathrm{ref},N})^T P (x_N - x_{\mathrm{ref},N})
\end{aligned}
```

subject to:

$$
\dot{x} = f(x,u)
$$

$$
x_0 = x_{\text{init}}
$$

$$
0 \leq u_k \leq \omega_{\max}^2
$$

$$
-\Delta u_{\max} \leq u_k - u_{k-1} \leq \Delta u_{\max}
$$

$$
-\frac{\pi}{3} \leq \phi_k \leq \frac{\pi}{3}
$$

$$
-\frac{\pi}{3} \leq \theta_k \leq \frac{\pi}{3}
$$

$$
x \in \mathcal{X}_{\text{free}}
$$

where:

- \(x_k\) is the predicted state at step \(k\);
- \(x\_{\text{ref},k}\) is the reference state;
- \(u_k\) is the control input;
- \(u\_{\text{ref}}\) is the hover control input;
- \(Q\) is the state tracking weight matrix;
- \(R\) is the input tracking weight matrix;
- \(S\) is the input variation weight matrix;
- \(P\) is the terminal state weight matrix;
- \(\omega\_{\max}\) is the maximum rotor speed;
- \(\Delta u\_{\max}\) is the maximum allowed command variation between two consecutive time steps;
- \(\phi_k\) and \(\theta_k\) are the roll and pitch angles;
- \(\mathcal{X}\_{\text{free}}\) is the obstacle-free state space.

The command effort constraint limits how much the rotor commands are allowed to change between consecutive prediction steps. This prevents aggressive input variations and keeps the control signals smoother.

---

### Backstepping Control

The Backstepping controller is formulated as a sequence of control blocks that invert the causal chain of the quadcopter model.

The main idea is to recursively compute virtual control inputs for the intermediate variables until the final rotor commands are obtained. This allows the quadcopter to follow the desired reference trajectory while taking into account the nonlinear coupling between position, attitude, angular velocity, and rotor inputs.

---

### Differential Flatness with PID Control

For the Differential Flatness model, a PID controller is used to track the desired flat outputs.

The virtual control input is defined as:

$$
v = \ddot{x}_d + K_d \dot{e} + K_p e + K_i \int_0^t e(\tau)\,d\tau
$$

where:

$$
e = x_d - x
$$

$$
\dot{e} = \dot{x}_d - \dot{x}
$$

The virtual input is then mapped to the required thrust and attitude references using the flatness-based quadcopter model.

---

## Simulations

To run the simulation for each control system, use:

```bash
python quadcopter_nmpc.py
```

Runs the NMPC controller.

```bash
python quadcopter_backstepping.py
```

Runs the Backstepping controller.

```bash
python quadcopter_flatness.py
```

Runs the Differential Flatness PID controller.

---

## Project Files

The main files used in the project are:

- `quadcopter_nmpc.py` — NMPC trajectory tracking controller
- `quadcopter_backstepping.py` — Backstepping trajectory tracking controller
- `quadcopter_flatness.py` — Differential flatness PID controller
- `trajectory_generator.py` — RRT and B-spline trajectory generation
- `requirements.txt` — Required Python dependencies

---

## Notes

The NMPC controller uses the B-spline trajectory directly during simulation.

The Backstepping and Differential Flatness controllers use a polynomial trajectory extracted from the generated B-spline reference, allowing all methods to be compared on the same planned path.
