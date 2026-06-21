import casadi as ca
import numpy as np

class FullStateNMPC:
    def __init__(self, dt, Npred, m, g, obstacles, B_mat, I_mat, w_max):
        self.dt = dt
        self.Npred = Npred
        self.m = m
        self.g = g
        self.B_mat = ca.DM(B_mat)
        self.invI = ca.DM(np.linalg.inv(I_mat))
        self.I_mat = ca.DM(I_mat)
        self.w_max = w_max

        self.opti = ca.Opti()

        # States: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.X = self.opti.variable(12, Npred + 1)
        # Inputs: [w1^2, w2^2, w3^2, w4^2]
        self.U = self.opti.variable(4, Npred)
        

        self.X_ref = self.opti.parameter(12, Npred + 1)
        self.x0 = self.opti.parameter(12)

        self.opti.subject_to(self.X[:, 0] == self.x0)
        
        # WARM START: Hovering motor speeds
        invB = np.linalg.inv(B_mat)
        hover_w2 = invB @ np.array([m*g, 0, 0, 0])
        self.opti.set_initial(self.U, np.tile(hover_w2, (Npred, 1)).T)
        self.opti.set_initial(self.X[6:12, :], 0.0) 

       # --- RUNNING COSTS old params---
        # Q_weights = np.array([400, 400, 700, 120, 120, 180, 30, 30, 20, 5, 5, 5])
        # R_weights = np.array([3, 3, 3, 3]) 
        # P_weights = np.array([ # Terminal cost
        #     1000, 1000, 1500,  # Goal position (X, Y, Z)
        #     400, 400, 400,     # Terminal velocity
        #     40,  40,  30,        # Heading lock
        #     1,  1,  1        
        # ])
        # S_weights = np.array([100, 100, 100, 100])

        # --- RUNNING COSTS new params---
        Q_weights = np.array([
            200, 200, 400,   # x, y, z position
            20,  20,  40,     # vx, vy, vz velocity
            10,  10,  7,     # phi, theta, psi
            1,   1,   1       # angular rates
        ])

        R_weights = np.array([
            3e-3, 3e-3, 3e-3, 3e-3
        ])

        P_weights = np.array([
            300, 300, 550,   # terminal x, y, z
            150,  150,  275,    # terminal velocity
            15,   15,   10,     # terminal attitude
            1,    1,    1       # terminal angular rates
        ])

        S_weights = np.array([
            3e-4, 3e-4, 3e-4, 3e-4
        ])

        cost = 0
        for k in range(Npred):
            # RK4 Dynamics
            k1 = self.f_dynamics(self.X[:, k], self.U[:, k])
            k2 = self.f_dynamics(self.X[:, k] + 0.5 * self.dt * k1, self.U[:, k])
            k3 = self.f_dynamics(self.X[:, k] + 0.5 * self.dt * k2, self.U[:, k])
            k4 = self.f_dynamics(self.X[:, k] + self.dt * k3, self.U[:, k])
            x_next = self.X[:, k] + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            self.opti.subject_to(self.X[:, k+1] == x_next) #\dotx = f(x,u) constraint

            # Running Cost
            err = self.X[:, k] - self.X_ref[:, k]
            u_hover = ca.DM(hover_w2)
            cost += ca.sumsqr(Q_weights * err) + ca.sumsqr(R_weights * (self.U[:, k] - u_hover))
            
            # Smoothness cost on control changes (jerk minimization)
            if k < Npred - 1: 
                delta_u = self.U[:, k+1] - self.U[:, k]
                
                # delta_u_max should take into account that the input is self.w_max**2
                # for old params
                # delta_u_max = 1.0
                # for new params
                delta_u_max = 5000
                self.opti.subject_to(delta_u <= delta_u_max)
                self.opti.subject_to(delta_u >= -delta_u_max)
                
                cost += ca.sumsqr(S_weights * delta_u)

            # Physical Limits
            self.opti.subject_to(self.U[:, k] <= self.w_max**2)
            self.opti.subject_to(self.U[:, k] >= 0)
            self.opti.subject_to(self.X[6, k] <= np.pi/3)
            self.opti.subject_to(self.X[6, k] >= -np.pi/3)
            self.opti.subject_to(self.X[7, k] <= np.pi/3)
            self.opti.subject_to(self.X[7, k] >= -np.pi/3)
            
            

        #* Terminal Cost
        terminal_err = self.X[:, Npred] - self.X_ref[:, Npred]
        cost += ca.sumsqr(P_weights * terminal_err)

        #* OBSTACLE CONSTRAINTS
        alpha = 5.0 
        margin = 0.05
        for k in range(1, Npred + 1):
            p_2d = ca.vertcat(self.X[0, k], self.X[1, k])
            for obs in obstacles:
                A, b = ca.DM(obs["A"]), ca.DM(obs["b"])
                
                d = ca.mtimes(A, p_2d) - b
                
                smooth_max = (1.0 / alpha) * ca.log(ca.sum1(ca.exp(alpha * d)))
                self.opti.subject_to(smooth_max >= margin)
        self.opti.minimize(cost)
        
        opts = {
            "print_time": False,
            "ipopt.print_level": 0, 
            "ipopt.sb": "yes", 
            "ipopt.max_iter": 500,
            "ipopt.acceptable_tol": 1e-4,             
            "ipopt.acceptable_obj_change_tol": 1e-4 
        }
        self.opti.solver('ipopt', opts)

    def f_dynamics(self, x_state, u_w2):
        """CasADi symbolic mapping of the Z-Down (NED) Drone Dynamics"""
        pos, vel = x_state[0:3], x_state[3:6]
        phi, theta, psi = x_state[6], x_state[7], x_state[8]
        p, q, r = x_state[9], x_state[10], x_state[11]

        # Torques and Thrust from Motors
        tau = ca.mtimes(self.B_mat, u_w2)
        thrust, tau_body = tau[0], tau[1:4]

        # Translation Acceleration
        ax = -(thrust / self.m) * (ca.cos(psi)*ca.sin(theta)*ca.cos(phi) + ca.sin(psi)*ca.sin(phi))
        ay = -(thrust / self.m) * (ca.sin(psi)*ca.sin(theta)*ca.cos(phi) - ca.cos(psi)*ca.sin(phi))
        az = self.g - (thrust / self.m) * ca.cos(theta) * ca.cos(phi)

        # Euler Angle Kinematics
        dphi = p + ca.sin(phi)*ca.tan(theta)*q + ca.cos(phi)*ca.tan(theta)*r
        dtheta = ca.cos(phi)*q - ca.sin(phi)*r
        dpsi = ca.sin(phi)/ca.cos(theta)*q + ca.cos(phi)/ca.cos(theta)*r

        # Rigid Body Angular Acceleration: inv(I) * (tau - w x (I * w))
        omega = ca.vertcat(p, q, r)
        I_omega = ca.mtimes(self.I_mat, omega)
        cross_w_Iw = ca.vertcat(
            q * I_omega[2] - r * I_omega[1],
            r * I_omega[0] - p * I_omega[2],
            p * I_omega[1] - q * I_omega[0]
        )
        d_omega = ca.mtimes(self.invI, tau_body - cross_w_Iw)

        return ca.vertcat(vel[0], vel[1], vel[2], ax, ay, az, dphi, dtheta, dpsi, d_omega[0], d_omega[1], d_omega[2])

    def get_control(self, x0_val, X_ref_val):
        # X_ref_val = X_ref_val.copy()

        # blend_steps = min(5, self.Npred + 1)

        # for i in range(blend_steps):
        #     alpha = i / (blend_steps - 1)
        #     X_ref_val[:, i] = (1 - alpha) * x0_val + alpha * X_ref_val[:, i]
        
        self.opti.set_value(self.x0, x0_val)
        self.opti.set_value(self.X_ref, X_ref_val)
        try:
            sol = self.opti.solve()
            self.opti.set_initial(self.X, sol.value(self.X))
            self.opti.set_initial(self.U, sol.value(self.U))
            return np.sqrt(np.abs(sol.value(self.U[:, 0]))) # Return actual w, not w^2
        except:
            print("Warning: Full-State NMPC solver failed. Hovering safely.")
            # If the solver crashes, hold previous thrust or hover to prevent dropping out of the sky
            safe_w2 = self.opti.debug.value(self.U[:, 0])
            return np.sqrt(np.abs(safe_w2))