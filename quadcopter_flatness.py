import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.io import loadmat

from roblib import clean3D


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

    T = m * np.sqrt(u1**2 + u2**2 + (u3 + g)**2)

    # Protect asin from numerical errors slightly outside [-1, 1]
    asin_arg = (u1 * np.sin(psi) - u2 * np.cos(psi)) * m / T
    asin_arg = np.clip(asin_arg, -1.0, 1.0)

    phi_d = np.arcsin(asin_arg)

    theta_d = np.arctan(
        (u1 * np.cos(psi) + u2 * np.sin(psi)) / (u3 + g)
    )

    control_angles = np.array([phi_d, theta_d], dtype=float)

    return T, control_angles


# complete high-level position control subsystem

def high_level_position_control(ref, position, velocity, angles, Kp_pos, Kd_pos, params):
    """
    Complete high-level controller.

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
    Full continuous-time quadcopter model.

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


# reference trajectory 

def eval_spline_manual(s):
    # Hardcoded piecewise polynomial exported from the B-spline

    if s <= 0.0454545454545:
        u = s - (0)
        x = (13494.8420399)*u**5 + (-4229.26775279)*u**4 + (71.5563415028)*u**3 + (73.943687841)*u**2 + (-1.09547962773)*u + (2.00995890571)
        y = (41090.4748082)*u**5 + (-17214.8381827)*u**4 + (3058.74650388)*u**3 + (-281.284075929)*u**2 + (38.4994382104)*u + (1.65000510718)
        dx = (67474.2101993)*u**4 + (-16917.0710111)*u**3 + (214.669024508)*u**2 + (147.887375682)*u + (-1.09547962773)
        dy = (205452.374041)*u**4 + (-68859.3527309)*u**3 + (9176.23951163)*u**2 + (-562.568151858)*u + (38.4994382104)

    elif s <= 0.0909090909091:
        u = s - (0.0454545454545)
        x = (13494.8420399)*u**5 + (-1162.25819827)*u**4 + (-418.58238132)*u**3 + (43.9460248895)*u**2 + (4.76948533401)*u + (2.10422522764)
        y = (41090.4748082)*u**5 + (-7876.09390814)*u**4 + (777.752677433)*u**3 + (-38.9995124813)*u**2 + (26.2974918637)*u + (3.04056013703)
        dx = (67474.2101993)*u**4 + (-4649.03279308)*u**3 + (-1255.74714396)*u**2 + (87.8920497791)*u + (4.76948533401)
        dy = (205452.374041)*u**4 + (-31504.3756326)*u**3 + (2333.2580323)*u**2 + (-77.9990249626)*u + (26.2974918637)

    elif s <= 0.136363636364:
        u = s - (0.0909090909091)
        x = (2179.78334735)*u**5 + (1904.75135624)*u**4 + (-351.083003323)*u**3 + (-14.8679570273)*u**2 + (6.021484929)*u + (2.37016371889)
        y = (-16836.7455698)*u**5 + (1462.65036645)*u**4 + (194.712355461)*u**3 + (8.00999539244)*u**2 + (25.4911918327)*u + (4.20271668018)
        dx = (10898.9167368)*u**4 + (7619.00542498)*u**3 + (-1053.24900997)*u**2 + (-29.7359140545)*u + (6.021484929)
        dy = (-84183.7278492)*u**4 + (5850.60146581)*u**3 + (584.137066382)*u**2 + (16.0199907849)*u + (25.4911918327)

    elif s <= 0.181818181818:
        u = s - (0.136363636364)
        x = (-8213.04206459)*u**5 + (2400.15666246)*u**4 + (40.2722711051)*u**3 + (-37.0831625426)*u**2 + (3.25577776422)*u + (2.58873095587)
        y = (5122.97433494)*u**5 + (-2363.8827176)*u**4 + (112.78214172)*u**3 + (36.8815893821)*u**2 + (27.616357064)*u + (5.39921992439)
        dx = (-41065.2103229)*u**4 + (9600.62664985)*u**3 + (120.816813315)*u**2 + (-74.1663250851)*u + (3.25577776422)
        dy = (25614.8716747)*u**4 + (-9455.5308704)*u**3 + (338.34642516)*u**2 + (73.7631787643)*u + (27.616357064)

    elif s <= 0.227272727273:
        u = s - (0.181818181818)
        x = (-2377.26278764)*u**5 + (533.556193238)*u**4 + (306.973439805)*u**3 + (-9.55070607716)*u**2 + (0.860538727066)*u + (2.67253712304)
        y = (-7247.26244166)*u**5 + (-1199.57036875)*u**4 + (-211.168138857)*u**3 + (27.7678491871)*u**2 + (30.8896272888)*u + (6.73220540245)
        dx = (-11886.3139382)*u**4 + (2134.22477295)*u**3 + (920.920319415)*u**2 + (-19.1014121543)*u + (0.860538727066)
        dy = (-36236.3122083)*u**4 + (-4798.281475)*u**3 + (-633.504416572)*u**2 + (55.5356983743)*u + (30.8896272888)

    elif s <= 0.272727272727:
        u = s - (0.227272727273)
        x = (-4521.29063557)*u**5 + (-6.7308039534)*u**4 + (354.866657013)*u**3 + (36.6910505182)*u**2 + (2.04471428306)*u + (2.72256525079)
        y = (36223.7849448)*u**5 + (-2846.67546913)*u**4 + (-579.008669573)*u**3 + (-22.7047325852)*u**2 + (31.4997695008)*u + (8.16729223025)
        dx = (-22606.4531779)*u**4 + (-26.9232158136)*u**3 + (1064.59997104)*u**2 + (73.3821010365)*u + (2.04471428306)
        dy = (181118.924724)*u**4 + (-11386.7018765)*u**3 + (-1737.02600872)*u**2 + (-45.4094651705)*u + (31.4997695008)

    elif s <= 0.318181818182:
        u = s - (0.272727272727)
        x = (-10584.9433851)*u**5 + (-1034.29685749)*u**4 + (260.227778699)*u**3 + (80.7523778571)*u**2 + (7.48081937276)*u + (2.92373580171)
        y = (27864.8153641)*u**5 + (5386.00292742)*u**4 + (-348.160718819)*u**3 + (-102.930493357)*u**2 + (25.5505981245)*u + (9.49268890975)
        dx = (-52924.7169253)*u**4 + (-4137.18742997)*u**3 + (780.683336098)*u**2 + (161.504755714)*u + (7.48081937276)
        dy = (139324.076821)*u**4 + (21544.0117097)*u**3 + (-1044.48215646)*u**2 + (-205.860986714)*u + (25.5505981245)

    elif s <= 0.363636363636:
        u = s - (0.318181818182)
        x = (19246.4637716)*u**5 + (-3439.96580864)*u**4 + (-146.523372768)*u**3 + (93.4753413838)*u**2 + (15.820458627)*u + (3.44858680253)
        y = (-205911.098459)*u**5 + (11718.9155102)*u**4 + (1206.83186642)*u**3 + (-57.4692634231)*u**2 + (16.6533027903)*u + (10.4371149209)
        dx = (96232.3188578)*u**4 + (-13759.8632346)*u**3 + (-439.570118303)*u**2 + (186.950682768)*u + (15.820458627)
        dy = (-1029555.4923)*u**4 + (46875.6620407)*u**3 + (3620.49559925)*u**2 + (-114.938526846)*u + (16.6533027903)

    elif s <= 0.409090909091:
        u = s - (0.363636363636)
        x = (1621.11916031)*u**5 + (934.230503079)*u**4 + (-374.317491455)*u**3 + (48.9258685923)*u**2 + (22.5285654014)*u + (4.33611868194)
        y = (349419.240374)*u**5 + (-35079.0614124)*u**4 + (-916.817761059)*u**3 + (58.9944605427)*u**2 + (18.916482853)*u + (11.1987554586)
        dx = (8105.59580155)*u**4 + (3736.92201231)*u**3 + (-1122.95247437)*u**2 + (97.8517371847)*u + (22.5285654014)
        dy = (1747096.20187)*u**4 + (-140316.24565)*u**3 + (-2750.45328318)*u**2 + (117.988921085)*u + (18.916482853)

    elif s <= 0.454545454545:
        u = s - (0.409090909091)
        x = (-5534.02556545)*u**5 + (1302.66667588)*u**4 + (-170.963202459)*u**3 + (10.9864076998)*u**2 + (25.0417739426)*u + (5.43037973701)
        y = (-337845.334485)*u**5 + (44334.4023089)*u**4 + (-75.423134108)*u**3 + (-172.735732147)*u**2 + (12.8772157274)*u + (12.0124365123)
        dx = (-27670.1278272)*u**4 + (5210.6667035)*u**3 + (-512.889607378)*u**2 + (21.9728153997)*u + (25.0417739426)
        dy = (-1689226.67242)*u**4 + (177337.609236)*u**3 + (-226.269402324)*u**2 + (-345.471464294)*u + (12.8772157274)

    elif s <= 0.5:
        u = s - (0.454545454545)
        x = (-197.707802313)*u**5 + (44.9335928196)*u**4 + (-48.4540871232)*u**3 + (-1.37523997198)*u**2 + (25.3520861995)*u + (6.57977253751)
        y = (108142.045799)*u**5 + (-32448.6282558)*u**4 + (1005.10177981)*u**3 + (49.2940733752)*u**2 + (6.14998649202)*u + (12.3574905812)
        dx = (-988.539011564)*u**4 + (179.734371279)*u**3 + (-145.36226137)*u**2 + (-2.75047994397)*u + (25.3520861995)
        dy = (540710.228994)*u**4 + (-129794.513023)*u**3 + (3015.30533942)*u**2 + (98.5881467503)*u + (6.14998649202)

    elif s <= 0.545454545455:
        u = s - (0.5)
        x = (-1.34715386688e-05)*u**5 + (1.38490048585e-06)*u**4 + (-44.3692149228)*u**3 + (-7.61126347629)*u**2 + (24.9393888607)*u + (7.72490160427)
        y = (282864.134448)*u**5 + (-7870.89057429)*u**4 + (-2660.30902293)*u**3 + (-114.341415684)*u**2 + (6.97987052213)*u + (12.7157420074)
        dx = (-6.73576933442e-05)*u**4 + (5.53960194338e-06)*u**3 + (-133.107644768)*u**2 + (-15.2225269526)*u + (24.9393888607)
        dy = (1414320.67224)*u**4 + (-31483.5622971)*u**3 + (-7980.9270688)*u**2 + (-228.682831368)*u + (6.97987052213)

    elif s <= 0.590909090909:
        u = s - (0.545454545455)
        x = (87.0146889096)*u**5 + (-1.67681284798e-06)*u**4 + (-44.3692149493)*u**3 + (-13.6616109612)*u**2 + (23.9724400227)*u + (8.83861753188)
        y = (-434261.321654)*u**5 + (56416.4127093)*u**4 + (1752.92026207)*u**3 + (-309.033837982)*u**2 + (-16.8235803522)*u + (12.5682119024)
        dx = (435.073444548)*u**4 + (-6.70725139192e-06)*u**3 + (-133.107644848)*u**2 + (-27.3232219225)*u + (23.9724400227)
        dy = (-2171306.60827)*u**4 + (225665.650837)*u**3 + (5258.76078621)*u**2 + (-618.067675963)*u + (-16.8235803522)

    elif s <= 0.636363636364:
        u = s - (0.590909090909)
        x = (2503.34483248)*u**5 + (19.7760639845)*u**4 + (-42.5713911032)*u**3 + (-19.6302391955)*u**2 + (22.4573168504)*u + (9.89589740577)
        y = (49202.5049216)*u**5 + (-42279.3422119)*u**4 + (3038.1084891)*u**3 + (221.544062246)*u**2 + (-22.1280621284)*u + (11.4861975519)
        dx = (12516.7241624)*u**4 + (79.1042559378)*u**3 + (-127.71417331)*u**2 + (-39.2604783911)*u + (22.4573168504)
        dy = (246012.524608)*u**4 + (-169117.368848)*u**3 + (9114.32546731)*u**2 + (443.088124491)*u + (-22.1280621284)

    elif s <= 0.681818181818:
        u = s - (0.636363636364)
        x = (2358.58558056)*u**5 + (588.718071367)*u**4 + (12.7462575652)*u**3 + (-22.8392710429)*u**2 + (20.4697382356)*u + (10.8726982899)
        y = (406523.883912)*u**5 + (-31096.9547298)*u**4 + (-3632.46396014)*u**3 + (157.915719216)*u**2 + (2.01119699714)*u + (11.05249785)
        dx = (11792.9279028)*u**4 + (2354.87228547)*u**3 + (38.2387726955)*u**2 + (-45.6785420858)*u + (20.4697382356)
        dy = (2032619.41956)*u**4 + (-124387.818919)*u**3 + (-10897.3918804)*u**2 + (315.831438431)*u + (2.01119699714)

    elif s <= 0.727272727273:
        u = s - (0.681818181818)
        x = (-7145.82833104)*u**5 + (1124.76024877)*u**4 + (168.517013941)*u**3 + (-11.587936782)*u**2 + (18.7439449498)*u + (11.7601202117)
        y = (-351287.297574)*u**5 + (61294.8370684)*u**4 + (-887.20192936)*u**3 + (-341.135450519)*u**2 + (-9.15298710216)*u + (11.0751808559)
        dx = (-35729.1416552)*u**4 + (4499.04099507)*u**3 + (505.551041823)*u**2 + (-23.1758735639)*u + (18.7439449498)
        dy = (-1756436.48787)*u**4 + (245179.348273)*u**3 + (-2661.60578808)*u**2 + (-682.270901038)*u + (-9.15298710216)

    elif s <= 0.772727272727:
        u = s - (0.727272727273)
        x = (-7529.17959541)*u**5 + (-499.291644651)*u**4 + (225.377796133)*u**3 + (18.6240066599)*u**2 + (19.0050258304)*u + (12.6074167104)
        y = (39674.1047919)*u**5 + (-18543.1851076)*u**4 + (2999.31188526)*u**3 + (-32.1733622535)*u**2 + (-30.1365631111)*u + (10.0644841202)
        dx = (-37645.8979771)*u**4 + (-1997.1665786)*u**3 + (676.1333884)*u**2 + (37.2480133199)*u + (19.0050258304)
        dy = (198370.52396)*u**4 + (-74172.7404303)*u**3 + (8997.93565577)*u**2 + (-64.346724507)*u + (-30.1365631111)

    elif s <= 0.818181818182:
        u = s - (0.772727272727)
        x = (6513.38303234)*u**5 + (-2210.46882543)*u**4 + (-20.9640647827)*u**3 + (36.0967965677)*u**2 + (21.7468204615)*u + (13.5273347433)
        y = (41302.9498685)*u**5 + (-9526.3431094)*u**4 + (447.536592806)*u**3 + (184.209195139)*u**2 + (-20.589711934)*u + (8.8383853796)
        dx = (32566.9151617)*u**4 + (-8841.87530171)*u**3 + (-62.8921943482)*u**2 + (72.1935931353)*u + (21.7468204615)
        dy = (206514.749343)*u**4 + (-38105.3724376)*u**3 + (1342.60977842)*u**2 + (368.418390278)*u + (-20.589711934)

    elif s <= 0.863636363636:
        u = s - (0.818181818182)
        x = (11771.7855924)*u**5 + (-730.154499895)*u**4 + (-288.293457994)*u**3 + (11.9525557058)*u**2 + (24.207048657)*u + (14.5802656327)
        y = (15594.9405554)*u**5 + (-139.309048368)*u**4 + (-431.1590579)*u**3 + (165.931146773)*u**2 + (-3.76649840627)*u + (8.29246496142)
        dx = (58858.9279621)*u**4 + (-2920.61799958)*u**3 + (-864.880373981)*u**2 + (23.9051114116)*u + (24.207048657)
        dy = (77974.7027768)*u**4 + (-557.236193474)*u**3 + (-1293.4771737)*u**2 + (331.862293546)*u + (-3.76649840627)

    elif s <= 0.909090909091:
        u = s - (0.863636363636)
        x = (-16356.172089)*u**5 + (1945.25131657)*u**4 + (-177.830111024)*u**3 + (-25.3562946951)*u**2 + (23.483672766)*u + (15.6773737607)
        y = (57362.6212774)*u**5 + (3404.99562331)*u**4 + (-134.278460178)*u**3 + (120.055645186)*u**2 + (8.92620639585)*u + (8.42603273608)
        dx = (-81780.8604452)*u**4 + (7781.00526626)*u**3 + (-533.490333072)*u**2 + (-50.7125893903)*u + (23.483672766)
        dy = (286813.106387)*u**4 + (13619.9824932)*u**3 + (-402.835380535)*u**2 + (240.111290371)*u + (8.92620639585)

    elif s <= 0.954545454545:
        u = s - (0.909090909091)
        x = (6941.73532899)*u**5 + (-1772.06052185)*u**4 + (-162.085493323)*u**3 + (-40.8519622675)*u**2 + (20.4579414395)*u + (16.6808538316)
        y = (252079.602335)*u**5 + (16441.9550045)*u**4 + (1669.98977872)*u**3 + (197.827368596)*u**2 + (21.5115208251)*u + (9.09287345456)
        dx = (34708.676645)*u**4 + (-7088.2420874)*u**3 + (-486.256479968)*u**2 + (-81.703924535)*u + (20.4579414395)
        dy = (1260398.01168)*u**4 + (65767.8200181)*u**3 + (5009.96933615)*u**2 + (395.654737191)*u + (21.5115208251)

    elif s <= 1:
        u = s - (0.954545454545)
        x = (6941.73532899)*u**5 + (-194.393401626)*u**4 + (-340.85403182)*u**3 + (-78.4029363376)*u**2 + (15.2219424724)*u + (17.5049155537)
        y = (252079.602335)*u**5 + (73732.773717)*u**4 + (9867.69238977)*u**3 + (866.118056297)*u**2 + (61.4039754575)*u + (10.7553410713)
        dx = (34708.676645)*u**4 + (-777.573606503)*u**3 + (-1022.56209546)*u**2 + (-156.805872675)*u + (15.2219424724)
        dy = (1260398.01168)*u**4 + (294931.094868)*u**3 + (29603.0771693)*u**2 + (1732.23611259)*u + (61.4039754575)

    else:
        u = s - (0.954545454545)
        x = (6941.73532899)*u**5 + (-194.393401626)*u**4 + (-340.85403182)*u**3 + (-78.4029363376)*u**2 + (15.2219424724)*u + (17.5049155537)
        y = (252079.602335)*u**5 + (73732.773717)*u**4 + (9867.69238977)*u**3 + (866.118056297)*u**2 + (61.4039754575)*u + (10.7553410713)
        dx = (34708.676645)*u**4 + (-777.573606503)*u**3 + (-1022.56209546)*u**2 + (-156.805872675)*u + (15.2219424724)
        dy = (1260398.01168)*u**4 + (294931.094868)*u**3 + (29603.0771693)*u**2 + (1732.23611259)*u + (61.4039754575)

    return np.array([x, y]), np.array([dx, dy])


s_min = 0.0
s_max = 1.0

# for flatness/MATLAB model, z is positive upward
ZD = 10.0

B_SPLINE_T_FINAL = 30.0
B_SPLINE_EPS = 1e-4


def s_from_time(t):
    return np.clip(t / B_SPLINE_T_FINAL, s_min, s_max)




# control system simulation

def reference_trajectory(t):
    """
    B-spline reference for the high-level position controller.

    Returns:
    [
        x_ref, y_ref, z_ref,
        x_dot_ref, y_dot_ref, z_dot_ref,
        x_ddot_ref, y_ddot_ref, z_ddot_ref
    ]
    """

    s = s_from_time(t)
    dsdt = 1.0 / B_SPLINE_T_FINAL

    xy, dxy_ds = eval_spline_manual(s)

    # Numerical second derivative with respect to s
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
        d2xy_ds2[0] * dsdt**2,
        d2xy_ds2[1] * dsdt**2,
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


def yaw_reference(t):
    """
    Desired yaw generated from the B-spline tangent.

    """

    s = s_from_time(t)
    _, dxy_ds = eval_spline_manual(s)

    psi_d = np.arctan2(dxy_ds[1], dxy_ds[0])

    # Keep yaw derivatives zero for now.
    # This is safer unless you specifically want aggressive yaw tracking.
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
# trajectory = loadmat("trajectory_1.mat")

# desired_position = trajectory["desired_position"]
# desired_psi = trajectory["desired_psi"]

# t_ref = desired_position[:, 0]

# pos_ref_data = desired_position[:, 1:10]
# psi_ref_data = desired_psi[:, 1]

# psi_dot_ref_data = np.gradient(psi_ref_data, t_ref)
# psi_ddot_ref_data = np.gradient(psi_dot_ref_data, t_ref)

# base obstacles for RRT/B-spline reference

base_obstacles = [
    {"vertices": np.array([[5, 5], [9, 5], [7, 10]], dtype=float)},
    {"vertices": np.array([[12, 2], [17, 3], [16, 7], [11, 8]], dtype=float)},
    {"vertices": np.array([[10, 14], [12, 12], [14, 12], [16, 16], [12, 16]], dtype=float)},
    {"vertices": np.array([[3, 14], [6.5, 14], [6.5, 19], [3, 19]], dtype=float)},
]


def sample_reference_curve(num_points=500):
    s_vals = np.linspace(s_min, s_max, num_points)
    ref_curve = np.zeros((num_points, 3))

    for i, s in enumerate(s_vals):
        xy, _ = eval_spline_manual(s)
        ref_curve[i, 0] = xy[0]
        ref_curve[i, 1] = xy[1]
        ref_curve[i, 2] = ZD

    return ref_curve


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
    
     # initial position
    initial_ref = reference_trajectory(0.0)
    state[0:3] = initial_ref[0:3]

    psi_d0, _, _ = yaw_reference(0.0)
    state[8] = psi_d0

    # Position controller gains
    # These act on:
    # x_ddot = v1, y_ddot = v2, z_ddot = v3
    Kp_pos = np.array([0.8, 0.8, 0.8]) * 5
    Kd_pos = np.array([1, 1, 1]) * 5

    # Attitude controller gains
    # These act on:
    # phi_ddot = sigma1, theta_ddot = sigma2, psi_ddot = sigma3
    # Attitude loop should be faster than position loop.
    Kp_att = np.array([40.0, 40.0, 40.0])
    Kd_att = np.array([20.0, 20.0, 20.0])

    for k, t in enumerate(time):
        position = state[0:3]
        velocity = state[3:6]
        angles = state[6:9]
        angle_rates = state[9:12]

        # High-level position controller

        ref_pos = reference_trajectory(t)

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

    return time, states, inputs, refs_pos, refs_att, virtual_inputs, sigmas, motor_commands


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
    ax.set_title("3D position tracking")
    ax.legend()
    ax.grid(True)
    
    #obstacle plotting
    draw_obstacles_3d(ax, base_obstacles)

    ref_curve = sample_reference_curve()
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
    
    ref_curve = sample_reference_curve()

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
    time, states, inputs, refs_pos, refs_att, virtual_inputs, sigmas, motor_commands = simulate_complete_control_system()

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
    
    
    
    
    
    
    
    
    
    