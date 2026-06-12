from roblib import *  # Ensure this is in your Python path or in the same directory

s_min = 0.0
s_max = 1.0
s_current = s_min

ZD = -10.0

LOOKAHEAD_S = 0.015
SEARCH_WINDOW_S = 0.12
K_PATH = 2
VD = 1.5

traj_history = []
time_history = []
error_history = []
command_history = []

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

def sample_reference_curve(num_points=500):
    s_vals = np.linspace(s_min, s_max, num_points)
    ref = np.zeros((num_points, 3))

    for i, s in enumerate(s_vals):
        # xy, _ = eval_spline_poly(s)
        xy, _ = eval_spline_manual(s)
        ref[i, 0] = xy[0]
        ref[i, 1] = xy[1]
        ref[i, 2] = ZD

    return ref


ref_curve = sample_reference_curve()


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

# Obstacles
base_obstacles = [
    {"vertices": np.array([[5, 5], [9, 5], [7, 10]], dtype=float)},
    {"vertices": np.array([[12, 2], [17, 3], [16, 7], [11, 8]], dtype=float)},
    {"vertices": np.array([[10, 14], [12, 12], [14, 12], [16, 16], [12, 16]], dtype=float)},
    {"vertices": np.array([[3, 14], [6.5, 14], [6.5, 19], [3, 19]], dtype=float)},
]

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
U_MAX = 700.0       # optional signed command magnitude limit


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
for t in arange(0, 30, dt):
    X = hstack((p.flatten(), eulermat2angles(R), vr.flatten(), wr.flatten())).reshape(-1, 1)
    w_cmd = control(X)

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
    error_history.append((p - ref_p).flatten())

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
