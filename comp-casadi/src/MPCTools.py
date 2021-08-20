import casadi as ca
from casadi import sin, cos, pi
import numpy as np

''' Parameters '''
# Cost parameters
Q_X = Q_Y = 5e5
Q_PHI = 1e10
Q_THETA = 1e10

R_VX = R_VY = 1e4
R_VPHI = 1e5
R_VTHETA = 1e10

# MPC parameters
SAMPLING_TIME = 0.5       # time between steps in seconds
N = 100                   # number of look ahead steps

# MPC limits
# States
X_MIN = Y_MIN = -2000        # mm
X_MAX = Y_MAX =  2000        # mm
PHI_MIN = -ca.inf            # rad
PHI_MAX = ca.inf             # rad

# Path Parameter  
THETA_MIN = 0                # Unitless
THETA_MAX = 100              # Unitless

# Control Actions
VX_MIN = VY_MIN = -300       # mm/sec
VX_MAX = VY_MAX =  300       # mm/sec
VPHI_MIN = -1.5              # rad/sec
VPHI_MAX = 1.5               # rad/sec

# Virtual Control
VTHETA_MIN = 0                # Unitless
VTHETA_MAX = 2                # Unitless

''' Path Tracking '''

# ---------
# |       |
# |       |
# x       |
#         |
#         |
#         |

#      x     y    phi
POINTS = np.array([
    [   0,    0,  0],
    [   0, 1750,  0],
    [1750, 1750,  0],
    [1750,    0,  0],
    [1750,-1750,  0],
    [   0,-1750,  0]
])

NUM_POINTS = POINTS.shape[0]
ORDER = 5

SCALE = np.linspace(THETA_MIN, THETA_MAX, NUM_POINTS)
X_COEFFS = np.polyfit(SCALE, POINTS[:, 0], ORDER+2)
Y_COEFFS = np.polyfit(SCALE, POINTS[:, 1], ORDER)   # twice continuously differentiable
# PHI_COEFFS = np.polyfit(SCALE, POINTS[:, 2], ORDER)

def path_poly(sym, coeffs):
    poly = 0
    degree = coeffs.shape[0]
    for i in range(degree):
        poly += coeffs[i] * ca.power(sym, (degree - 1 - i))
    return poly

# for quick test
x_gen = np.poly1d(X_COEFFS)
y_gen = np.poly1d(Y_COEFFS)
# PHI_gen = np.poly1d(y_coeffs
# plt.plot(x_gen(k), y_gen(k), PHI_gen(k))
# plt.show()

''' Functions '''
# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
phi = ca.SX.sym('phi')

states = ca.vertcat(
    x,
    y,
    phi
)
n_states = states.numel()

# control symbolic variables
V_x = ca.SX.sym('V_x')
V_y = ca.SX.sym('V_y')
V_phi = ca.SX.sym('V_phi')

controls = ca.vertcat(
    V_x,
    V_y,
    V_phi
)
n_controls = controls.numel()

rotmat_3d_z = ca.vertcat(
    ca.horzcat(cos(phi), -sin(phi), 0),
    ca.horzcat(sin(phi),  cos(phi), 0),
    ca.horzcat(       0,         0, 1)
)

state_change_rate = rotmat_3d_z @ controls
x_dot = ca.Function('x_dot', [states, controls], [state_change_rate])

h_states = (n_states+1) * (N+1)
h_controls = (n_controls+1) * N        # controls for all steps horizon

''' Defining Upper and Lower Bounds '''
# initialize boundaries arrays
lbx = ca.DM.zeros((h_states+h_controls, 1))
ubx = ca.DM.zeros((h_states+h_controls, 1))

# state lower bounds
lbx[0: h_states: (n_states+1)] = X_MIN      # X lower bound
lbx[1: h_states: (n_states+1)] = Y_MIN      # Y lower bound
lbx[2: h_states: (n_states+1)] = PHI_MIN    # PHI lower bound
lbx[3: h_states: (n_states+1)] = THETA_MIN  # path lower bounds

# state upper bounds
ubx[0: h_states: (n_states+1)] = X_MAX      # X upper bound
ubx[1: h_states: (n_states+1)] = Y_MAX      # Y upper bound
ubx[2: h_states: (n_states+1)] = PHI_MAX    # PHI upper bound
ubx[3: h_states: (n_states+1)] = THETA_MAX  # path upper bounds

# control lower bounds 
lbx[h_states+0: h_states+h_controls: (n_controls+1)] = VX_MIN       # Vx lower bound
lbx[h_states+1: h_states+h_controls: (n_controls+1)] = VY_MIN       # Vy lower bound
lbx[h_states+2: h_states+h_controls: (n_controls+1)] = VPHI_MIN        # Vphi lower bound
lbx[h_states+3: h_states+h_controls: (n_controls+1)] = VTHETA_MIN   # Vtheta lower bound

# control upper bounds
ubx[h_states+0: h_states+h_controls: (n_controls+1)] = VX_MAX       # Vx upper bound 
ubx[h_states+1: h_states+h_controls: (n_controls+1)] = VY_MAX       # Vy upper bound
ubx[h_states+2: h_states+h_controls: (n_controls+1)] = VPHI_MAX        # Vphi upper bound
ubx[h_states+3: h_states+h_controls: (n_controls+1)] = VTHETA_MAX   # Vtheta upper bound
