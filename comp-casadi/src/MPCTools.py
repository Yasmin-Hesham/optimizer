import casadi as ca
from casadi import sin, cos, pi
import numpy as np
 
''' Constants '''
# Costs
Q_x = 150
Q_y = 150
Q_theta = 3000
R1 = R2 = R3 = 0.5     # speed cost
A1 = A2 = A3 = 0       # Acceleration
H = 1000               # path parameter cost
 
# MPC parameters
sampling_time = 1   # time between steps in seconds
N = 100             # number of look ahead steps

# MPC limits
x_min = -2000    # mm
x_max = 2000     # mm
y_min = -2000    # mm
y_max = 2000     # mm
theta_min = -ca.inf
theta_max = ca.inf
vx_max =  300    # mm/sec
vy_max =  300    # mm/sec
w_max = 1.5      # rad/sec
# a_max =  ca.inf    # rad/s^2


''' Path Tracking '''
import numpy as np
import casadi as ca
points = np.array([
    [ 1700,     0, 0],
    [ 1700, -1700, 0],
    [-1700, -1700, 0],
    [-1700,     0, 0],
    [    0,     0, 0]
])

num_points = points.shape[0]
k = np.linspace(0, 1, num_points)  # path parameter
order = 7
x_coeffs = np.polyfit(k, points[:, 0], order)
# x_gen = np.poly1d(x_k)

y_coeffs = np.polyfit(k, points[:, 1], order)
# y_gen = np.poly1d(y_k)

def path_poly(sym, coeffs):
    poly = 0
    for i in range(len(coeffs)):
        poly += coeffs[i] * (sym**i)
    return poly

# import matplotlib.pyplot as plt
# plt.plot(points[:, 0], points[:,   1]); plt.show()
# plt.plot(k, points[:, 0], k, x_gen(k)); plt.show()
# plt.plot(k, points[:, 1], k, y_gen(k)); plt.show()
# plt.plot(x_gen(k), y_gen(k)); plt.show()

''' Symbols '''
# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(
    x,
    y,
    theta
)
n_states = states.numel()
 
# control symbolic variables
V_x = ca.SX.sym('V_x')
V_y = ca.SX.sym('V_y')
Omega = ca.SX.sym('Omega')
controls = ca.vertcat(
    V_x,
    V_y,
    Omega
)
n_controls = controls.numel()
 
# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_theta)
 
# controls weights matrix
R = ca.diagcat(R1, R2, R3)
 
# acceleration weights matrix
A = ca.diagcat(A1, A2, A3)
 
rot_3d_z = ca.vertcat(
    ca.horzcat(cos(theta), -sin(theta), 0),
    ca.horzcat(sin(theta),  cos(theta), 0),
    ca.horzcat(         0,           0, 1)
)
 
state_change_rate = rot_3d_z @ controls
 
derivatives = ca.Function('derivatives', [states, controls], [state_change_rate])


''' Defining Upper and Lower Bounds '''
# initialize boundaries arrays
lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
 
# state lower bounds
lbx[0: n_states*(N+1): n_states] = x_min      # X lower bound
lbx[1: n_states*(N+1): n_states] = y_min      # Y lower bound
lbx[2: n_states*(N+1): n_states] = theta_min  # theta lower bound
 
# state upper bounds
ubx[0: n_states*(N+1): n_states] = x_max      # X upper bound
ubx[1: n_states*(N+1): n_states] = y_max      # Y upper bound
ubx[2: n_states*(N+1): n_states] = theta_max  # theta upper bound
 
# control lower bounds
lbx[n_states*(N+1)+0:: n_controls] = -vx_max  # Vx lower bound
lbx[n_states*(N+1)+1:: n_controls] = -vy_max  # Vy lower bound
lbx[n_states*(N+1)+2:: n_controls] = -w_max   # w lower bound
 
# control upper bounds
ubx[n_states*(N+1)+0:: n_controls] = vx_max   # Vx upper bound 
ubx[n_states*(N+1)+1:: n_controls] = vy_max   # Vy upper bound
ubx[n_states*(N+1)+2:: n_controls] = w_max    # w upper bound
