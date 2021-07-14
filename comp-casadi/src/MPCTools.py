import casadi as ca
from casadi import sin, cos, pi
 
''' Constants '''
# Costs
Q_x = 50
Q_y = 50
Q_theta = 50
R1 = R2 = R3 = 0.5     # speed cost
A1 = A2 = A3 = 0       # No Acceleration
 
# MPC parameters
sampling_time = 2     # time between steps in seconds
N = 100               # number of look ahead steps
 
# MPC limits
# TODO: 1- set limits, 2- make sure of units
x_min = -ca.inf
x_max = ca.inf
y_min = -ca.inf
y_max = ca.inf
theta_min = -ca.inf
theta_max = ca.inf
vx_max =  300    # mm/sec
vy_max =  300    # mm/sec
w_max = 1.5      # rad/sec
# a_max =  ca.inf    # rad/s^2
 
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
