from MPCTools import *
import numpy as np

''' Preparing variables '''
# states: matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states+1, N+1)
# controls: matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls+1, N)

# parameters: column vector for storing initial state, target state, initial controls and path tracking 
initialStates = ca.SX.sym('initialState', n_states+1, 1)
# initialControls = ca.SX.sym('initialControls', n_controls+1, 1)
polyX_coeffs = ca.SX.sym('polyX_coeffs', ORDER+3, 1)
polyY_coeffs = ca.SX.sym('polyY_coeffs', ORDER+1, 1)

# y = c0*k^n + c1*k^(n-1) + ... + c(n-1)*k^1 + cn*k^0
# dK/dtheta = n*c0*k^(n-1) + (n-1)*c1*k^(n-2) + ... + c(n-1)*k^0 
dK_dtheta = path_poly(X[3, :], polyX_coeffs[:-1] * np.arange(ORDER+2, 0, -1))
dP_dtheta = path_poly(X[3, :], polyY_coeffs[:-1] * np.arange(ORDER, 0, -1))
dK_dP = dK_dtheta / dK_dtheta
d2K_dtheta2 = path_poly(X[3, :], polyY_coeffs[:-2] * np.arange(ORDER, 1, -1) * np.arange(ORDER-1, 0, -1))


''' Constructing cost function '''
cost_fn = 0  # initialize cost

# accumulate state-error cost
for i in range(N+1):
    #state_error = state - reference_state with respect to theta (path parameter)
    # p(theta) = (K(theta), P(theta), tan-1 dp/dk)
    pathX_error = X[0, i] - path_poly(X[3, i], polyX_coeffs)
    pathY_error = X[1, i] - path_poly(X[3, i], polyY_coeffs)
    phi_error = X[2, i] - ca.atan2(dK_dtheta[:, i], dP_dtheta[:, i])
    theta_error = X[3, i] - THETA_MAX

    # add the weighted path error to the total cost
    cost_fn += pathX_error.T @ Q_X @ pathX_error
    cost_fn += pathY_error.T @ Q_Y @ pathY_error
    cost_fn += phi_error.T @ Q_PHI @ phi_error
    cost_fn += theta_error.T @ Q_THETA @ theta_error

# accumulate control-error cost
for i in range(N):
    # control_error = U[:, i] # control_error = control - (target_control=0)
    # Ux_ref = theta_dot * sqroot((dK/dtheta)^2 + (dP/dtheta)^2)
    # Ux = path_poly(K[0, i], d_dK_coeff(x_coeff))
    # vx_ref = U[3, i] * ca.sqrt(ca.power(dK_dtheta[:, i], 2) + ca.power(dP_dtheta[:, i], 2))
    vx_ref = 0
    vx_error = U[0, i] -  vx_ref

    # vy_ref = U[3, i] * ca.sqrt(ca.power(dK_dtheta[:, i], 2) + ca.power(dP_dtheta[:, i], 2))
    vy_ref = U[3, i] * ca.sqrt(ca.power(dK_dtheta[:, i], 2) + ca.power(dP_dtheta[:, i], 2))
    vy_error = U[1, i] - vy_ref

    # vphi_ref = (U[3: i] * d2P_dtheta2[:, i] * dK_dtheta[:, i]) / (1 + ca.power(dP_dK[:, i], 2))
    vphi_ref = 0
    vphi_error = U[2, i] - vphi_ref

    vtheta_ref = 0
    vtheta_error =  U[3, i] - vtheta_ref

    # add the weighted control error to the total cost
    cost_fn += vx_error.T @ R_VX @ vx_error
    cost_fn += vy_error.T @ R_VY @ vy_error
    cost_fn += vphi_error.T @ R_VPHI @ vphi_error
    cost_fn += vphi_error.T @ R_VTHETA @ vphi_error

''' Constructing constraints '''
g = ca.SX()    # initialize contraints vector
lbg = ca.DM()  # initialize lowerbounds vector
ubg = ca.DM()  # initialize upperbounds vector

# Equality Constraints
# adding physical constraints (next state and path parameter must adhere to the robot model) 
g = ca.vertcat(g, X[:, 0] - initialStates)
lbg = ca.vertcat(lbg, [0] * (n_states+1))
ubg = ca.vertcat(ubg, [0] * (n_states+1))

# System Dynamics Constraints
for i in range(N):
    states = X[:n_states, i]
    controls = U[:n_controls, i]
    next_state = X[:n_states, i+1]
    # 4th order Runge-Kutta discretization
    k1 = x_dot(states, controls)
    k2 = x_dot(states + SAMPLING_TIME/2*k1, controls)
    k3 = x_dot(states + SAMPLING_TIME/2*k2, controls)
    k4 = x_dot(states + SAMPLING_TIME * k3, controls)
    predicted_state = states + (SAMPLING_TIME / 6) * (k1 + k2*2 + k3*2 + k4)

    # add constraint for meccanum wheeled: next_state == predicted_state
    g = ca.vertcat(g, next_state - predicted_state)
    lbg = ca.vertcat(lbg, [0] * n_states)
    ubg = ca.vertcat(ubg, [0] * n_states)

# Virtual System Dynamics Constraints
for i in range(N):
    v_states = X[-1, i]
    v_controls = U[-1, i]
    next_v_states = X[-1, i+1]

    # Euler discretization (x2 = x1 + v*dt)
    predicted_v_states = v_states + v_controls * SAMPLING_TIME

    g = ca.vertcat(g, next_v_states - predicted_v_states)
    lbg = ca.vertcat(lbg, [0])
    ubg = ca.vertcat(ubg, [0])

# TODO: add dynamic obstacles

''' Creating the NLP Solver '''
optimization_variables = ca.vertcat(
    X.reshape((-1, 1)),
    U.reshape((-1, 1))
)

parameters = ca.vertcat(
    initialStates.reshape((-1, 1)),
    polyX_coeffs.reshape((-1, 1)),
    polyY_coeffs.reshape((-1, 1))
)

nlp_prob = {
    'f': cost_fn,                 # cost function
    'x': optimization_variables,  # manipulated variables
    'g': g,                       # constraints
    'p': parameters               # parameters variables
}

options = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': False
}

args = {
    'lbx': lbx,
    'ubx': ubx,
    'lbg': lbg,
    'ubg': ubg
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, options)

#######################################################

# TODO: needs edits
 
class MPC:
    X0 = ca.DM.zeros(((n_states+1) * (N+1), 1)) + THETA_MIN
    U0 = ca.DM.zeros(((n_controls+1) * N, 1))

    def compute(self, initial_state, x_coeffs, y_coeffs, isDifferential=False):
        initial_state = ca.vertcat(
            initial_state,
            self.X0[7, 0]
        )

        args['p'] = ca.vertcat(
            initial_state,          # current state
            # self.U0[:n_controls], # current controls
            x_coeffs,               # coefficients of x in path in terms of k
            y_coeffs                # coefficients of y in path in terms of k
        )

        # optimization variable current state
        # 1. initial state from filtered sensor readings
        # 2. initial next states are extracted from previous optimized state
        # For the next step, the initial state is from current sensors' readings.
        # Then, previous optimization outputs are used as initialization for the next step.
        # For the last state, copy the previous cell in the state vector "X0".
        self.X0[:(n_states+1)] = initial_state
        self.X0[(n_states+1):-(n_states+1)] = self.X0[2*(n_states+1):]
        # self.X0[-(n_states+1):] = self.X0[-2*(n_states+1):-(n_states+1)]

        # Same for the controls as the states.
        self.U0[:-(n_controls+1)] = self.U0[(n_controls+1):]


        args['x0'] = ca.vertcat(
            self.X0,
            self.U0
        ) 

        if isDifferential:
            args['lbx'][h_states: h_states+h_controls: n_controls+1] = 0
            args['ubx'][h_states: h_states+h_controls: n_controls+1] = 0

        else:
            args['lbx'][h_states: h_states+h_controls: n_controls+1] = VX_MIN
            args['ubx'][h_states: h_states+h_controls: n_controls+1] = VX_MAX

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        self.X0 = sol['x'][:h_states]
        self.U0 = sol['x'][h_states: h_states+h_controls]

        print(f"Virtual State: {self.X0[3, 0]} || {self.X0[7, 0]} || {self.X0[11, 0]}")
        print(f"Virtual Control: {self.U0[3, 0]} || {self.U0[7, 0]} || {self.U0[11, 0]}")
        print()

        return self.U0[:n_controls]
