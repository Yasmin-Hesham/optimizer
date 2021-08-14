from MPCTools import *
import numpy as np

''' Preparing variables '''
# states: matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N+1)
# controls: matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)
# path parameter: TODO: revise it (N-->2)
K = ca.SX.sym('K', 1, N+1)
# parameters: column vector for storing initial state, target state, initial controls and path tracking 
P = ca.SX.sym('P', (n_states+n_controls) + (order+1)*2 + 1)

x_coeff = P[n_states+n_controls: n_states+n_controls+(order+1)]
y_coeff = P[n_states+n_controls+(order+1):-1]

''' Constructing cost function '''
cost_fn = 0  # initialize cost
# accumulate state-error cost
for i in range(N+1):
    #state_error = state - target_state
    pathX_error = X[0, i] - path_poly(
                                      K[0, i], 
                                      x_coeff
                                      )
    pathY_error = X[1, i] - path_poly(
                                      K[0, i],
                                      y_coeff
                                      )
    theta_error = X[2, i]
    # pathTheta_error = X[2, i+1] - P[2*n_states - 1]
    # add the weighted pathX error to the total cost
    cost_fn += pathX_error.T @ Q_x @ pathX_error
    cost_fn += pathY_error.T @ Q_y @ pathY_error
    cost_fn += theta_error.T @ Q_theta @ theta_error

# accumulate control-error cost
for i in range(N):
    control_error = U[:, i]  # control_error = control - (target_control=0)
    # add the weighted control error to the total cost
    cost_fn += control_error.T @ R @ control_error

# accumulate path tracking cost
path_error = 1 - K
cost_fn += (path_error @ path_error.T) * H
 

''' Constructing constraints '''
g = ca.SX()    # initialize contraints vector
lbg = ca.DM()  # initialize lowerbounds vector
ubg = ca.DM()  # initialize upperbounds vector

# adding physical constraints (next state must adhere to the robot model)
# equality constraints
g = ca.vertcat(g, X[:, 0] - P[:n_states])                                    
lbg = ca.vertcat(lbg, [0] * (n_states))
ubg = ca.vertcat(ubg, [0] * (n_states))

# path constraints
g = ca.vertcat(g, K[0, 0] - P[-1])  
lbg = ca.vertcat(lbg, [0])
ubg = ca.vertcat(ubg, [0])

# g = ca.vertcat(g, (K[0, 1:] - K[0, :-1]).T)
# lbg = ca.vertcat(lbg, [0] * (N-1))
# ubg = ca.vertcat(ubg, [ca.inf] * (N-1))

# for i in range(N):
#     # dK/dX * dX/dt = dK/dY * dY/dt
#     # dY/dK * dX/dt = dX/dK * dY/dt
#     dxdk_coeff = x_coeff[:-1] * np.arange(order, 0, -1)
#     dxdk = path_poly(K[0, i], dxdk_coeff)

#     dydk_coeff = y_coeff[:-1] * np.arange(order, 0, -1)
#     dydk = path_poly(K[0, i], dydk_coeff)

#     g = ca.vertcat(g, (dydk * U[0, i]) - (dxdk * U[1, i]))
#     lbg = ca.vertcat(lbg, [0])
#     ubg = ca.vertcat(ubg, [0])

#    # K_next = K_current + K_dot (K, V_x, x_coeff) * delta_t
#     # or
#     # K_next = K_current + K_dot (K, V_y, y_coeff) * delta_t 
#     g = ca.vertcat(g, K[0, i+1] - K[0, i] + dKdT(K[0, i], U[1, i], P[n_states+n_controls+(order+1): -1]) * sampling_time)
#     lbg = ca.vertcat(lbg, [0]) # TODO: problem here?
#     ubg = ca.vertcat(ubg, [0])

for i in range(N):
    state = X[:, i]
    controls = U[:, i]
    next_state = X[:, i+1]
    # 4th order Runge-Kutta discretization
    k1 = derivatives(state, controls)
    k2 = derivatives(state + sampling_time/2*k1, controls)
    k3 = derivatives(state + sampling_time/2*k2, controls)
    k4 = derivatives(state + sampling_time * k3, controls)
    predicted_state = state + (sampling_time / 6) * (k1 + k2*2 + k3*2 + k4)
    # add constraint for meccanum wheeled: next_state == predicted_state
    g = ca.vertcat(g, next_state - predicted_state)
    lbg = ca.vertcat(lbg, [0] * n_states)
    ubg = ca.vertcat(ubg, [0] * n_states)

# TODO: add dynamic obstacles

''' Creating the NLP Solver '''
optimization_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1)),
    K.reshape((-1, 1))
)

nlp_prob = {
    'f': cost_fn,                   # cost function
    # manipulated variables (aka optimization variables)
    'x': optimization_variables,
    'g': g,                         # constraints
    'p': P                          # parameters variables
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
    X0 = ca.DM.zeros((n_states * (N+1), 1))
    U0 = ca.DM.zeros((n_controls * N, 1))
    K0 = ca.DM.zeros((N+1, 1))

    def compute(self, initial_state, initial_controls, x_coeffs, y_coeffs, isDifferential=False):
        args['p'] = ca.vertcat(
            initial_state,          # current state
            # self.U0[:n_controls], # current controls
            initial_controls,       # current controls
            x_coeffs,               # coefficients of x in path in terms of k
            y_coeffs,               # coefficients of y in path in terms of k
            self.K0[1]
        )

        # optimization variable current state
        # 1. initial state from filtered sensor readings
        # 2. initial next states are extracted from previous optimized state
        # For the next step, the initial state is from current sensors' readings.
        # Then, previous optimization outputs are used as initialization for the next step.
        # For the last state, copy the previous cell in the state vector "X0".
        self.X0[:n_states] = initial_state
        self.X0[n_states:-n_states] = self.X0[2*n_states:]
        self.X0[-n_states:] = self.X0[-2*n_states:-n_states]

        # Same as the states.
        self.U0[:n_controls] = initial_controls
        self.U0[n_controls:-n_controls] = self.U0[2*n_controls:]
        self.U0[-n_controls:] = self.U0[-2*n_controls:-n_controls]

        self.K0[:-1] = self.K0[1:]
        self.K0[-1:] = self.K0[-2:-1]

        args['x0'] = ca.vertcat(
            self.X0,
            self.U0,
            self.K0
        ) 

        if isDifferential:
            args['lbx'][n_states*(N+1):n_states*(N+1) + n_controls*N: n_controls] = 0
            args['ubx'][n_states*(N+1):n_states*(N+1) + n_controls*N: n_controls] = 0

        else:
            args['lbx'][n_states*(N+1):n_states*(N+1) + n_controls*N: n_controls] = -vx_max
            args['ubx'][n_states*(N+1):n_states*(N+1) + n_controls*N: n_controls] = vx_max

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        self.X0 = sol['x'][:n_states * (N+1)]
        self.U0 = sol['x'][n_states * (N+1) : (n_states * (N+1)) + (n_controls * N)]
        self.K0 = sol['x'][(n_states * (N+1)) + (n_controls * N):]
        print(f"K: {self.K0[:5]}")

        return self.U0[:n_controls]
