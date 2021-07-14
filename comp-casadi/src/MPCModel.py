from MPCTools import *
import numpy as np

''' Preparing variables '''
# states: matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N+1)
# controls: matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)
# parameters: column vector for storing initial state, target state and initial controls
P = ca.SX.sym('P', n_states + n_states + n_controls)

''' Constructing cost function '''
cost_fn = 0  # initialize cost
# accumulate state-error cost
for i in range(N):
    #state_error = state - target_state
    state_error = X[:, i] - P[n_states: n_states*2]
    state_error[2, 0] = ca.mod(state_error[2, 0] + pi, 2*pi) - pi  # error in angle is the shortest arc
    # add the weighted state error to the total cost
    cost_fn += state_error.T @ Q @ state_error

# accumulate control-error cost
for i in range(N):
    control_error = U[:, i]  # control_error = control - (target_control=0)
    # add the weighted control error to the total cost
    cost_fn += control_error.T @ R @ control_error

''' Constructing constraints '''
g = ca.SX()    # initialize contraints vector
lbg = ca.DM()  # initialize lowerbounds vector
ubg = ca.DM()  # initialize upperbounds vector

# adding physical constraints (next state must adhere to the robot model)
g = ca.vertcat(g, X[:, 0] - P[:n_states])  # constraints in the equation
lbg = ca.vertcat(lbg, [0] * n_states)
ubg = ca.vertcat(ubg, [0] * n_states)
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

    # add constraint: next_state == predicted_state
    g = ca.vertcat(g, next_state - predicted_state)
    lbg = ca.vertcat(lbg, [0] * n_states)
    ubg = ca.vertcat(ubg, [0] * n_states)

# TODO: add dynamic obstacles

''' Creating the NLP Solver '''
optimization_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
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


class MPC:
    X0 = ca.DM.zeros((n_states * (N+1), 1))
    U0 = ca.DM.zeros((n_controls * N, 1))

    def compute(self, initial_state, target_state, initial_controls):
        args['p'] = ca.vertcat(
            initial_state,          # current state
            target_state,           # target state
            self.U0[:n_controls]    # current controls
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
        self.U0[:-n_controls] = self.U0[n_controls:]
        self.U0[-n_controls:] = self.U0[-2*n_controls:-n_controls]

        args['x0'] = ca.vertcat(
            self.X0,
            self.U0
        )

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        self.X0 = sol['x'][:n_states * (N+1)]
        self.U0 = sol['x'][n_states * (N+1):]

        return self.U0[:n_controls]