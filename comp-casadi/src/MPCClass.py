    import numpy as np
    import casadi as ca
    from casadi import sin, cos, pi

    class MPC:
        def __init__(self):
            ''' Constants '''
            # Costs' weights
            self.Q_x = 150
            self.Q_y = 150
            self.Q_theta = 3000
            self.R1 = self.R2 = self.R3 = 0.5     # speed cost
            self.A1 = self.A2 = self.A3 = 0       # Acceleration
            # MPC parameters
            self.sampling_time = 1   # time between steps in seconds
            self.N = 100             # number of look ahead steps
            # MPC limits
            self.x_min = -2000
            self.x_max = 2000
            self.y_min = -2000
            self.y_max = 2000
            self.theta_min = -ca.inf
            self.theta_max = ca.inf
            self.vx_max =  300    # mm/sec
            self.vy_max =  300    # mm/sec
            self.w_max = 1.5      # rad/sec

            self.createSymbols()
            self.createBounds()
            self.createModel() 
            
        def createSymbols(self):
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
            self.n_states = states.numel()
            
            # control symbolic variables
            V_x = ca.SX.sym('V_x')
            V_y = ca.SX.sym('V_y')
            Omega = ca.SX.sym('Omega')
            controls = ca.vertcat(
                V_x,
                V_y,
                Omega
            )
            self.n_controls = controls.numel()
            
            # state weights matrix (Q_X, Q_Y, Q_THETA)
            self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)
            
            # controls weights matrix
            self.R = ca.diagcat(self.R1, self.R2, self.R3)
            
            # acceleration weights matrix
            self.A = ca.diagcat(self.A1, self.A2, self.A3)
            
            rot_3d_z = ca.vertcat(
                ca.horzcat(cos(theta), -sin(theta), 0),
                ca.horzcat(sin(theta),  cos(theta), 0),
                ca.horzcat(         0,           0, 1)
            )
            
            state_change_rate = rot_3d_z @ controls
            
            self.derivatives = ca.Function('derivatives', [states, controls], [state_change_rate])
            
        def createBounds(self):
            ''' Defining Upper and Lower Bounds '''
            # initialize boundaries arrays
            self.lbx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
            self.ubx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
            
            # state lower bounds
            self.lbx[0: self.n_states*(self.N+1): self.n_states] = self.x_min      # X lower bound
            self.lbx[1: self.n_states*(self.N+1): self.n_states] = self.y_min      # Y lower bound
            self.lbx[2: self.n_states*(self.N+1): self.n_states] = self.theta_min  # theta lower bound
            
            # state upper bounds
            self.ubx[0: self.n_states*(self.N+1): self.n_states] = self.x_max      # X upper bound
            self.ubx[1: self.n_states*(self.N+1): self.n_states] = self.y_max      # Y upper bound
            self.ubx[2: self.n_states*(self.N+1): self.n_states] = self.theta_max  # theta upper bound
            # control lower bounds
            self.lbx[self.n_states*(self.N+1)+0:: self.n_controls] = -self.vx_max        # Vx lower bound
            self.lbx[self.n_states*(self.N+1)+1:: self.n_controls] = -self.vy_max  # Vy lower bound
            self.lbx[self.n_states*(self.N+1)+2:: self.n_controls] = -self.w_max   # w lower bound
            
            # control upper bounds
            self.ubx[self.n_states*(self.N+1)+0:: self.n_controls] = self.vx_max        # Vx upper bound 
            self.ubx[self.n_states*(self.N+1)+1:: self.n_controls] = self.vy_max   # Vy upper bound
            self.ubx[self.n_states*(self.N+1)+2:: self.n_controls] = self.w_max    # w upper bound

        def compute(self, differentialMode=False):
            # TODO
            return
        
        def createCostFunction(self, Q, R, A):
            
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
                # state_error[2, 0] = ca.mod(state_error[2, 0] + pi, 2*pi) - pi  # error in angle is the shortest arc
                # add the weighted state error to the total cost
                cost_fn += state_error.T @ Q @ state_error

            # accumulate control-error cost
            for i in range(N):
                control_error = U[:, i]  # control_error = control - (target_control=0)
                # add the weighted control error to the total cost
                cost_fn += control_error.T @ R @ control_error

