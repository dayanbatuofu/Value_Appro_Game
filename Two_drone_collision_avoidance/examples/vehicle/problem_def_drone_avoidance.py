import numpy as np
import torch
from examples.problem_def_template import config_prototype, problem_prototype
import math

class config_NN (config_prototype):
    def __init__(self, N_states, time_dependent):
        self.N_layers = 3
        self.N_neurons = 64
        self.layers = self.build_layers(N_states,
                                        time_dependent,
                                        self.N_layers,
                                        self.N_neurons)

        self.random_seeds = {'train': 7, 'generate': 100}

        self.ODE_solver = 'RK23'
        # Accuracy level of BVP data
        self.data_tol = 1e-3   # 1e-03
        # Max number of nodes to use in BVP
        self.max_nodes = 2500   # 800000
        # Time horizon
        self.t1 = 3.

        # Time subintervals to use in time marching
        Nt = 10  # 10
        self.tseq = np.linspace(0., self.t1, Nt+1)[1:]

        # Time step for integration and sampling
        self.dt = 1e-01
        # Standard deviation of measurement noise
        self.sigma = np.pi * 1e-02

        # Which dimensions to plot when predicting value function V(0,x)?
        # (unspecified dimensions are held at mean value)
        self.plotdims = [0, 3]

        # Number of training trajectories
        self.Ns = {'train': 1800, 'val': 100, 'test': 900}

        ##### Options for training #####
        # Number of data points to use in first training rounds
        # Set to None to use whole data set
        self.batch_size = None  # 200

        # Maximum factor to increase data set size each round
        self.Ns_scale = 2
        # Number of candidate points to pick from when selecting large gradient
        # points during adaptive sampling
        self.Ns_cand = 2
        # Maximum size of batch size to use
        self.Ns_max = 8192

        # Convergence tolerance parameter (see paper)
        self.conv_tol = 1e-03

        # maximum and minimum number of training rounds
        self.max_rounds = 1
        self.min_rounds = 1

        # List or array of weights on gradient term, length = max_rounds
        self.weight_A = [10.]  # 1
        # List or array of weights on control learning term, not used in paper
        self.weight_U = [0.]  # 0.1

        # Dictionary of options to be passed to L-BFGS-B optimizer
        # Leave empty for default values
        self.BFGS_opts = {}

class setup_problem(problem_prototype):
    def __init__(self):
        self.N_states = 6
        self.t1 = 4.  # 4

        # Parameter setting for the equation X_dot = Ax+Bu
        self.A = np.array([[0, 1], [0, 0]])
        self.B1 = np.array([[0], [0], [0], [1], [0], [0]])
        self.B2 = np.array([[0], [0], [0], [0], [1], [0]])
        self.B3 = np.array([[0], [0], [0], [0], [0], [1]])

        # Initial condition bounds (different initial setting)
        self.X0_lb = np.array([[0.], [0.], [-0.1], [2], [2], [0], [0.], [0.], [-0.1], [2], [2], [0]])
        self.X0_ub = np.array([[1.], [1.], [0.1], [4], [4], [0.1], [1.], [1.], [0.1], [4], [4], [0.1]])

        self.beta = 10000   # 10000

        # weight for terminal lose
        self.alpha = 1e-06  # 1e-06

        # Length for each vehicle
        self.L1 = 3
        self.L2 = 3

        # Width for each vehicle
        self.W1 = 1.5
        self.W2 = 1.5

        # Width for Road
        self.D1 = 3
        self.D2 = 3

        # Road length setting
        self.R1 = 5
        self.R2 = 5

        # Threshold to compute the collision
        self.threshold = 0.9

        # gravity acceleration
        self.g = 9.81  # 9.81

    def Theta_star(self, X_aug):
        '''Control as a function of the costate.'''
        # If we keep collision function L in the value cost function V, we consider dH/du = 0 and get the U*
        A = X_aug[2 * self.N_states:3 * self.N_states]
        Theta1 = np.arctan(np.matmul(self.B1.T, A) * self.g / 200)
        A = X_aug[5 * self.N_states:6 * self.N_states]
        Theta2 = np.arctan(np.matmul(self.B1.T, A) * self.g / 200)

        max_acc = 0.05
        min_acc = -0.05
        Theta1[np.where(Theta1 > max_acc)] = max_acc
        Theta1[np.where(Theta1 < min_acc)] = min_acc
        Theta2[np.where(Theta2 > max_acc)] = max_acc
        Theta2[np.where(Theta2 < min_acc)] = min_acc

        return Theta1, Theta2

    def Phi_star(self, X_aug):
        '''Control as a function of the costate.'''
        # If we keep collision function L in the value cost function V, we consider dH/du = 0 and get the U*
        A = X_aug[2 * self.N_states:3 * self.N_states]
        Phi1 = np.arctan(-np.matmul(self.B2.T, A) * self.g / 200)
        A = X_aug[5 * self.N_states:6 * self.N_states]
        Phi2 = np.arctan(-np.matmul(self.B2.T, A) * self.g / 200)

        max_acc = 0.05
        min_acc = -0.05  # -0.15 for np.tan(Phi)
        Phi1[np.where(Phi1 > max_acc)] = max_acc
        Phi1[np.where(Phi1 < min_acc)] = min_acc
        Phi2[np.where(Phi2 > max_acc)] = max_acc
        Phi2[np.where(Phi2 < min_acc)] = min_acc

        return Phi1, Phi2

    def Thrust_star(self, X_aug):
        '''Control as a function of the costate.'''
        # If we keep collision function L in the value cost function V, we consider dH/du = 0 and get the U*
        A = X_aug[2 * self.N_states:3 * self.N_states]
        Thrust1 = np.matmul(self.B3.T, A) / 2 + self.g
        A = X_aug[5 * self.N_states:6 * self.N_states]
        Thrust2 = np.matmul(self.B3.T, A) / 2 + self.g

        max_acc = 11.81
        min_acc = 7.81
        Thrust1[np.where(Thrust1 > max_acc)] = max_acc
        Thrust1[np.where(Thrust1 < min_acc)] = min_acc
        Thrust2[np.where(Thrust2 > max_acc)] = max_acc
        Thrust2[np.where(Thrust2 < min_acc)] = min_acc

        return Thrust1, Thrust2

    # Boundary function for BVP
    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:2 * self.N_states]
            XT = X_aug_T[:2 * self.N_states]
            AT = X_aug_T[2 * self.N_states:6 * self.N_states]
            VT = X_aug_T[6 * self.N_states:]

            # Boundary setting for lambda(T) when it is the final time T
            dFdXT = np.concatenate((np.array([self.alpha]),
                                    np.array([self.alpha]),
                                    np.array([-2 * (XT[2] - 0)]),
                                    np.array([-2 * (XT[3] - 0)]),
                                    np.array([-2 * (XT[4] - 0)]),
                                    np.array([-2 * (XT[5] - 0)]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([self.alpha]),
                                    np.array([self.alpha]),
                                    np.array([-2 * (XT[8] - 0)]),
                                    np.array([-2 * (XT[9] - 0)]),
                                    np.array([-2 * (XT[10] - 0)]),
                                    np.array([-2 * (XT[11] - 0)])))

            # Terminal cost in the value function, see the new version of HJI equation
            F = -np.array((self.alpha * XT[0] + self.alpha * XT[1] - (XT[2] - 0) ** 2 - (XT[3] - 0) ** 2 - (XT[4] - 0) ** 2 - (XT[5] - 0) ** 2,
                           self.alpha * XT[6] + self.alpha * XT[7] - (XT[8] - 0) ** 2 - (XT[9] - 0) ** 2 - (XT[10] - 0) ** 2 - (XT[11] - 0) ** 2))
            return np.concatenate((X0 - X0_in, AT - dFdXT, VT - F))
        return bc

    # PMP equation for BVP
    def aug_dynamics(self, t, X_aug):
        '''Evaluation of the augmented dynamics at a vector of time instances'''
        # Control as a function of the costate
        Theta1, Theta2 = self.Theta_star(X_aug)
        Phi1, Phi2 = self.Phi_star(X_aug)
        Thrust1, Thrust2 = self.Thrust_star(X_aug)

        # State for each vehicle
        X1 = X_aug[:self.N_states]
        X2 = X_aug[self.N_states:2 * self.N_states]

        s1x, s1y, s1z, v1x, v1y, v1z = X1[0], X1[1], X1[2], X1[3], X1[4], X1[5]
        s2x, s2y, s2z, v2x, v2y, v2z = X2[0], X2[1], X2[2], X2[3], X2[4], X2[5]

        # State space function: X_dot = Ax+Bu
        dXdt_1 = np.vstack((v1x,
                            v1y,
                            v1z,
                            self.g * np.tan(Theta1),
                            -self.g * np.tan(Phi1),
                            Thrust1 - self.g))
        dXdt_2 = np.vstack((v2x,
                            v2y,
                            v2z,
                            self.g * np.tan(Theta2),
                            -self.g * np.tan(Phi2),
                            Thrust2 - self.g))

        # lambda in Hamiltonian equation
        A_11 = X_aug[2 * self.N_states:3 * self.N_states]
        A_12 = X_aug[3 * self.N_states:4 * self.N_states]
        A_21 = X_aug[4 * self.N_states:5 * self.N_states]
        A_22 = X_aug[5 * self.N_states:6 * self.N_states]

        # Sigmoid function: sigmoid(x1_in)*inverse_sigmoid(x1_out)*sigmoid(x2_in)*inverse_sigmoid(x2_out)
        dx1 = torch.tensor(X1[0], requires_grad=True, dtype=torch.float32)
        dy1 = torch.tensor(X1[1], requires_grad=True, dtype=torch.float32)
        dz1 = torch.tensor(X1[2], requires_grad=True, dtype=torch.float32)
        dx2 = torch.tensor(X2[0], requires_grad=True, dtype=torch.float32)
        dy2 = torch.tensor(X2[1], requires_grad=True, dtype=torch.float32)
        dz2 = torch.tensor(X2[2], requires_grad=True, dtype=torch.float32)

        R1 = torch.tensor(self.R1, requires_grad=True, dtype=torch.float32)
        R2 = torch.tensor(self.R2, requires_grad=True, dtype=torch.float32)
        dist_diff = torch.sigmoid(-(torch.sqrt(((R1 - dx2) - dx1) ** 2 + ((R2 - dy2) - dy1) ** 2 + (dz2 - dz1) ** 2) - self.threshold) * 5)

        Collision_F_x = self.beta * dist_diff

        Collision_F_x_sum = torch.sum(Collision_F_x)
        Collision_F_x_sum.requires_grad_()

        dL1dx1 = torch.autograd.grad(Collision_F_x_sum, dx1, create_graph=True)[0].detach().numpy()
        dL1dx2 = torch.autograd.grad(Collision_F_x_sum, dx2, create_graph=True)[0].detach().numpy()
        dL2dx1 = torch.autograd.grad(Collision_F_x_sum, dx1, create_graph=True)[0].detach().numpy()
        dL2dx2 = torch.autograd.grad(Collision_F_x_sum, dx2, create_graph=True)[0].detach().numpy()

        dL1dy1 = torch.autograd.grad(Collision_F_x_sum, dy1, create_graph=True)[0].detach().numpy()
        dL1dy2 = torch.autograd.grad(Collision_F_x_sum, dy2, create_graph=True)[0].detach().numpy()
        dL2dy1 = torch.autograd.grad(Collision_F_x_sum, dy1, create_graph=True)[0].detach().numpy()
        dL2dy2 = torch.autograd.grad(Collision_F_x_sum, dy2, create_graph=True)[0].detach().numpy()

        dL1dz1 = torch.autograd.grad(Collision_F_x_sum, dz1, create_graph=True)[0].detach().numpy()
        dL1dz2 = torch.autograd.grad(Collision_F_x_sum, dz2, create_graph=True)[0].detach().numpy()
        dL2dz1 = torch.autograd.grad(Collision_F_x_sum, dz1, create_graph=True)[0].detach().numpy()
        dL2dz2 = torch.autograd.grad(Collision_F_x_sum, dz2, create_graph=True)[0].detach().numpy()

        dL1dvx1 = np.zeros(dL1dx1.shape[0], dtype=np.int32)
        dL1dvx2 = np.zeros(dL1dx2.shape[0], dtype=np.int32)
        dL2dvx1 = np.zeros(dL2dx1.shape[0], dtype=np.int32)
        dL2dvx2 = np.zeros(dL2dx2.shape[0], dtype=np.int32)

        dL1dvy1 = np.zeros(dL1dy1.shape[0], dtype=np.int32)
        dL1dvy2 = np.zeros(dL1dy2.shape[0], dtype=np.int32)
        dL2dvy1 = np.zeros(dL2dy1.shape[0], dtype=np.int32)
        dL2dvy2 = np.zeros(dL2dy2.shape[0], dtype=np.int32)

        dL1dvz1 = np.zeros(dL1dz1.shape[0], dtype=np.int32)
        dL1dvz2 = np.zeros(dL1dz2.shape[0], dtype=np.int32)
        dL2dvz1 = np.zeros(dL2dz1.shape[0], dtype=np.int32)
        dL2dvz2 = np.zeros(dL2dz2.shape[0], dtype=np.int32)

        # Jacobian of dynamic
        """
        Jac_X1 = np.array([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])

        Jac_X2 = np.array([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])
        """

        # lambda_dot in PMP equation
        element11_1 = np.zeros(dL1dx1.shape[0], dtype=np.int32)
        element11_2 = np.zeros(dL1dx1.shape[0], dtype=np.int32)
        element11_3 = np.zeros(dL1dx1.shape[0], dtype=np.int32)
        element11_4 = A_11[0, :]
        element11_5 = A_11[1, :]
        element11_6 = A_11[2, :]

        element12_1 = np.zeros(dL1dx2.shape[0], dtype=np.int32)
        element12_2 = np.zeros(dL1dx2.shape[0], dtype=np.int32)
        element12_3 = np.zeros(dL1dx2.shape[0], dtype=np.int32)
        element12_4 = A_12[0, :]
        element12_5 = A_12[1, :]
        element12_6 = A_12[2, :]

        element21_1 = np.zeros(dL2dx1.shape[0], dtype=np.int32)
        element21_2 = np.zeros(dL2dx1.shape[0], dtype=np.int32)
        element21_3 = np.zeros(dL2dx1.shape[0], dtype=np.int32)
        element21_4 = A_21[0, :]
        element21_5 = A_21[1, :]
        element21_6 = A_21[2, :]

        element22_1 = np.zeros(dL2dx2.shape[0], dtype=np.int32)
        element22_2 = np.zeros(dL2dx2.shape[0], dtype=np.int32)
        element22_3 = np.zeros(dL2dx2.shape[0], dtype=np.int32)
        element22_4 = A_22[0, :]
        element22_5 = A_22[1, :]
        element22_6 = A_22[2, :]

        dAdt_11 = -np.array([element11_1, element11_2, element11_3, element11_4, element11_5, element11_6]) \
                  + np.array([dL1dx1, dL1dy1, dL1dz1, dL1dvx1, dL1dvy1, dL1dvz1])
        dAdt_12 = -np.array([element12_1, element12_2, element12_3, element12_4, element12_5, element12_6]) \
                  + np.array([dL1dx2, dL1dy2, dL1dz2, dL1dvx2, dL1dvy2, dL1dvz2])
        dAdt_21 = -np.array([element21_1, element21_2, element21_3, element21_4, element21_5, element21_6]) \
                  + np.array([dL2dx1, dL2dy1, dL2dz1, dL2dvx1, dL2dvy1, dL2dvz1])
        dAdt_22 = -np.array([element22_1, element22_2, element22_3, element22_4, element22_5, element22_6]) \
                  + np.array([dL2dx2, dL2dy2, dL2dz2, dL2dvx2, dL2dvy2, dL2dvz2])

        # Collision function L1, L2
        L1 = (Thrust1 - self.g) ** 2 + 100 * np.tan(Theta1) ** 2 + 100 * np.tan(Phi1) ** 2 + Collision_F_x.detach().numpy()
        L2 = (Thrust2 - self.g) ** 2 + 100 * np.tan(Theta2) ** 2 + 100 * np.tan(Phi2) ** 2 + Collision_F_x.detach().numpy()

        return np.vstack((dXdt_1, dXdt_2, dAdt_11, dAdt_12, dAdt_21, dAdt_22, -L1, -L2))
