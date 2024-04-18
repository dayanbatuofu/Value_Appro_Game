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

        self.random_seeds = {'train': 7, 'generate': 200}

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
        self.Ns = {'train': 100, 'val': 100, 'test': 900}

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
        self.N_states = 4
        self.t1 = 4.

        # Parameter setting for the equation X_dot = Ax+Bu
        self.A = np.array([[0, 1], [0, 0]])
        self.B1 = np.array([[0], [0], [1], [0]])
        self.B2 = np.array([[0], [0], [0], [1]])

        # Initial condition bounds (different initial setting)
        self.X0_lb = np.array([[0.], [32.25], [-np.pi/180], [18.], [0.], [36.25], [-np.pi/180], [18.]])
        self.X0_ub = np.array([[3.], [33.75], [np.pi/180], [25.], [3.], [37.75], [np.pi/180], [25.]])

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
        self.D1 = 4
        self.D2 = 4

        # Road length setting
        self.R = torch.tensor(70, requires_grad=True, dtype=torch.float32)

        # Threshold to compute the collision
        self.threshold = 2.5  # 3 * math.sqrt(2)

        # weigth for omega
        self.omega_weight = 100

    def U_star(self, X_aug):
        '''Control as a function of the costate.'''
        # If we keep collision function L in the value cost function V, we consider dH/du = 0 and get the U*
        A = X_aug[2 * self.N_states:3 * self.N_states]
        U1 = np.matmul(self.B2.T, A) / 2
        A = X_aug[5 * self.N_states:6 * self.N_states]
        U2 = np.matmul(self.B2.T, A) / 2

        max_acc = 10
        min_acc = -5
        U1[np.where(U1 > max_acc)] = max_acc
        U1[np.where(U1 < min_acc)] = min_acc
        U2[np.where(U2 > max_acc)] = max_acc
        U2[np.where(U2 < min_acc)] = min_acc

        return U1, U2

    def Omega_star(self, X_aug):
        '''Control as a function of the costate.'''
        # If we keep collision function L in the value cost function V, we consider dH/du = 0 and get the U*
        A = X_aug[2 * self.N_states:3 * self.N_states]
        Omega1 = np.matmul(self.B1.T, A) / (2 * self.omega_weight)
        A = X_aug[5 * self.N_states:6 * self.N_states]
        Omega2 = np.matmul(self.B1.T, A) / (2 * self.omega_weight)

        max_acc = 1
        min_acc = -1
        Omega1[np.where(Omega1 > max_acc)] = max_acc
        Omega1[np.where(Omega1 < min_acc)] = min_acc
        Omega2[np.where(Omega2 > max_acc)] = max_acc
        Omega2[np.where(Omega2 < min_acc)] = min_acc

        return Omega1, Omega2

    # Boundary function for BVP
    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:2 * self.N_states]
            XT = X_aug_T[:2 * self.N_states]
            AT = X_aug_T[2 * self.N_states:6 * self.N_states]
            VT = X_aug_T[6 * self.N_states:]

            # Boundary setting for lambda(T) when it is the final time T
            dFdXT = np.concatenate((np.array([self.alpha]),
                                    np.array([-2 * (XT[1] - 37)]),
                                    np.array([-200 * (XT[2] - 0)]),
                                    np.array([-2 * (XT[3] - 18)]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([0]),
                                    np.array([self.alpha]),
                                    np.array([-2 * (XT[5] - 33)]),
                                    np.array([-200 * (XT[6] - 0)]),
                                    np.array([-2 * (XT[7] - 18)])))

            # Terminal cost in the value function, see the new version of HJI equation
            F = -np.array((self.alpha * XT[0] - (XT[1] - 37) ** 2 - 100 * (XT[2] - 0) ** 2 - (XT[3] - 18) ** 2,
                           self.alpha * XT[4] - (XT[5] - 33) ** 2 - 100 * (XT[6] - 0) ** 2 - (XT[7] - 18) ** 2))

            return np.concatenate((X0 - X0_in, AT - dFdXT, VT - F))
        return bc

    # PMP equation for BVP
    def aug_dynamics(self, t, X_aug):
        '''Evaluation of the augmented dynamics at a vector of time instances'''
        # Control as a function of the costate
        U1, U2 = self.U_star(X_aug)
        Omega1, Omega2 = self.Omega_star(X_aug)

        # State for each vehicle
        X1 = X_aug[:self.N_states]
        X2 = X_aug[self.N_states:2 * self.N_states]

        s1x, s1y, ang1, v1, = X1[0], X1[1], X1[2], X1[3]
        s2x, s2y, ang2, v2, = X2[0], X2[1], X2[2], X2[3]

        # State space function: X_dot = Ax+Bu
        dXdt_1 = np.vstack((v1 * np.cos(ang1), v1 * np.sin(ang1), Omega1, U1))
        dXdt_2 = np.vstack((v2 * np.cos(ang2), v2 * np.sin(ang2), Omega2, U2))

        # lambda in Hamiltonian equation
        A_11 = X_aug[2 * self.N_states:3 * self.N_states]
        A_12 = X_aug[3 * self.N_states:4 * self.N_states]
        A_21 = X_aug[4 * self.N_states:5 * self.N_states]
        A_22 = X_aug[5 * self.N_states:6 * self.N_states]

        # Sigmoid function: sigmoid(x1_in)*inverse_sigmoid(x1_out)*sigmoid(x2_in)*inverse_sigmoid(x2_out)
        dx1 = torch.tensor(X1[0], requires_grad=True, dtype=torch.float32)
        dy1 = torch.tensor(X1[1], requires_grad=True, dtype=torch.float32)
        dx2 = torch.tensor(X2[0], requires_grad=True, dtype=torch.float32)
        dy2 = torch.tensor(X2[1], requires_grad=True, dtype=torch.float32)

        dist_diff = torch.sigmoid(-(torch.sqrt((dx2 - dx1) ** 2 + (dy2 - dy1) ** 2) - self.threshold) * 5)

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

        dL1dtheta1 = np.zeros(dL1dx1.shape[0], dtype=np.int32)
        dL1dtheta2 = np.zeros(dL1dx2.shape[0], dtype=np.int32)
        dL2dtheta1 = np.zeros(dL2dx1.shape[0], dtype=np.int32)
        dL2dtheta2 = np.zeros(dL2dx2.shape[0], dtype=np.int32)

        dL1dv1 = np.zeros(dL1dx1.shape[0], dtype=np.int32)
        dL1dv2 = np.zeros(dL1dx2.shape[0], dtype=np.int32)
        dL2dv1 = np.zeros(dL2dx1.shape[0], dtype=np.int32)
        dL2dv2 = np.zeros(dL2dx2.shape[0], dtype=np.int32)

        # Jacobian of dynamic
        """
        Jac_X1 = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [-v1 * np.sin(ang1), v1 * np.cos(ang1), 0, 0],
                           [np.cos(ang1), np.sin(ang1), 0, 0]])

        Jac_X2 = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [-v2 * np.sin(ang2), v1 * np.cos(ang2), 0, 0],
                           [np.cos(ang2), np.sin(ang2), 0, 0]])
        """

        # lambda_dot in PMP equation
        element11_1 = np.zeros(dL1dx1.shape[0], dtype=np.int32)
        element11_2 = np.zeros(dL1dx1.shape[0], dtype=np.int32)
        element11_3 = -v1 * np.sin(ang1) * A_11[0, :] + v1 * np.cos(ang1) * A_11[1, :]
        element11_4 = np.cos(ang1) * A_11[0, :] + np.sin(ang1) * A_11[1, :]

        element12_1 = np.zeros(dL1dx2.shape[0], dtype=np.int32)
        element12_2 = np.zeros(dL1dx2.shape[0], dtype=np.int32)
        element12_3 = -v2 * np.sin(ang2) * A_12[0, :] + v2 * np.cos(ang2) * A_12[1, :]
        element12_4 = np.cos(ang2) * A_12[0, :] + np.sin(ang2) * A_12[1, :]

        element21_1 = np.zeros(dL2dx1.shape[0], dtype=np.int32)
        element21_2 = np.zeros(dL2dx1.shape[0], dtype=np.int32)
        element21_3 = -v1 * np.sin(ang1) * A_21[0, :] + v1 * np.cos(ang1) * A_21[1, :]
        element21_4 = np.cos(ang1) * A_21[0, :] + np.sin(ang1) * A_21[1, :]

        element22_1 = np.zeros(dL2dx2.shape[0], dtype=np.int32)
        element22_2 = np.zeros(dL2dx2.shape[0], dtype=np.int32)
        element22_3 = -v2 * np.sin(ang2) * A_22[0, :] + v2 * np.cos(ang2) * A_22[1, :]
        element22_4 = np.cos(ang2) * A_22[0, :] + np.sin(ang2) * A_22[1, :]

        dAdt_11 = -np.array([element11_1, element11_2, element11_3, element11_4]) + np.array([dL1dx1, dL1dy1, dL1dtheta1, dL1dv1])
        dAdt_12 = -np.array([element12_1, element12_2, element12_3, element12_4]) + np.array([dL1dx2, dL1dy2, dL1dtheta2, dL1dv2])
        dAdt_21 = -np.array([element21_1, element21_2, element21_3, element21_4]) + np.array([dL2dx1, dL2dy1, dL2dtheta1, dL2dv1])
        dAdt_22 = -np.array([element22_1, element22_2, element22_3, element22_4]) + np.array([dL2dx2, dL2dy2, dL2dtheta2, dL2dv2])

        # Collision function L1, L2
        L1 = U1 ** 2 + Collision_F_x.detach().numpy() + self.omega_weight * Omega1 ** 2
        L2 = U2 ** 2 + Collision_F_x.detach().numpy() + self.omega_weight * Omega2 ** 2

        return np.vstack((dXdt_1, dXdt_2, dAdt_11, dAdt_12, dAdt_21, dAdt_22, -L1, -L2))
