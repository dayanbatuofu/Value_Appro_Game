import torch
from torch.utils.data import Dataset
import scipy.io
import os
import numpy as np

class IntersectionHJI_Supervised(Dataset):
    def __init__(self, seed=0):

        super().__init__()
        torch.manual_seed(0)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path1 = current_dir + '/validation_scripts/train_data/EL_data/data_train_a_a_500_safe.mat'
        train_data1 = scipy.io.loadmat(data_path1)
        self.train_data1 = train_data1

        data_path2 = current_dir + '/validation_scripts/train_data/EL_data/data_train_a_a_500_spare_safe.mat'
        train_data2 = scipy.io.loadmat(data_path2)
        self.train_data2 = train_data2

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        self.t_train1 = torch.tensor(self.train_data1['t'], dtype=torch.float32).flip(1)
        self.X_train1 = torch.tensor(self.train_data1['X'], dtype=torch.float32)
        self.V_train1 = -torch.tensor(self.train_data1['V'], dtype=torch.float32)
        self.A_train1 = -torch.tensor(self.train_data1['A'], dtype=torch.float32)
        self.constraint1 = torch.tensor(self.train_data1['C'], dtype=torch.float32)
        self.t_train2 = torch.tensor(self.train_data2['t'], dtype=torch.float32).flip(1)
        self.X_train2 = torch.tensor(self.train_data2['X'], dtype=torch.float32)
        self.V_train2 = -torch.tensor(self.train_data2['V'], dtype=torch.float32)
        self.A_train2 = -torch.tensor(self.train_data2['A'], dtype=torch.float32)
        self.constraint2 = torch.tensor(self.train_data2['C'], dtype=torch.float32)
        self.lb = torch.tensor([[15], [15], [15], [15]], dtype=torch.float32)
        self.ub = torch.tensor([[105], [32], [105], [32]], dtype=torch.float32)

        self.X_train1 = 2.0 * (self.X_train1 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train2 = 2.0 * (self.X_train2 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train = torch.cat((self.X_train1, self.X_train2), dim=1)
        self.t_train = torch.cat((self.t_train1, self.t_train2), dim=1)

        coords_1 = self.X_train.T
        coords_2 = torch.cat((coords_1[:, 2:], coords_1[:, :2]), dim=1)

        # set up ground truth for values and costates
        groundtruth_values1_1 = self.V_train1[0, :].reshape(-1, 1)
        groundtruth_values2_1 = self.V_train2[0, :].reshape(-1, 1)
        groundtruth_values1 = torch.cat((groundtruth_values1_1, groundtruth_values2_1), dim=0)
        groundtruth_values1_2 = self.V_train1[1, :].reshape(-1, 1)
        groundtruth_values2_2 = self.V_train2[1, :].reshape(-1, 1)
        groundtruth_values2 = torch.cat((groundtruth_values1_2, groundtruth_values2_2), dim=0)
        groundtruth_values = torch.cat((groundtruth_values1, groundtruth_values2), dim=0)

        groundtruth_costates1_1 = self.A_train1[:4, :].T
        groundtruth_costates2_1 = self.A_train2[:4, :].T
        groundtruth_costates1 = torch.cat((groundtruth_costates1_1, groundtruth_costates2_1), dim=0)
        groundtruth_costates1_2 = self.A_train1[4:, :].T
        groundtruth_costates2_2 = self.A_train2[4:, :].T
        groundtruth_costates2 = torch.cat((groundtruth_costates1_2, groundtruth_costates2_2), dim=0)
        groundtruth_costates = torch.cat((groundtruth_costates1, groundtruth_costates2), dim=0)

        groundtruth_constraint1 = torch.cat((self.constraint1.reshape(-1, 1),
                                             self.constraint2.reshape(-1, 1)), dim=0)
        groundtruth_constraint2 = torch.cat((self.constraint1.reshape(-1, 1),
                                             self.constraint2.reshape(-1, 1)), dim=0)
        groundtruth_constraints = torch.cat((groundtruth_constraint1, groundtruth_constraint2), dim=0)

        z1_sl = groundtruth_values1
        z2_sl = groundtruth_values2

        sl_num = coords_1.shape[0]

        label1 = torch.zeros(sl_num, 1)
        label2 = torch.ones(sl_num, 1)

        coords_1 = torch.cat((self.t_train.T, coords_1, label1, z1_sl), dim=1)
        coords_2 = torch.cat((self.t_train.T, coords_2, label2, z2_sl), dim=1)

        coords = torch.cat((coords_1, coords_2), dim=0)
        return {'coords': coords}, \
               {'groundtruth_values': groundtruth_values,
                'groundtruth_constraints': groundtruth_constraints,
                'groundtruth_costates': groundtruth_costates}


class IntersectionHJI_EL(Dataset):
    def __init__(self, numpoints, pretrain=False, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3,
                 pretrain_iters=2000, num_src_samples=1000, seed=0):

        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        self.num_states = 4

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end
        self.alpha = 1e-6
        self.L1 = 3
        self.L2 = 3
        self.W1 = 3
        self.W2 = 3
        self.R1 = 70
        self.R2 = 70
        self.epsilon = torch.tensor([4.5])   # (a,a):4.5

        # Set the seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path1 = current_dir + '/validation_scripts/train_data/EL_data/data_train_a_a_500_safe.mat'
        train_data1 = scipy.io.loadmat(data_path1)
        self.train_data1 = train_data1

        data_path2 = current_dir + '/validation_scripts/train_data/EL_data/data_train_a_a_500_spare_safe.mat'
        train_data2 = scipy.io.loadmat(data_path2)
        self.train_data2 = train_data2

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        self.t_train1 = torch.tensor(self.train_data1['t'], dtype=torch.float32).flip(1)
        self.X_train1 = torch.tensor(self.train_data1['X'], dtype=torch.float32)
        self.V_train1 = -torch.tensor(self.train_data1['V'], dtype=torch.float32)
        self.A_train1 = -torch.tensor(self.train_data1['A'], dtype=torch.float32)
        self.constraint1 = torch.tensor(self.train_data1['C'], dtype=torch.float32)
        self.t_train2 = torch.tensor(self.train_data2['t'], dtype=torch.float32).flip(1)
        self.X_train2 = torch.tensor(self.train_data2['X'], dtype=torch.float32)
        self.V_train2 = -torch.tensor(self.train_data2['V'], dtype=torch.float32)
        self.A_train2 = -torch.tensor(self.train_data2['A'], dtype=torch.float32)
        self.constraint2 = torch.tensor(self.train_data2['C'], dtype=torch.float32)
        self.lb = torch.tensor([[15], [15], [15], [15]], dtype=torch.float32)
        self.ub = torch.tensor([[105], [32], [105], [32]], dtype=torch.float32)

        self.X_train1 = 2.0 * (self.X_train1 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train2 = 2.0 * (self.X_train2 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train = torch.cat((self.X_train1, self.X_train2), dim=1)
        self.t_train = torch.cat((self.t_train1, self.t_train2), dim=1)

        # set up ground truth for values and costates
        groundtruth_values1_1 = self.V_train1[0, :].reshape(-1, 1)
        groundtruth_values2_1 = self.V_train2[0, :].reshape(-1, 1)
        groundtruth_values1 = torch.cat((groundtruth_values1_1, groundtruth_values2_1), dim=0)
        groundtruth_values1_2 = self.V_train1[1, :].reshape(-1, 1)
        groundtruth_values2_2 = self.V_train2[1, :].reshape(-1, 1)
        groundtruth_values2 = torch.cat((groundtruth_values1_2, groundtruth_values2_2), dim=0)
        groundtruth_values = torch.cat((groundtruth_values1, groundtruth_values2), dim=0)

        groundtruth_costates1_1 = self.A_train1[:4, :].T
        groundtruth_costates2_1 = self.A_train2[:4, :].T
        groundtruth_costates1 = torch.cat((groundtruth_costates1_1, groundtruth_costates2_1), dim=0)
        groundtruth_costates1_2 = self.A_train1[4:, :].T
        groundtruth_costates2_2 = self.A_train2[4:, :].T
        groundtruth_costates2 = torch.cat((groundtruth_costates1_2, groundtruth_costates2_2), dim=0)
        groundtruth_costates = torch.cat((groundtruth_costates1, groundtruth_costates2), dim=0)

        groundtruth_constraint1 = torch.cat((self.constraint1.reshape(-1, 1),
                                             self.constraint2.reshape(-1, 1)), dim=0)
        groundtruth_constraint2 = torch.cat((self.constraint1.reshape(-1, 1),
                                             self.constraint2.reshape(-1, 1)), dim=0)
        groundtruth_constraints = torch.cat((groundtruth_constraint1, groundtruth_constraint2), dim=0)

        coords1_sl = self.X_train.T
        coords2_sl = torch.cat((coords1_sl[:, 2:], coords1_sl[:, :2]), dim=1)

        sl_num = coords1_sl.shape[0]

        z1_sl = groundtruth_values1
        z2_sl = groundtruth_values2

        label1 = torch.zeros(sl_num, 1)
        label2 = torch.ones(sl_num, 1)

        coords1_sl = torch.cat((self.t_train.T, coords1_sl, label1, z1_sl), dim=1)
        coords2_sl = torch.cat((self.t_train.T, coords2_sl, label2, z2_sl), dim=1)

        coords1_el = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        coords2_el = torch.cat((coords1_el[:, 2:], coords1_el[:, :2]), dim=1)

        label1 = torch.zeros(self.numpoints, 1)
        label2 = torch.ones(self.numpoints, 1)

        self.z_1 = torch.zeros(self.numpoints, 1).uniform_(-1.05e-4, 300)
        self.z_2 = torch.zeros(self.numpoints, 1).uniform_(-1.05e-4, 300)

        coords1_el = torch.cat((coords1_el, label1, self.z_1), dim=1)
        coords2_el = torch.cat((coords2_el, label2, self.z_2), dim=1)

        time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (
                self.counter / self.full_count))

        label1 = torch.zeros(self.numpoints, 1)
        label2 = torch.ones(self.numpoints, 1)
        coords1_el = torch.cat((coords1_el, label1, self.z_1), dim=1)
        coords2_el = torch.cat((coords2_el, label2, self.z_2), dim=1)

        coords1_el = torch.cat((time, coords1_el), dim=1)
        coords2_el = torch.cat((time, coords2_el), dim=1)

        # make sure we always have training samples at the initial time
        coords1_el[-self.N_src_samples:, 0] = start_time
        coords2_el[-self.N_src_samples:, 0] = start_time

        # set up boundary condition
        # unnormalize the state for agent 1
        d11 = (coords1_el[:, 1:2] + 1) * (105 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d12 = (coords1_el[:, 3:4] + 1) * (105 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        # unnormalize the state for agent 1
        d21 = (coords2_el[:, 3:4] + 1) * (105 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d22 = (coords2_el[:, 1:2] + 1) * (105 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        gT_1 = -self.alpha * ((coords1_el[:, 1:2] + 1) * (105 - 15) / 2 + 15) + \
                             ((coords1_el[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        zT_1 = coords1_el[:, -1:]

        # state constraint for (a,a) setting
        cT_1 = self.epsilon - (torch.abs((d11 - 36.5) + (d12 - 36.5)) + torch.abs((d11 - 36.5) - (d12 - 36.5)))

        gT_2 = -self.alpha * ((coords2_el[:, 1:2] + 1) * (105 - 15) / 2 + 15) + \
                             ((coords2_el[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        zT_2 = coords2_el[:, -1:]

        # state constraint for (a,a) setting
        cT_2 = self.epsilon - (torch.abs((d21 - 36.5) + (d22 - 36.5)) + torch.abs((d21 - 36.5) - (d22 - 36.5)))

        boundary_valuesA = torch.cat((cT_1, cT_2), dim=0)
        boundary_valuesB = torch.cat((gT_1 - zT_1, gT_2 - zT_2), dim=0)

        dirichlet_mask = (coords1_el[:, 0, None] == start_time)

        if self.counter < self.full_count:
            self.counter += 1

        coords_1 = torch.cat((coords1_sl, coords1_el), dim=0)
        coords_2 = torch.cat((coords2_sl, coords2_el), dim=0)

        coords = torch.cat((coords_1, coords_2), dim=0)

        return {'coords': coords}, \
               {'groundtruth_values': groundtruth_values,
                'groundtruth_constraints': groundtruth_constraints,
                'groundtruth_costates': groundtruth_costates,
                'source_boundary_valuesA': boundary_valuesA,
                'source_boundary_valuesB': boundary_valuesB,
                'dirichlet_mask': dirichlet_mask}
