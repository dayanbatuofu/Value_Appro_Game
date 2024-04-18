import torch
from torch.utils.data import Dataset
import scipy.io
import os

class IntersectionHJI_EL(Dataset):
    def __init__(self, numpoints, pretrain=False, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3,
                 pretrain_iters=2000, num_src_samples=1000, seed=0):

        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        self.num_states = 8

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
        self.epsilon = torch.tensor([1.5])
        self.counter_checkpoint = 30000
        self.counter_data = 3000
        self.num_add = 0
        self.num_vio = 18000
        self.counter_next = 0
        self.num_vio_sample = 18000
        self.R = torch.tensor([70.], dtype=torch.float32)

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        if self.pretrain:
            # uniformly sample domain and include coordinates for both agents
            self.n_sample = self.numpoints - self.num_vio - 2000

            coords_11 = torch.cat((torch.zeros(self.num_vio, 1).uniform_(-0.51, -0.43),  # x=[33.5,36.5], y=[33.5,36.5]
                                   torch.zeros(self.num_vio, 1).uniform_(-0.5, 0.5),
                                   torch.zeros(self.num_vio, 1).uniform_(-1, 1),
                                   torch.zeros(self.num_vio, 1).uniform_(-1, 1),
                                   torch.zeros(self.num_vio, 1).uniform_(-0.51, -0.43),
                                   torch.zeros(self.num_vio, 1).uniform_(-0.5, 0.5),
                                   torch.zeros(self.num_vio, 1).uniform_(-1, 1),
                                   torch.zeros(self.num_vio, 1).uniform_(-1, 1)), dim=1)
            coords_12 = torch.zeros(self.n_sample, self.num_states).uniform_(-1, 1)
            coords_13 = torch.cat((torch.zeros(2000, 1).uniform_(-0.51, -0.43),
                                   torch.zeros(2000, 1).uniform_(-0.5, 0.5),
                                   torch.zeros(2000, 1).uniform_(-1, 1),
                                   torch.zeros(2000, 1).uniform_(-1, 1),
                                   torch.zeros(2000, 1).uniform_(-0.51, -0.43),
                                   torch.zeros(2000, 1).uniform_(-0.5, 0.5),
                                   torch.zeros(2000, 1).uniform_(-1, 1),
                                   torch.zeros(2000, 1).uniform_(-1, 1)), dim=1)
            z_1 = torch.zeros(self.numpoints, 1).uniform_(-9e-5, 300)
            coords_21 = torch.cat((coords_11[:, 4:], coords_11[:, :4]), dim=1)
            coords_22 = torch.cat((coords_12[:, 4:], coords_12[:, :4]), dim=1)
            coords_23 = torch.cat((coords_13[:, 4:], coords_13[:, :4]), dim=1)
            z_2 = torch.zeros(self.numpoints, 1).uniform_(-9e-5, 300)

            coords_1 = torch.cat((coords_11, coords_12, coords_13), dim=0)
            coords_2 = torch.cat((coords_21, coords_22, coords_23), dim=0)

            coords_1 = torch.cat((coords_1, z_1), dim=1)
            coords_2 = torch.cat((coords_2, z_2), dim=1)

        else:
            if not self.counter % self.counter_checkpoint and (self.counter + 1):
                self.num_vio_sample = int(0.2 * self.numpoints)
                self.numpoints = self.numpoints + 55000
                self.num_vio = int(0.2 * self.numpoints)
                print(self.numpoints)

            self.n_sample = self.numpoints - self.num_vio - 2000

            if not self.counter % self.counter_data and (self.counter + 1):
                self.coords_11 = torch.cat(
                    (torch.zeros(self.num_vio, 1).uniform_(-0.51, -0.43),  # x=[33.5,36.5], y=[33.5,36.5]
                     torch.zeros(self.num_vio, 1).uniform_(-0.5, 0.5),
                     torch.zeros(self.num_vio, 1).uniform_(-1, 1),
                     torch.zeros(self.num_vio, 1).uniform_(-1, 1),
                     torch.zeros(self.num_vio, 1).uniform_(-0.51, -0.43),
                     torch.zeros(self.num_vio, 1).uniform_(-0.5, 0.5),
                     torch.zeros(self.num_vio, 1).uniform_(-1, 1),
                     torch.zeros(self.num_vio, 1).uniform_(-1, 1)), dim=1)
                self.coords_12 = torch.zeros(self.n_sample, self.num_states).uniform_(-1, 1)
                self.coords_13 = torch.cat((torch.zeros(2000, 1).uniform_(-0.51, -0.43),
                                            torch.zeros(2000, 1).uniform_(-0.5, 0.5),
                                            torch.zeros(2000, 1).uniform_(-1, 1),
                                            torch.zeros(2000, 1).uniform_(-1, 1),
                                            torch.zeros(2000, 1).uniform_(-0.51, -0.43),
                                            torch.zeros(2000, 1).uniform_(-0.5, 0.5),
                                            torch.zeros(2000, 1).uniform_(-1, 1),
                                            torch.zeros(2000, 1).uniform_(-1, 1)), dim=1)
                self.z_1 = torch.zeros(self.numpoints, 1).uniform_(-9e-5, 300)
                self.coords_21 = torch.cat((self.coords_11[:, 4:], self.coords_11[:, :4]), dim=1)
                self.coords_22 = torch.cat((self.coords_12[:, 4:], self.coords_12[:, :4]), dim=1)
                self.coords_23 = torch.cat((self.coords_13[:, 4:], self.coords_13[:, :4]), dim=1)
                self.z_2 = torch.zeros(self.numpoints, 1).uniform_(-9e-5, 300)

            coords_1 = torch.cat((self.coords_11, self.coords_12, self.coords_13), dim=0)
            coords_2 = torch.cat((self.coords_21, self.coords_22, self.coords_23), dim=0)

            coords_1 = torch.cat((coords_1, self.z_1), dim=1)
            coords_2 = torch.cat((coords_2, self.z_2), dim=1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords_1 = torch.cat((time, coords_1), dim=1)
            coords_2 = torch.cat((time, coords_2), dim=1)

        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            if not self.counter % self.counter_data and (self.counter + 1):
                if not self.counter % self.counter_checkpoint and (self.counter + 1):
                    self.counter_next = self.counter + self.counter_checkpoint
                    print(self.counter_next)
                else:
                    pass
                self.time_horizon = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (
                        self.counter_next / self.full_count))

            time = self.time_horizon

            coords_1 = torch.cat((time, coords_1), dim=1)
            coords_2 = torch.cat((time, coords_2), dim=1)

            # make sure we always have training samples at the initial time
            coords_1[-self.N_src_samples:, 0] = start_time
            coords_2[-self.N_src_samples:, 0] = start_time

        # set up boundary condition
        dx_11 = (coords_1[:, 1:2] + 1) * (90 - 15) / 2 + 15
        dy_11 = (coords_1[:, 2:3] + 1) * (38 - 32) / 2 + 32
        dx_12 = (coords_1[:, 5:6] + 1) * (90 - 15) / 2 + 15
        dy_12 = (coords_1[:, 6:7] + 1) * (38 - 32) / 2 + 32

        dx_21 = (coords_2[:, 5:6] + 1) * (90 - 15) / 2 + 15
        dy_21 = (coords_2[:, 6:7] + 1) * (38 - 32) / 2 + 32
        dx_22 = (coords_2[:, 1:2] + 1) * (90 - 15) / 2 + 15
        dy_22 = (coords_2[:, 2:3] + 1) * (38 - 32) / 2 + 32

        gT_1 = -self.alpha * ((coords_1[:, 1:2] + 1) * (90 - 15) / 2 + 15) + \
               ((coords_1[:, 4:5] + 1) * (25 - 18) / 2 + 18 - 18) ** 2 + \
               ((coords_1[:, 2:3] + 1) * (38 - 32) / 2 + 32 - 35) ** 2
        zT_1 = coords_1[:, 9:]
        cT_1 = self.epsilon - torch.sqrt(((self.R - dx_12) - dx_11) ** 2 + (dy_12 - dy_11) ** 2)

        gT_2 = -self.alpha * ((coords_2[:, 1:2] + 1) * (90 - 15) / 2 + 15) + \
               ((coords_2[:, 4:5] + 1) * (25 - 18) / 2 + 18 - 18) ** 2 + \
               ((coords_2[:, 2:3] + 1) * (38 - 32) / 2 + 32 - 35) ** 2
        zT_2 = coords_2[:, 9:]
        cT_2 = self.epsilon - torch.sqrt(((self.R - dx_22) - dx_21) ** 2 + (dy_22 - dy_21) ** 2)

        boundary_valuesA = torch.cat((cT_1, cT_2), dim=0)
        boundary_valuesB = torch.cat((gT_1 - zT_1, gT_2 - zT_2), dim=0)

        if self.pretrain:
            dirichlet_mask = torch.ones(coords_1.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords_1[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        coords = torch.cat((coords_1, coords_2), dim=0)
        return {'coords': coords}, {'source_boundary_valuesA': boundary_valuesA,
                                    'source_boundary_valuesB': boundary_valuesB,
                                    'dirichlet_mask': dirichlet_mask,
                                    'counter': self.counter,
                                    'full_count': self.full_count,
                                    'tMin': self.tMin,
                                    'tMax': self.tMax,
                                    'N_src_samples': self.N_src_samples,
                                    'num_vio': self.num_vio,
                                    'counter_checkpoint': self.counter_checkpoint,
                                    'counter_next': self.counter_next,
                                    'num_points': self.numpoints,
                                    'num_vio_sample': self.num_vio_sample,
                                    'counter_data': self.counter_data}


class IntersectionHJI_PINN(Dataset):
    def __init__(self, numpoints, pretrain=False, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3,
                 pretrain_iters=2000, num_src_samples=1000, seed=0):

        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        self.num_states = 8

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end
        self.alpha = 1e-6

        self.spike_iters = 50
        self.spike_counter = 1

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates for both agents
        coords_1 = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        coords_2 = torch.cat((coords_1[:, 4:], coords_1[:, :4]), dim=1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords_1 = torch.cat((time, coords_1), dim=1)
            coords_2 = torch.cat((time, coords_2), dim=1)

        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (
                    self.counter / self.full_count))

            coords_1 = torch.cat((time, coords_1), dim=1)
            coords_2 = torch.cat((time, coords_2), dim=1)

            # make sure we always have training samples at the initial time
            coords_1[-self.N_src_samples:, 0] = start_time
            coords_2[-self.N_src_samples:, 0] = start_time

        # v0 = torch.zeros(self.numpoints, 1).uniform_(18, 25)

        # set up boundary condition
        boundary_values_1 = self.alpha * ((coords_1[:, 1:2] + 1) * (90 - 15) / 2 + 15) - \
                            ((coords_1[:, 4:5] + 1) * (25 - 18) / 2 + 18 - 18) ** 2 - \
                            ((coords_2[:, 2:3] + 1) * (38 - 32) / 2 + 32 - 35) ** 2
        boundary_values_2 = self.alpha * ((coords_2[:, 1:2] + 1) * (90 - 15) / 2 + 15) - \
                            ((coords_2[:, 4:5] + 1) * (25 - 18) / 2 + 18 - 18) ** 2 - \
                            ((coords_2[:, 2:3] + 1) * (38 - 32) / 2 + 32 - 35) ** 2
        boundary_values = torch.cat((boundary_values_1, boundary_values_2), dim=0)

        if self.pretrain:
            dirichlet_mask = torch.ones(coords_1.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords_1[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        coords = torch.cat((coords_1, coords_2), dim=0)
        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}


class IntersectionHJI_Supervised(Dataset):
    def __init__(self, Hybrid_use, seed=0):

        super().__init__()
        torch.manual_seed(0)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not Hybrid_use:
            data_path1 = current_dir + '/validation_scripts/train_data/data_train_HD_950.mat'
        else:
            data_path1 = current_dir + '/validation_scripts/train_data/data_train_HD_500.mat'
        train_data1 = scipy.io.loadmat(data_path1)
        self.train_data1 = train_data1

        data_path2 = current_dir + '/validation_scripts/train_data/data_train_HD_spare.mat'
        train_data2 = scipy.io.loadmat(data_path2)
        self.train_data2 = train_data2

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        self.t_train1 = torch.tensor(self.train_data1['t'], dtype=torch.float32).flip(1)
        self.X_train1 = torch.tensor(self.train_data1['X'], dtype=torch.float32)
        self.A_train1 = torch.tensor(self.train_data1['A'], dtype=torch.float32)
        self.V_train1 = torch.tensor(self.train_data1['V'], dtype=torch.float32)
        self.t_train2 = torch.tensor(self.train_data2['t'], dtype=torch.float32).flip(1)
        self.X_train2 = torch.tensor(self.train_data2['X'], dtype=torch.float32)
        self.A_train2 = torch.tensor(self.train_data2['A'], dtype=torch.float32)
        self.V_train2 = torch.tensor(self.train_data2['V'], dtype=torch.float32)
        self.lb = torch.tensor([[15], [32], [-0.15], [18], [15], [32], [-0.15], [18]], dtype=torch.float32)
        self.ub = torch.tensor([[90], [38], [0.18], [25], [90], [38], [0.18], [25]], dtype=torch.float32)

        self.X_train1 = 2.0 * (self.X_train1 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train2 = 2.0 * (self.X_train2 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train = torch.cat((self.X_train1, self.X_train2), dim=1)
        self.t_train = torch.cat((self.t_train1, self.t_train2), dim=1)

        coords_1 = self.X_train.T
        coords_2 = torch.cat((coords_1[:, 4:], coords_1[:, :4]), dim=1)

        coords_1 = torch.cat((self.t_train.T, coords_1), dim=1)
        coords_2 = torch.cat((self.t_train.T, coords_2), dim=1)

        # set up ground truth for values and costates
        groundtruth_values1_1 = self.V_train1[0, :].reshape(-1, 1)
        groundtruth_values2_1 = self.V_train2[0, :].reshape(-1, 1)
        groundtruth_values1 = torch.cat((groundtruth_values1_1, groundtruth_values2_1), dim=0)
        groundtruth_values1_2 = self.V_train1[1, :].reshape(-1, 1)
        groundtruth_values2_2 = self.V_train2[1, :].reshape(-1, 1)
        groundtruth_values2 = torch.cat((groundtruth_values1_2, groundtruth_values2_2), dim=0)
        groundtruth_values = torch.cat((groundtruth_values1, groundtruth_values2), dim=0)

        groundtruth_costates1_1 = self.A_train1[:8, :].T
        groundtruth_costates2_1 = self.A_train2[:8, :].T
        groundtruth_costates1 = torch.cat((groundtruth_costates1_1, groundtruth_costates2_1), dim=0)
        groundtruth_costates1_2 = self.A_train1[8:, :].T
        groundtruth_costates2_2 = self.A_train2[8:, :].T
        groundtruth_costates2 = torch.cat((groundtruth_costates1_2, groundtruth_costates2_2), dim=0)
        groundtruth_costates = torch.cat((groundtruth_costates1, groundtruth_costates2), dim=0)

        coords = torch.cat((coords_1, coords_2), dim=0)
        return {'coords': coords}, {'groundtruth_values': groundtruth_values, 'groundtruth_costates': groundtruth_costates}


class IntersectionHJI_Hybrid(Dataset):
    def __init__(self, numpoints, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3, num_src_samples=1000,
                 seed=0):

        super().__init__()
        torch.manual_seed(0)

        self.numpoints = numpoints
        self.num_states = 8

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.full_count = counter_end
        self.alpha = 1e-6

        # Set the seed
        torch.manual_seed(seed)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path1 = current_dir + '/validation_scripts/train_data/data_train_HD_500.mat'
        train_data1 = scipy.io.loadmat(data_path1)
        self.train_data1 = train_data1

        data_path2 = current_dir + '/validation_scripts/train_data/data_train_HD_spare.mat'
        train_data2 = scipy.io.loadmat(data_path2)
        self.train_data2 = train_data2

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # supervised learning data
        self.t_train1 = torch.tensor(self.train_data1['t'], dtype=torch.float32).flip(1)
        self.X_train1 = torch.tensor(self.train_data1['X'], dtype=torch.float32)
        self.A_train1 = torch.tensor(self.train_data1['A'], dtype=torch.float32)
        self.V_train1 = torch.tensor(self.train_data1['V'], dtype=torch.float32)
        self.t_train2 = torch.tensor(self.train_data2['t'], dtype=torch.float32).flip(1)
        self.X_train2 = torch.tensor(self.train_data2['X'], dtype=torch.float32)
        self.A_train2 = torch.tensor(self.train_data2['A'], dtype=torch.float32)
        self.V_train2 = torch.tensor(self.train_data2['V'], dtype=torch.float32)
        self.lb = torch.tensor([[15], [32], [-0.15], [18], [15], [32], [-0.15], [18]], dtype=torch.float32)
        self.ub = torch.tensor([[90], [38], [0.18], [25], [90], [38], [0.18], [25]], dtype=torch.float32)

        self.X_train1 = 2.0 * (self.X_train1 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train2 = 2.0 * (self.X_train2 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train = torch.cat((self.X_train1, self.X_train2), dim=1)
        self.t_train = torch.cat((self.t_train1, self.t_train2), dim=1)

        coords_1 = self.X_train.T
        coords_2 = torch.cat((coords_1[:, 2:], coords_1[:, :2]), dim=1)

        coords_1_supervised = torch.cat((self.t_train.T, coords_1), dim=1)
        coords_2_supervised = torch.cat((self.t_train.T, coords_2), dim=1)

        # set up ground truth for values and costates
        groundtruth_values1_1 = self.V_train1[0, :].reshape(-1, 1)
        groundtruth_values2_1 = self.V_train2[0, :].reshape(-1, 1)
        groundtruth_values1 = torch.cat((groundtruth_values1_1, groundtruth_values2_1), dim=0)
        groundtruth_values1_2 = self.V_train1[1, :].reshape(-1, 1)
        groundtruth_values2_2 = self.V_train2[1, :].reshape(-1, 1)
        groundtruth_values2 = torch.cat((groundtruth_values1_2, groundtruth_values2_2), dim=0)
        groundtruth_values = torch.cat((groundtruth_values1, groundtruth_values2), dim=0)

        groundtruth_costates1_1 = self.A_train1[:8, :].T
        groundtruth_costates2_1 = self.A_train2[:8, :].T
        groundtruth_costates1 = torch.cat((groundtruth_costates1_1, groundtruth_costates2_1), dim=0)
        groundtruth_costates1_2 = self.A_train1[8:, :].T
        groundtruth_costates2_2 = self.A_train2[8:, :].T
        groundtruth_costates2 = torch.cat((groundtruth_costates1_2, groundtruth_costates2_2), dim=0)
        groundtruth_costates = torch.cat((groundtruth_costates1, groundtruth_costates2), dim=0)

        # HJI data(sample entire state space)
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates for both agents
        coords_1 = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        coords_2 = torch.cat((coords_1[:, 4:], coords_1[:, :4]), dim=1)

        # slowly grow time values from start time
        # this currently assumes start_time = 0 and max time value is tMax
        time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (
                self.counter / self.full_count))

        coords_1_hji = torch.cat((time, coords_1), dim=1)
        coords_2_hji = torch.cat((time, coords_2), dim=1)

        # make sure we always have training samples at the initial time
        coords_1_hji[-self.N_src_samples:, 0] = start_time
        coords_2_hji[-self.N_src_samples:, 0] = start_time

        # set up boundary condition: V(T) = alpha*X(T) - (V(T) - V(0))^2
        boundary_values_1 = self.alpha * ((coords_1_hji[:, 1:2] + 1) * (90 - 15) / 2 + 15) - \
                            ((coords_1_hji[:, 4:5] + 1) * (25 - 18) / 2 + 18 - 18) ** 2 - \
                            ((coords_1_hji[:, 2:3] + 1) * (38 - 32) / 2 + 32 - 35) ** 2
        boundary_values_2 = self.alpha * ((coords_2_hji[:, 1:2] + 1) * (90 - 15) / 2 + 15) - \
                            ((coords_2_hji[:, 4:5] + 1) * (25 - 18) / 2 + 18 - 18) ** 2 - \
                            ((coords_2_hji[:, 2:3] + 1) * (38 - 32) / 2 + 32 - 35) ** 2
        boundary_values = torch.cat((boundary_values_1, boundary_values_2), dim=0)

        dirichlet_mask = (coords_1_hji[:, 0, None] == start_time)

        if self.counter < self.full_count:
            self.counter += 1

        coords_1 = torch.cat((coords_1_supervised, coords_1_hji), dim=0)
        coords_2 = torch.cat((coords_2_supervised, coords_2_hji), dim=0)

        coords = torch.cat((coords_1, coords_2), dim=0)

        return {'coords': coords}, {'groundtruth_values': groundtruth_values, 'groundtruth_costates': groundtruth_costates,
                                    'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}

