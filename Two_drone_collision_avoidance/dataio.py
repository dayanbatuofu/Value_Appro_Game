import torch
from torch.utils.data import Dataset
import scipy.io
import os

class IntersectionHJI_PINN(Dataset):
    def __init__(self, numpoints, pretrain=False, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3,
                 pretrain_iters=2000, num_src_samples=1000, seed=0):

        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        self.num_states = 12

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
        coords_2 = torch.cat((coords_1[:, 6:], coords_1[:, :6]), dim=1)

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

        # set up boundary condition
        boundary_values_1 = self.alpha * ((coords_1[:, 1:2] + 1) * (15.5 - 0) / 2 + 0) + \
                            self.alpha * ((coords_1[:, 2:3] + 1) * (15.5 - 0) / 2 + 0) - \
                            ((coords_1[:, 3:4] + 1) * (2 - (-1.8)) / 2 + (-1.8) - 0) ** 2 - \
                            ((coords_1[:, 4:5] + 1) * (4.5 - 0.3) / 2 + 0.3 - 0) ** 2 - \
                            ((coords_1[:, 5:6] + 1) * (4.5 - 0.3) / 2 + 0.3 - 0) ** 2 - \
                            ((coords_1[:, 6:7] + 1) * (1.8 - (-1.8)) / 2 + (-1.8) - 0) ** 2
        boundary_values_2 = self.alpha * ((coords_2[:, 1:2] + 1) * (16 - 0) / 2 + 0) + \
                            self.alpha * ((coords_2[:, 2:3] + 1) * (16 - 0) / 2 + 0) - \
                            ((coords_2[:, 3:4] + 1) * (2 - (-1.8)) / 2 + (-1.8) - 0) ** 2 - \
                            ((coords_2[:, 4:5] + 1) * (4.5 - 0.3) / 2 + 0.3 - 0) ** 2 - \
                            ((coords_2[:, 5:6] + 1) * (4.5 - 0.3) / 2 + 0.3 - 0) ** 2 - \
                            ((coords_2[:, 6:7] + 1) * (1.8 - (-1.8)) / 2 + (-1.8) - 0) ** 2
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
            data_path1 = current_dir + '/validation_scripts/train_data/data_train_drone_750.mat'
        else:
            data_path1 = current_dir + '/validation_scripts/train_data/data_train_drone_500.mat'
        train_data1 = scipy.io.loadmat(data_path1)
        self.train_data1 = train_data1

        data_path2 = current_dir + '/validation_scripts/train_data/data_train_drone_spare.mat'
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
        self.lb = torch.tensor([[0], [0], [-1.8], [0.3], [0.3], [-1.8],
                                [0], [0], [-1.8], [0.3], [0.3], [-1.8]], dtype=torch.float32)
        self.ub = torch.tensor([[15.5], [15.5], [2], [4.5], [4.5], [1.8],
                                [15.5], [15.5], [2], [4.5], [4.5], [1.8]], dtype=torch.float32)

        self.X_train1 = 2.0 * (self.X_train1 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train2 = 2.0 * (self.X_train2 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train = torch.cat((self.X_train1, self.X_train2), dim=1)
        self.t_train = torch.cat((self.t_train1, self.t_train2), dim=1)

        coords_1 = self.X_train.T
        coords_2 = torch.cat((coords_1[:, 6:], coords_1[:, :6]), dim=1)

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

        groundtruth_costates1_1 = self.A_train1[:12, :].T
        groundtruth_costates2_1 = self.A_train2[:12, :].T
        groundtruth_costates1 = torch.cat((groundtruth_costates1_1, groundtruth_costates2_1), dim=0)
        groundtruth_costates1_2 = self.A_train1[12:, :].T
        groundtruth_costates2_2 = self.A_train2[12:, :].T
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
        self.num_states = 12

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
        data_path1 = current_dir + '/validation_scripts/train_data/data_train_drone_500.mat'
        train_data1 = scipy.io.loadmat(data_path1)
        self.train_data1 = train_data1

        data_path2 = current_dir + '/validation_scripts/train_data/data_train_drone_spare.mat'
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
        self.lb = torch.tensor([[0], [0], [-1.8], [0.3], [0.3], [-1.8],
                                [0], [0], [-1.8], [0.3], [0.3], [-1.8]], dtype=torch.float32)
        self.ub = torch.tensor([[15.5], [15.5], [2], [4.5], [4.5], [1.8],
                                [15.5], [15.5], [2], [4.5], [4.5], [1.8]], dtype=torch.float32)

        self.X_train1 = 2.0 * (self.X_train1 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train2 = 2.0 * (self.X_train2 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train = torch.cat((self.X_train1, self.X_train2), dim=1)
        self.t_train = torch.cat((self.t_train1, self.t_train2), dim=1)

        coords_1 = self.X_train.T
        coords_2 = torch.cat((coords_1[:, 6:], coords_1[:, :6]), dim=1)

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

        groundtruth_costates1_1 = self.A_train1[:12, :].T
        groundtruth_costates2_1 = self.A_train2[:12, :].T
        groundtruth_costates1 = torch.cat((groundtruth_costates1_1, groundtruth_costates2_1), dim=0)
        groundtruth_costates1_2 = self.A_train1[12:, :].T
        groundtruth_costates2_2 = self.A_train2[12:, :].T
        groundtruth_costates2 = torch.cat((groundtruth_costates1_2, groundtruth_costates2_2), dim=0)
        groundtruth_costates = torch.cat((groundtruth_costates1, groundtruth_costates2), dim=0)

        # HJI data(sample entire state space)
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates for both agents
        coords_1 = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        coords_2 = torch.cat((coords_1[:, 6:], coords_1[:, :6]), dim=1)

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
        boundary_values_1 = self.alpha * ((coords_1_hji[:, 1:2] + 1) * (15.5 - 0) / 2 + 0) + \
                            self.alpha * ((coords_1_hji[:, 2:3] + 1) * (15.5 - 0) / 2 + 0) - \
                            ((coords_1_hji[:, 3:4] + 1) * (2 - (-1.8)) / 2 + (-1.8) - 0) ** 2 - \
                            ((coords_1_hji[:, 4:5] + 1) * (4.5 - 0.3) / 2 + 0.3 - 0) ** 2 - \
                            ((coords_1_hji[:, 5:6] + 1) * (4.5 - 0.3) / 2 + 0.3 - 0) ** 2 - \
                            ((coords_1_hji[:, 6:7] + 1) * (1.8 - (-1.8)) / 2 + (-1.8) - 0) ** 2
        boundary_values_2 = self.alpha * ((coords_2_hji[:, 1:2] + 1) * (15.5 - 0) / 2 + 0) + \
                            self.alpha * ((coords_2_hji[:, 2:3] + 1) * (15.5 - 0) / 2 + 0) - \
                            ((coords_2_hji[:, 3:4] + 1) * (2 - (-1.8)) / 2 + (-1.8) - 0) ** 2 - \
                            ((coords_2_hji[:, 4:5] + 1) * (4.5 - 0.3) / 2 + 0.3 - 0) ** 2 - \
                            ((coords_2_hji[:, 5:6] + 1) * (4.5 - 0.3) / 2 + 0.3 - 0) ** 2 - \
                            ((coords_2_hji[:, 6:7] + 1) * (1.8 - (-1.8)) / 2 + (-1.8) - 0) ** 2
        boundary_values = torch.cat((boundary_values_1, boundary_values_2), dim=0)

        dirichlet_mask = (coords_1_hji[:, 0, None] == start_time)

        if self.counter < self.full_count:
            self.counter += 1

        coords_1 = torch.cat((coords_1_supervised, coords_1_hji), dim=0)
        coords_2 = torch.cat((coords_2_supervised, coords_2_hji), dim=0)

        coords = torch.cat((coords_1, coords_2), dim=0)

        return {'coords': coords}, {'groundtruth_values': groundtruth_values, 'groundtruth_costates': groundtruth_costates,
                                    'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}

