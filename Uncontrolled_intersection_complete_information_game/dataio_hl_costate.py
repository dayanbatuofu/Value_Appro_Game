import torch
from torch.utils.data import Dataset
import scipy.io
import os

class IntersectionHJI_Supervised(Dataset):
    def __init__(self, seed=0):

        super().__init__()
        torch.manual_seed(0)

        current_dir = os.path.dirname(os.path.abspath(__file__))

        data_path1 = current_dir + '/validation_scripts/train_data/data_train_a_a_500.mat'
        train_data1 = scipy.io.loadmat(data_path1)
        self.train_data1 = train_data1

        data_path2 = current_dir + '/validation_scripts/train_data/data_train_a_a_500_spare.mat'
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
        self.lb = torch.tensor([[15], [15], [15], [15]], dtype=torch.float32)
        self.ub = torch.tensor([[105], [32], [105], [32]], dtype=torch.float32)

        self.X_train1 = 2.0 * (self.X_train1 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train2 = 2.0 * (self.X_train2 - self.lb) / (self.ub - self.lb) - 1.
        self.X_train = torch.cat((self.X_train1, self.X_train2), dim=1)
        self.t_train = torch.cat((self.t_train1, self.t_train2), dim=1)

        coords_1 = self.X_train.T
        coords_2 = torch.cat((coords_1[:, 2:], coords_1[:, :2]), dim=1)

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

        groundtruth_costates1_1 = self.A_train1[:4, :].T
        groundtruth_costates2_1 = self.A_train2[:4, :].T
        groundtruth_costates1 = torch.cat((groundtruth_costates1_1, groundtruth_costates2_1), dim=0)
        groundtruth_costates1_2 = self.A_train1[4:, :].T
        groundtruth_costates2_2 = self.A_train2[4:, :].T
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
        self.num_states = 4

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
        data_path1 = current_dir + '/validation_scripts/train_data/data_train_a_a_500.mat'
        train_data1 = scipy.io.loadmat(data_path1)
        self.train_data1 = train_data1

        data_path2 = current_dir + '/validation_scripts/train_data/data_train_a_a_500_spare.mat'
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
        self.lb = torch.tensor([[15], [15], [15], [15]], dtype=torch.float32)
        self.ub = torch.tensor([[105], [32], [105], [32]], dtype=torch.float32)

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

        groundtruth_costates1_1 = self.A_train1[:4, :].T
        groundtruth_costates2_1 = self.A_train2[:4, :].T
        groundtruth_costates1 = torch.cat((groundtruth_costates1_1, groundtruth_costates2_1), dim=0)
        groundtruth_costates1_2 = self.A_train1[4:, :].T
        groundtruth_costates2_2 = self.A_train2[4:, :].T
        groundtruth_costates2 = torch.cat((groundtruth_costates1_2, groundtruth_costates2_2), dim=0)
        groundtruth_costates = torch.cat((groundtruth_costates1, groundtruth_costates2), dim=0)

        # HJI data(sample entire state space)
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates for both agents
        coords_1 = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        coords_2 = torch.cat((coords_1[:, 2:], coords_1[:, :2]), dim=1)

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
        boundary_values_1 = self.alpha * ((coords_1_hji[:, 1:2] + 1) * (105 - 15) / 2 + 15) - \
                            ((coords_1_hji[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        boundary_values_2 = self.alpha * ((coords_2_hji[:, 1:2] + 1) * (105 - 15) / 2 + 15) - \
                            ((coords_2_hji[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        boundary_values = torch.cat((boundary_values_1, boundary_values_2), dim=0)

        dirichlet_mask = (coords_1_hji[:, 0, None] == start_time)

        if self.counter < self.full_count:
            self.counter += 1

        coords_1 = torch.cat((coords_1_supervised, coords_1_hji), dim=0)
        coords_2 = torch.cat((coords_2_supervised, coords_2_hji), dim=0)

        coords = torch.cat((coords_1, coords_2), dim=0)

        return {'coords': coords}, {'groundtruth_values': groundtruth_values, 'groundtruth_costates': groundtruth_costates,
                                    'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}

