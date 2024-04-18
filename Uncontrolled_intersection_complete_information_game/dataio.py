import torch
from torch.utils.data import Dataset
import scipy.io
import os

class IntersectionHJI_EL(Dataset):
    def __init__(self, N_choice, numpoints, pretrain=False, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3,
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
        self.N_choice = N_choice
        if N_choice == 0:
            self.epsilon = torch.tensor([4.5])   # (a,a)
        if N_choice == 1:
            self.epsilon = torch.tensor([7.5])   # (a,na)
        if N_choice == 2:
            self.epsilon = torch.tensor([7.5])   # (na,a)
        if N_choice == 3:
            self.epsilon = torch.tensor([7.5])   # (na,na)
        self.counter_checkpoint = 30000
        self.counter_data = 3000
        self.num_add = 0
        self.num_vio = 18000
        self.counter_next = 0
        self.num_vio_sample = 18000

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        if self.pretrain:
            # uniformly sample domain and include coordinates for both agents
            # (a,a): (-0.573, -0.472); (a,na): (-0.573, -0.472)/(-0.638, -0.472);
            # (na,a): (-0.638, -0.472)/(-0.573, -0.472); (na,na): (-0.638, -0.472)
            if self.N_choice == 0:
                self.d1_min, self.d1_max = -0.573, -0.472
                self.d2_min, self.d2_max = -0.573, -0.472
            if self.N_choice == 1:
                self.d1_min, self.d1_max = -0.573, -0.472
                self.d2_min, self.d2_max = -0.638, -0.472
            if self.N_choice == 2:
                self.d1_min, self.d1_max = -0.638, -0.472
                self.d2_min, self.d2_max = -0.573, -0.472
            if self.N_choice == 3:
                self.d1_min, self.d1_max = -0.638, -0.472
                self.d2_min, self.d2_max = -0.638, -0.472

            self.n_sample = self.numpoints - self.num_vio - 2000

            coords_11 = torch.cat((torch.zeros(self.num_vio, 1).uniform_(self.d1_min, self.d1_max),
                                   torch.zeros(self.num_vio, 1).uniform_(-1, 1),
                                   torch.zeros(self.num_vio, 1).uniform_(self.d2_min, self.d2_max),
                                   torch.zeros(self.num_vio, 1).uniform_(-1, 1)), dim=1)
            coords_12 = torch.zeros(self.n_sample, self.num_states).uniform_(-1, 1)
            coords_13 = torch.cat((torch.zeros(2000, 1).uniform_(self.d1_min, self.d1_max),
                                   torch.zeros(2000, 1).uniform_(-1, 1),
                                   torch.zeros(2000, 1).uniform_(self.d2_min, self.d2_max),
                                   torch.zeros(2000, 1).uniform_(-1, 1)), dim=1)
            z_1 = torch.zeros(self.numpoints, 1).uniform_(-1.05e-4, 300)
            coords_21 = torch.cat((coords_11[:, 2:], coords_11[:, :2]), dim=1)
            coords_22 = torch.cat((coords_12[:, 2:], coords_12[:, :2]), dim=1)
            coords_23 = torch.cat((coords_13[:, 2:], coords_13[:, :2]), dim=1)
            z_2 = torch.zeros(self.numpoints, 1).uniform_(-1.05e-4, 300)

            coords_1 = torch.cat((coords_11, coords_12, coords_13), dim=0)
            coords_2 = torch.cat((coords_21, coords_22, coords_23), dim=0)

            if self.N_choice == 0 or 3:
                coords_1 = torch.cat((coords_1, z_1), dim=1)
                coords_2 = torch.cat((coords_2, z_2), dim=1)

            else:
                label1 = torch.zeros(self.numpoints, 1)
                label2 = torch.ones(self.numpoints, 1)

                coords_1 = torch.cat((coords_1, label1, z_1), dim=1)
                coords_2 = torch.cat((coords_2, label2, z_2), dim=1)

        else:
            if not self.counter % self.counter_checkpoint and (self.counter + 1):
                self.num_vio_sample = int(0.2 * self.numpoints)
                self.numpoints = self.numpoints + 55000
                self.num_vio = int(0.2 * self.numpoints)
                print(self.numpoints)

            self.n_sample = self.numpoints - self.num_vio - 2000

            if not self.counter % self.counter_data and (self.counter + 1):
                self.coords_11 = torch.cat((torch.zeros(self.num_vio, 1).uniform_(self.d1_min, self.d1_max),
                                            torch.zeros(self.num_vio, 1).uniform_(-1, 1),
                                            torch.zeros(self.num_vio, 1).uniform_(self.d2_min, self.d2_max),
                                            torch.zeros(self.num_vio, 1).uniform_(-1, 1)), dim=1)
                self.coords_12 = torch.zeros(self.n_sample, self.num_states).uniform_(-1, 1)
                self.coords_13 = torch.cat((torch.zeros(2000, 1).uniform_(self.d1_min, self.d1_max),
                                            torch.zeros(2000, 1).uniform_(-1, 1),
                                            torch.zeros(2000, 1).uniform_(self.d2_min, self.d2_max),
                                            torch.zeros(2000, 1).uniform_(-1, 1)), dim=1)
                self.z_1 = torch.zeros(self.numpoints, 1).uniform_(-1.05e-4, 300)
                self.coords_21 = torch.cat((self.coords_11[:, 2:], self.coords_11[:, :2]), dim=1)
                self.coords_22 = torch.cat((self.coords_12[:, 2:], self.coords_12[:, :2]), dim=1)
                self.coords_23 = torch.cat((self.coords_13[:, 2:], self.coords_13[:, :2]), dim=1)
                self.z_2 = torch.zeros(self.numpoints, 1).uniform_(-1.05e-4, 300)

            coords_1 = torch.cat((self.coords_11, self.coords_12, self.coords_13), dim=0)
            coords_2 = torch.cat((self.coords_21, self.coords_22, self.coords_23), dim=0)

            if self.N_choice == 0 or 3:
                coords_1 = torch.cat((coords_1, self.z_1), dim=1)
                coords_2 = torch.cat((coords_2, self.z_2), dim=1)

            else:
                label1 = torch.zeros(self.numpoints, 1)
                label2 = torch.ones(self.numpoints, 1)

                coords_1 = torch.cat((coords_1, label1, self.z_1), dim=1)
                coords_2 = torch.cat((coords_2, label2, self.z_2), dim=1)

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
        # unnormalize the state for agent 1
        d11 = (coords_1[:, 1:2] + 1) * (105 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d12 = (coords_1[:, 3:4] + 1) * (105 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        # unnormalize the state for agent 1
        d21 = (coords_2[:, 3:4] + 1) * (105 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d22 = (coords_2[:, 1:2] + 1) * (105 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        gT_1 = -self.alpha * ((coords_1[:, 1:2] + 1) * (105 - 15) / 2 + 15) + \
                             ((coords_1[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        zT_1 = coords_1[:, -1:]

        gT_2 = -self.alpha * ((coords_2[:, 1:2] + 1) * (105 - 15) / 2 + 15) + \
                             ((coords_2[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        zT_2 = coords_2[:, -1:]

        if self.N_choice == 0:
            # state constraint for (a,a) setting
            cT_1 = self.epsilon - (torch.abs((d11 - 36.5) + (d12 - 36.5)) + torch.abs((d11 - 36.5) - (d12 - 36.5)))
            cT_2 = self.epsilon - (torch.abs((d21 - 36.5) + (d22 - 36.5)) + torch.abs((d21 - 36.5) - (d22 - 36.5)))

        if self.N_choice == 1:
            # state constraint for (a,na) setting
            cT_1 = self.epsilon - (torch.abs((5/3)*(d11 - 36.5) + (d12 - 35)) + torch.abs((5/3)*(d11 - 36.5) - (d12 - 35)))
            cT_2 = self.epsilon - (torch.abs((5/3)*(d21 - 36.5) + (d22 - 35)) + torch.abs((5/3)*(d21 - 36.5) - (d22 - 35)))

        if self.N_choice == 2:
            # state constraint for (na,a) setting
            cT_1 = self.epsilon - (torch.abs((d11 - 35) + (5/3)*(d12 - 36.5)) + torch.abs((d11 - 35) - (5/3)*(d12 - 36.5)))
            cT_2 = self.epsilon - (torch.abs((d21 - 35) + (5/3)*(d22 - 36.5)) + torch.abs((d21 - 35) - (5/3)*(d22 - 36.5)))

        if self.N_choice == 3:
            # state constraint for (na,na) setting
            cT_1 = self.epsilon - (torch.abs((d11 - 35) + (d12 - 35)) + torch.abs((d11 - 35) - (d12 - 35)))
            cT_2 = self.epsilon - (torch.abs((d21 - 35) + (d22 - 35)) + torch.abs((d21 - 35) - (d22 - 35)))

        boundary_valuesA = torch.cat((cT_1, cT_2), dim=0)
        boundary_valuesB = torch.cat((gT_1 - zT_1, gT_2 - zT_2), dim=0)

        # boundary_values = torch.cat((boundary_values_1, boundary_values_2), dim=0)

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
        self.num_states = 4

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
        coords_2 = torch.cat((coords_1[:, 2:], coords_1[:, :2]), dim=1)

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

        # set up boundary condition: V(T) = alpha*X(T) - (V(T) - V(0))^2
        boundary_values_1 = self.alpha * ((coords_1[:, 1:2] + 1) * (105 - 15) / 2 + 15) - \
                            ((coords_1[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        boundary_values_2 = self.alpha * ((coords_2[:, 1:2] + 1) * (105 - 15) / 2 + 15) - \
                            ((coords_2[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
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
    def __init__(self, Hybrid_use, N_choice, seed=0):

        super().__init__()
        torch.manual_seed(0)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not Hybrid_use:
            if N_choice == 0:
                data_path1 = current_dir + '/validation_scripts/train_data/data_train_a_a_1200.mat'
            if N_choice == 1:
                data_path1 = current_dir + '/validation_scripts/train_data/data_train_a_na_1200.mat'
            if N_choice == 2:
                data_path1 = current_dir + '/validation_scripts/train_data/data_train_na_a_1200.mat'
            if N_choice == 3:
                data_path1 = current_dir + '/validation_scripts/train_data/data_train_na_na_1200.mat'
            train_data1 = scipy.io.loadmat(data_path1)
            self.train_data1 = train_data1
        else:
            if N_choice == 0:
                data_path1 = current_dir + '/validation_scripts/train_data/data_train_a_a_500.mat'
            if N_choice == 1:
                data_path1 = current_dir + '/validation_scripts/train_data/data_train_a_na_500.mat'
            if N_choice == 2:
                data_path1 = current_dir + '/validation_scripts/train_data/data_train_na_a_500.mat'
            if N_choice == 3:
                data_path1 = current_dir + '/validation_scripts/train_data/data_train_na_na_500.mat'
            train_data1 = scipy.io.loadmat(data_path1)
            self.train_data1 = train_data1

        if N_choice == 0:
            data_path2 = current_dir + '/validation_scripts/train_data/data_train_a_a_500_spare.mat'
        if N_choice == 1:
            data_path2 = current_dir + '/validation_scripts/train_data/data_train_a_na_500_spare.mat'
        if N_choice == 2:
            data_path2 = current_dir + '/validation_scripts/train_data/data_train_na_a_500_spare.mat'
        if N_choice == 3:
            data_path2 = current_dir + '/validation_scripts/train_data/data_train_na_na_500_spare.mat'
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
    def __init__(self, N_choice, numpoints, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3, num_src_samples=1000,
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
        if N_choice == 0:
            data_path1 = current_dir + '/validation_scripts/train_data/data_train_a_a_500.mat'
        if N_choice == 1:
            data_path1 = current_dir + '/validation_scripts/train_data/data_train_a_na_500.mat'
        if N_choice == 2:
            data_path1 = current_dir + '/validation_scripts/train_data/data_train_na_a_500.mat'
        if N_choice == 3:
            data_path1 = current_dir + '/validation_scripts/train_data/data_train_na_na_500.mat'
        train_data1 = scipy.io.loadmat(data_path1)
        self.train_data1 = train_data1

        if N_choice == 0:
            data_path2 = current_dir + '/validation_scripts/train_data/data_train_a_a_500_spare.mat'
        if N_choice == 1:
            data_path2 = current_dir + '/validation_scripts/train_data/data_train_a_na_500_spare.mat'
        if N_choice == 2:
            data_path2 = current_dir + '/validation_scripts/train_data/data_train_na_a_500_spare.mat'
        if N_choice == 3:
            data_path2 = current_dir + '/validation_scripts/train_data/data_train_na_na_500_spare.mat'
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

