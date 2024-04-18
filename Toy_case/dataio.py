import torch
from torch.utils.data import Dataset
import scipy.io
import os
import math

class SSL_1D(Dataset):
    def __init__(self, numpoints, seed=0):
        super().__init__()
        self.numpoints = numpoints
        self.num_states = 1
        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        coords = torch.zeros(int(self.numpoints), self.num_states).uniform_(-1, 1)

        # make sure we always have training samples at the boundary
        coords[0] = 1

        # set up boundary condition: V(X=1) = 0
        dirichlet_mask = coords[:self.numpoints] == 1

        boundary_values = torch.zeros((self.numpoints, 1))


        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}

class SSL_1D_vh(Dataset):
    def __init__(self, numpoints, alpha, seed=0):
        super().__init__()
        self.numpoints = numpoints
        self.num_states = 1
        self.alpha = alpha

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        num_d = 150 - math.ceil(self.alpha/(6.25 * 1e-3))  # gradually increase number of samples at discnt. region
        coords_0 = torch.zeros(int(num_d), self.num_states).uniform_(-1*self.alpha, self.alpha)
        coords_1 = torch.zeros(int(self.numpoints-num_d), self.num_states).uniform_(-1, 1)

        coords = torch.cat((coords_1, coords_0))


        # make sure we always have training samples at the boundary
        coords[0:80] = 1

        # set up boundary condition: V(X=1) = 0
        dirichlet_mask = coords[:self.numpoints] == 1

        boundary_values = torch.zeros((self.numpoints, 1))


        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}


class Sup_1D(Dataset):
    def __init__(self, seed=0):
        super().__init__()
        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        coords_1 = torch.linspace(-0.75, -0.25, 1)
        coords_2 = torch.linspace(0.25, 0.75, 1)

        coords = torch.cat((coords_1, coords_2)).reshape(-1, 1)

        gt_values_1 = -1 * torch.ones(1, 1)
        gt_values_2 = torch.zeros(1, 1)
        gt_values = torch.cat((gt_values_1, gt_values_2))

        return {'coords': coords}, {'gt_values': gt_values}

class Hybrid_1D(Dataset):
    def __init__(self, numpoints, seed=0):
        super().__init__()
        self.numpoints = numpoints
        self.num_states = 1

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        coords_h = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        coords_1 = torch.linspace(-0.75, -0.25, 1)
        coords_2 = torch.linspace(0.25, 0.75, 1)
        coords_s = torch.cat((coords_1, coords_2)).reshape(-1, 1)
        coords = torch.cat((coords_h, coords_s)).reshape(-1, 1)

        gt_values_1 = -1 * torch.ones(1, 1)
        gt_values_2 = torch.zeros(1, 1)
        gt_values = torch.cat((gt_values_1, gt_values_2))

        # make sure we always have training samples at the boundary
        coords[0] = 1

        # set up boundary condition: V(X=1) = 0
        dirichlet_mask = coords[:self.numpoints] == 1

        boundary_values = torch.zeros((self.numpoints, 1))

        return {'coords': coords}, {'gt_values': gt_values,
                                    'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}
