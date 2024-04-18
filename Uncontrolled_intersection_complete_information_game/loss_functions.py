import torch
import diff_operators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_intersection_HJI_EL(dataset, Weight, N_choice):
    def intersection_hji(model_output, gt):
        weight1, weight2 = Weight
        source_boundary_valuesA = gt['source_boundary_valuesA']
        source_boundary_valuesB = gt['source_boundary_valuesB']
        x = model_output['model_in']
        yA = model_output['model_outA']
        yB = model_output['model_outB']
        y = torch.max(yA, yB)
        cut_index = x.shape[1] // 2

        y1 = y[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2 = y[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value

        yA1 = yA[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        yA2 = yA[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        yB1 = yB[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        yB2 = yB[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        dirichlet_mask = gt['dirichlet_mask']

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:4] / ((32 - 15) / 2)  # lambda_12
        lam1_z = dvdx_1[:, -1:]  # lambda1_z

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:4] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22
        lam2_z = dvdx_2[:, -1:]  # lambda2_z

        # calculate the collision area for aggressive-aggressive case
        # collision ratio (a,a):4.5; (na,na):7.5; (a,na):7.5; (na,a):7.5;
        if N_choice == 0:
            epsilon = (torch.tensor([4.5], dtype=torch.float32)).to(device)   # (a,a)
        if N_choice == 1:
            epsilon = (torch.tensor([7.5], dtype=torch.float32)).to(device)   # (a,na)
        if N_choice == 2:
            epsilon = (torch.tensor([7.5], dtype=torch.float32)).to(device)   # (na,a)
        if N_choice == 3:
            epsilon = (torch.tensor([7.5], dtype=torch.float32)).to(device)   # (na,na)

        index_P1_1 = torch.where(lam1_z == -1)[0]  # -1
        index_P2_1 = torch.where(lam2_z == -1)[0]

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

        # H = -(dV/dx)^T * f + (dV/dz)^T * L, optimal action u = 1/2 * B^T * lambda_x / lambda_z
        # H = -(dV/dx)^T * f when dV/dz=0, optimal action u = max H

        # Agent 1's action, be careful about the order of u1>0 and u1<0
        u1 = 1 * lam11_2
        u1[torch.where(u1 > 0)] = 1
        u1[torch.where(u1 < 0)] = -1
        u1[torch.where(u1 == 1)] = min_acc
        u1[torch.where(u1 == -1)] = max_acc
        u1[index_P1_1] = (0.5 * lam11_2[index_P1_1] / lam1_z[index_P1_1])

        # Agent 2's action, be careful about the order of u2>0 and u2<0
        u2 = 1 * lam22_2
        u2[torch.where(u2 > 0)] = 1
        u2[torch.where(u2 < 0)] = -1
        u2[torch.where(u2 == 1)] = min_acc
        u2[torch.where(u2 == -1)] = max_acc
        u2[index_P2_1] = (0.5 * lam22_2[index_P2_1] / lam2_z[index_P2_1])

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        # unnormalize the state for agent 1
        d11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (105 - 15) / 2 + 15
        v11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d12 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (105 - 15) / 2 + 15
        v12 = (model_output['model_in'][:, :cut_index, 4:5] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 1
        d21 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (105 - 15) / 2 + 15
        v21 = (model_output['model_in'][:, cut_index:, 4:5] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (105 - 15) / 2 + 15
        v22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (32 - 15) / 2 + 15

        # calculate instantaneous loss, (a,a): (36.5,36.5);
        if N_choice == 0:
            # calculate instantaneous loss, (a,a): (36.5,36.5);
            ct_1 = (epsilon - (torch.abs((d11 - 36.5) + (d12 - 36.5)) + torch.abs((d11 - 36.5) - (d12 - 36.5)))).squeeze()
            ct_2 = (epsilon - (torch.abs((d21 - 36.5) + (d22 - 36.5)) + torch.abs((d21 - 36.5) - (d22 - 36.5)))).squeeze()
        if N_choice == 1:
            # calculate instantaneous loss, (a,na): (36.5,35);
            ct_1 = (epsilon - (torch.abs((5/3)*(d11 - 36.5) + (d12 - 35)) + torch.abs((5/3)*(d11 - 36.5) - (d12 - 35)))).squeeze()
            ct_2 = (epsilon - (torch.abs((5/3)*(d21 - 36.5) + (d22 - 35)) + torch.abs((5/3)*(d21 - 36.5) - (d22 - 35)))).squeeze()
        if N_choice == 2:
            # calculate instantaneous loss, (a,na): (35,36.5);
            ct_1 = (epsilon - (torch.abs((d11 - 35) + (5/3)*(d12 - 36.5)) + torch.abs((d11 - 35) - (5/3)*(d12 - 36.5)))).squeeze()
            ct_2 = (epsilon - (torch.abs((d21 - 35) + (5/3)*(d22 - 36.5)) + torch.abs((d21 - 35) - (5/3)*(d22 - 36.5)))).squeeze()
        if N_choice == 3:
            # calculate instantaneous loss, (na,na): (35,35);
            ct_1 = (epsilon - (torch.abs((d11 - 35) + (d12 - 35)) + torch.abs((d11 - 35) - (d12 - 35)))).squeeze()
            ct_2 = (epsilon - (torch.abs((d21 - 35) + (d22 - 35)) + torch.abs((d21 - 35) - (d22 - 35)))).squeeze()

        # calculate hamiltonian, -H = (dV/dx)^T * f - (dV/dz)^T * L
        ham_1 = lam11_1.squeeze() * v11.squeeze() + lam11_2.squeeze() * u1.squeeze() + \
                lam12_1.squeeze() * v12.squeeze() + lam12_2.squeeze() * u2.squeeze() - lam1_z.squeeze() * (u1**2).squeeze()
        ham_2 = lam21_1.squeeze() * v21.squeeze() + lam21_2.squeeze() * u1.squeeze() + \
                lam22_1.squeeze() * v22.squeeze() + lam22_2.squeeze() * u2.squeeze() - lam2_z.squeeze() * (u2**2).squeeze()

        # boundary condition check
        dirichletA_1 = yA1[dirichlet_mask] - source_boundary_valuesA[:, :yA1.shape[1]][dirichlet_mask]
        dirichletA_2 = yA2[dirichlet_mask] - source_boundary_valuesA[:, yA2.shape[1]:][dirichlet_mask]
        dirichletA = torch.cat((dirichletA_1, dirichletA_2), dim=0)

        dirichletB_1 = yB1[dirichlet_mask] - source_boundary_valuesB[:, :yB1.shape[1]][dirichlet_mask]
        dirichletB_2 = yB2[dirichlet_mask] - source_boundary_valuesB[:, yB2.shape[1]:][dirichlet_mask]
        dirichletB = torch.cat((dirichletB_1, dirichletB_2), dim=0)

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check, HJI = dV/dt - H = -dV/dt + (dV/dx)^T * f - (dV/dz)^T * L because we invert the time
        if torch.all(dirichlet_mask):
            diff_constraint_hom_1 = torch.Tensor([0])
            diff_constraint_hom_2 = torch.Tensor([0])
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)
        else:
            num_boundary = dirichletB.shape[0] // 2
            diff_constraint_hom_1 = torch.max(ct_1 - y1.squeeze(), -dvdt_1 + ham_1)[:cut_index - num_boundary]
            diff_constraint_hom_2 = torch.max(ct_2 - y2.squeeze(), -dvdt_2 + ham_2)[:cut_index - num_boundary]
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        weight_ratio = torch.abs(diff_constraint_hom).sum() * weight1 / torch.abs(dirichletB).sum()

        weight_ratio = weight_ratio.detach()

        if weight_ratio == 0:
            hjpde_weight = 1
        else:
            hjpde_weight = float(weight_ratio)

        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichletA': torch.abs(dirichletA).sum()/weight1,
                'dirichletB': torch.abs(dirichletB).sum()/weight1,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / hjpde_weight,
                'weight': hjpde_weight}

    return intersection_hji

def initialize_intersection_HJI_pinn(dataset, Weight, Theta):
    def intersection_hji(model_output, gt):
        weight1, weight2 = Weight
        theta1, theta2 = Theta
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        dirichlet_mask = gt['dirichlet_mask']

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 1
        R2 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 2
        W1 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        W2 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        L1 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        L2 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        theta1 = torch.tensor([theta1], dtype=torch.float32).to(device)  # behavior for agent 1
        theta2 = torch.tensor([theta2], dtype=torch.float32).to(device)  # behavior for agent 2
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action
        # H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        u1 = 0.5 * lam11_2 * 10

        # Agent 2's action
        u2 = 0.5 * lam22_2 * 10

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        # unnormalize the state for agent 1
        d11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (105 - 15) / 2 + 15
        v11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d12 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (105 - 15) / 2 + 15
        v12 = (model_output['model_in'][:, :cut_index, 4:] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x11_in = ((d11 - R1 / 2 + theta1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out = (-(d11 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in = ((d12 - R2 / 2 + theta2 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out = (-(d12 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11 = torch.sigmoid(x11_in) * torch.sigmoid(x11_out)
        sigmoid12 = torch.sigmoid(x12_in) * torch.sigmoid(x12_out)
        loss_instant1 = beta * sigmoid11 * sigmoid12

        # unnormalize the state for agent 1
        d21 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (105 - 15) / 2 + 15
        v21 = (model_output['model_in'][:, cut_index:, 4:] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (105 - 15) / 2 + 15
        v22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds

        x21_in = ((d21 - R1 / 2 + theta1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out = (-(d21 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in = ((d22 - R2 / 2 + theta2 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out = (-(d22 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21 = torch.sigmoid(x21_in) * torch.sigmoid(x21_out)
        sigmoid22 = torch.sigmoid(x22_in) * torch.sigmoid(x22_out)
        loss_instant2 = beta * sigmoid21 * sigmoid22

        # calculate instantaneous loss
        loss_fun_1 = 0.1 * (u1 ** 2 + loss_instant1)
        loss_fun_2 = 0.1 * (u2 ** 2 + loss_instant2)

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v11.squeeze() - lam11_2.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v12.squeeze() - lam12_2.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v21.squeeze() - lam21_2.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v22.squeeze() - lam22_2.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        if torch.all(dirichlet_mask):
            diff_constraint_hom_1 = torch.Tensor([0])
            diff_constraint_hom_2 = torch.Tensor([0])
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)
        else:
            diff_constraint_hom_1 = dvdt_1 + ham_1
            diff_constraint_hom_2 = dvdt_2 + ham_2
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # boundary condition check
        dirichlet_1 = y1[dirichlet_mask] - 0.1 * source_boundary_values[:, :y1.shape[1]][dirichlet_mask]
        dirichlet_2 = y2[dirichlet_mask] - 0.1 * source_boundary_values[:, y2.shape[1]:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() / weight1,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / weight2}
    return intersection_hji

def initialize_intersection_HJI_supervised(dataset, Weight, Theta, alpha):
    def intersection_hji(model_output, gt):
        weight1, weight2 = Weight
        theta1, theta2 = Theta
        groundtruth_values = gt['groundtruth_values']
        groundtruth_costates = gt['groundtruth_costates']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]   # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]   # (meta_batch_size, num_points, 1); agent 2's value

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # supervised learning for values
        value1_difference = y1 - alpha * groundtruth_values[:, :y1.shape[1]]
        value2_difference = y2 - alpha * groundtruth_values[:, y2.shape[1]:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1, lam11_2, lam12_1, lam12_2), dim=1)
        costate2_prediction = torch.cat((lam21_1, lam21_2, lam22_1, lam22_2), dim=1)
        costate1_difference = costate1_prediction - alpha * groundtruth_costates[:, :y1.shape[1]].squeeze()
        costate2_difference = costate2_prediction - alpha * groundtruth_costates[:, y2.shape[1]:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # A factor of (weight1, weight2) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / weight1,
                'costates_difference': torch.abs(costates_difference).sum() / weight2}
    return intersection_hji


def initialize_intersection_HJI_hyrid(dataset, Weight, Theta, alpha):
    def intersection_hji(model_output, gt):
        weight1, weight2, weight3, weight4 = Weight
        theta1, theta2 = Theta
        groundtruth_values = gt['groundtruth_values']
        groundtruth_costates = gt['groundtruth_costates']
        source_boundary_values = gt['source_boundary_values']
        dirichlet_mask = gt['dirichlet_mask']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2
        supervised_index = groundtruth_values.shape[1] // 2
        hji_index = source_boundary_values.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]   # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]   # (meta_batch_size, num_points, 1); agent 2's value

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 1
        R2 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 2
        W1 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        W2 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        L1 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        L2 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        theta1 = torch.tensor([theta1], dtype=torch.float32).to(device)  # behavior for agent 1
        theta2 = torch.tensor([theta2], dtype=torch.float32).to(device)  # behavior for agent 2
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision loss weight

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action
        # H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        u1 = 0.5 * lam11_2 * (1/alpha)

        # Agent 2's action
        u2 = 0.5 * lam22_2 * (1/alpha)

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        # unnormalize the state for agent 1
        d11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (105 - 15) / 2 + 15
        v11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d12 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (105 - 15) / 2 + 15
        v12 = (model_output['model_in'][:, :cut_index, 4:] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x11_in = ((d11 - R1 / 2 + theta1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out = (-(d11 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in = ((d12 - R2 / 2 + theta2 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out = (-(d12 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11 = torch.sigmoid(x11_in) * torch.sigmoid(x11_out)
        sigmoid12 = torch.sigmoid(x12_in) * torch.sigmoid(x12_out)
        loss_instant1 = beta * sigmoid11 * sigmoid12

        # unnormalize the state for agent 1
        d21 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (105 - 15) / 2 + 15
        v21 = (model_output['model_in'][:, cut_index:, 4:] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (105 - 15) / 2 + 15
        v22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x21_in = ((d21 - R1 / 2 + theta1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out = (-(d21 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in = ((d22 - R2 / 2 + theta2 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out = (-(d22 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21 = torch.sigmoid(x21_in) * torch.sigmoid(x21_out)
        sigmoid22 = torch.sigmoid(x22_in) * torch.sigmoid(x22_out)
        loss_instant2 = beta * sigmoid21 * sigmoid22

        # calculate instantaneous loss
        loss_fun_1 = alpha * (u1 ** 2 + loss_instant1)
        loss_fun_2 = alpha * (u2 ** 2 + loss_instant2)

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v11.squeeze() - lam11_2.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v12.squeeze() - lam12_2.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v21.squeeze() - lam21_2.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v22.squeeze() - lam22_2.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        diff_constraint_hom_1 = dvdt_1 + ham_1
        diff_constraint_hom_2 = dvdt_2 + ham_2
        diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # supervised learning for values
        value1_difference = y1[:, :supervised_index] - alpha * groundtruth_values[:, :supervised_index]
        value2_difference = y2[:, :supervised_index] - alpha * groundtruth_values[:, supervised_index:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1[:supervised_index, :],
                                         lam11_2[:supervised_index, :],
                                         lam12_1[:supervised_index, :],
                                         lam12_2[:supervised_index, :]), dim=1)
        costate2_prediction = torch.cat((lam21_1[:supervised_index, :],
                                         lam21_2[:supervised_index, :],
                                         lam22_1[:supervised_index, :],
                                         lam22_2[:supervised_index, :]), dim=1)
        costate1_difference = costate1_prediction - alpha *groundtruth_costates[:, :supervised_index].squeeze()
        costate2_difference = costate2_prediction - alpha *groundtruth_costates[:, supervised_index:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # boundary condition check
        dirichlet_1 = y1[:, supervised_index:][dirichlet_mask] - alpha *source_boundary_values[:, :hji_index][dirichlet_mask]
        dirichlet_2 = y2[:, supervised_index:][dirichlet_mask] - alpha *source_boundary_values[:, hji_index:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (weight1, weight2, weight3, weight4) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / weight1,
                'costates_difference': torch.abs(costates_difference).sum() / weight2,
                'dirichlet': torch.abs(dirichlet).sum() / weight3,  # L2 norm: torch.norm(torch.abs(dirichlet)) / weight3
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / weight4}
    return intersection_hji

def initialize_intersection_HJI_valuehardening(dataset, gamma, Weight, Theta):
    def intersection_hji(model_output, gt):
        weight1, weight2 = Weight
        theta1, theta2 = Theta
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        dirichlet_mask = gt['dirichlet_mask']

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 1
        R2 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 2
        W1 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        W2 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        L1 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        L2 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        theta1 = torch.tensor([theta1], dtype=torch.float32).to(device)  # behavior for agent 1
        theta2 = torch.tensor([theta2], dtype=torch.float32).to(device)  # behavior for agent 2
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action
        # H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        u1 = 0.5 * lam11_2 * 10

        # Agent 2's action
        u2 = 0.5 * lam22_2 * 10

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        # unnormalize the state for agent 1
        d11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (105 - 15) / 2 + 15
        v11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d12 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (105 - 15) / 2 + 15
        v12 = (model_output['model_in'][:, :cut_index, 4:] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x11_in = ((d11 - R1 / 2 + theta1 * W2 / 2) * gamma).squeeze().reshape(-1, 1).to(device)
        x11_out = (-(d11 - R1 / 2 - W2 / 2 - L1) * gamma).squeeze().reshape(-1, 1).to(device)
        x12_in = ((d12 - R2 / 2 + theta2 * W1 / 2) * gamma).squeeze().reshape(-1, 1).to(device)
        x12_out = (-(d12 - R2 / 2 - W1 / 2 - L2) * gamma).squeeze().reshape(-1, 1).to(device)

        sigmoid11 = torch.sigmoid(x11_in) * torch.sigmoid(x11_out)
        sigmoid12 = torch.sigmoid(x12_in) * torch.sigmoid(x12_out)
        loss_instant1 = beta * sigmoid11 * sigmoid12

        # unnormalize the state for agent 1
        d21 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (105 - 15) / 2 + 15
        v21 = (model_output['model_in'][:, cut_index:, 4:] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (105 - 15) / 2 + 15
        v22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x21_in = ((d21 - R1 / 2 + theta1 * W2 / 2) * gamma).squeeze().reshape(-1, 1).to(device)
        x21_out = (-(d21 - R1 / 2 - W2 / 2 - L1) * gamma).squeeze().reshape(-1, 1).to(device)
        x22_in = ((d22 - R2 / 2 + theta2 * W1 / 2) * gamma).squeeze().reshape(-1, 1).to(device)
        x22_out = (-(d22 - R2 / 2 - W1 / 2 - L2) * gamma).squeeze().reshape(-1, 1).to(device)

        sigmoid21 = torch.sigmoid(x21_in) * torch.sigmoid(x21_out)
        sigmoid22 = torch.sigmoid(x22_in) * torch.sigmoid(x22_out)
        loss_instant2 =  beta * sigmoid21 * sigmoid22

        # calculate instantaneous loss
        loss_fun_1 = 0.1 * (u1 ** 2 + loss_instant1)
        loss_fun_2 = 0.1 * (u2 ** 2 + loss_instant2)

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v11.squeeze() - lam11_2.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v12.squeeze() - lam12_2.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v21.squeeze() - lam21_2.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v22.squeeze() - lam22_2.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        if torch.all(dirichlet_mask):
            diff_constraint_hom_1 = torch.Tensor([0])
            diff_constraint_hom_2 = torch.Tensor([0])
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)
        else:
            diff_constraint_hom_1 = dvdt_1 + ham_1
            diff_constraint_hom_2 = dvdt_2 + ham_2
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # boundary condition check
        dirichlet_1 = y1[dirichlet_mask] - 0.1 * source_boundary_values[:, :y1.shape[1]][dirichlet_mask]
        dirichlet_2 = y2[dirichlet_mask] - 0.1 * source_boundary_values[:, y2.shape[1]:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (weight1, weight2) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() / weight1,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / weight2}

    return intersection_hji
