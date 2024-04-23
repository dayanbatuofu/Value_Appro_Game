import torch
import diff_operators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_intersection_HJI_EL(dataset, Weight):
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
        lam11_1 = dvdx_1[:, :1] / ((90 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((38 - 32) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((0.18 - (-0.15)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((25 - 18) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 4:5] / ((90 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 5:6] / ((38 - 32) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 6:7] / ((0.18 - (-0.15)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 7:8] / ((25 - 18) / 2)  # lambda_12
        lam1_z = dvdx_1[:, 8:]  # lambda1_z

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 4:5] / ((90 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 5:6] / ((38 - 32) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 6:7] / ((0.18 - (-0.15)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 7:8] / ((25 - 18) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((90 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((38 - 32) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((0.18 - (-0.15)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((25 - 18) / 2)  # lambda_22
        lam2_z = dvdx_2[:, 8:]  # lambda2_z

        # calculate the collision area for aggressive-aggressive case
        R = torch.tensor([70.], dtype=torch.float32).to(device)
        epsilon = (torch.tensor([1.5], dtype=torch.float32)).to(device)  # collision ratio

        index_P1_1 = torch.where(lam1_z == -1)[0]  # -1
        index_P2_1 = torch.where(lam2_z == -1)[0]

        # set up bounds for u1 and u2
        max_acc_u = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc_u = torch.tensor([-5.], dtype=torch.float32).to(device)
        max_acc_w = torch.tensor([1.], dtype=torch.float32).to(device)
        min_acc_w = torch.tensor([-1.], dtype=torch.float32).to(device)

        # H = -(dV/dx)^T * f + (dV/dz)^T * L, optimal action u = 1/2 * B^T * lambda_x / lambda_z
        # H = -(dV/dx)^T * f when dV/dz=0, optimal action u = max H

        # Agent 1's action, be careful about the order of u1>0 and u1<0
        u1 = 1 * lam11_4
        u1[torch.where(u1 > 0)] = 1
        u1[torch.where(u1 < 0)] = -1
        u1[torch.where(u1 == 1)] = min_acc_u
        u1[torch.where(u1 == -1)] = max_acc_u
        u1[index_P1_1] = (0.5 * lam11_4[index_P1_1] / lam1_z[index_P1_1])

        w1 = 1 * lam11_3
        w1[torch.where(w1 > 0)] = 1
        w1[torch.where(w1 < 0)] = -1
        w1[torch.where(w1 == 1)] = min_acc_w
        w1[torch.where(w1 == -1)] = max_acc_w
        w1[index_P1_1] = (lam11_3[index_P1_1] / lam1_z[index_P1_1]) / 200

        # Agent 2's action, be careful about the order of u2>0 and u2<0
        u2 = 1 * lam22_4
        u2[torch.where(u2 > 0)] = 1
        u2[torch.where(u2 < 0)] = -1
        u2[torch.where(u2 == 1)] = min_acc_u
        u2[torch.where(u2 == -1)] = max_acc_u
        u2[index_P2_1] = (0.5 * lam22_4[index_P2_1] / lam2_z[index_P2_1])

        w2 = 1 * lam22_3
        w2[torch.where(w2 > 0)] = 1
        w2[torch.where(w2 < 0)] = -1
        w2[torch.where(w2 == 1)] = min_acc_w
        w2[torch.where(w2 == -1)] = max_acc_w
        w2[index_P2_1] = (lam22_3[index_P2_1] / lam2_z[index_P2_1]) / 200

        u1[torch.where(u1 > max_acc_u)] = max_acc_u
        u1[torch.where(u1 < min_acc_u)] = min_acc_u
        u2[torch.where(u2 > max_acc_u)] = max_acc_u
        u2[torch.where(u2 < min_acc_u)] = min_acc_u

        w1[torch.where(w1 > max_acc_w)] = max_acc_w
        w1[torch.where(w1 < min_acc_w)] = min_acc_w
        w2[torch.where(w2 > max_acc_w)] = max_acc_w
        w2[torch.where(w2 < min_acc_w)] = min_acc_w

        # unnormalize the state for agent 1
        dx_11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (90 - 15) / 2 + 15
        dy_11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (38 - 32) / 2 + 32
        theta_11 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_11 = (model_output['model_in'][:, :cut_index, 4:5] + 1) * (25 - 18) / 2 + 18

        # unnormalize the state for agent 2
        dx_12 = (model_output['model_in'][:, :cut_index, 5:6] + 1) * (90 - 15) / 2 + 15
        dy_12 = (model_output['model_in'][:, :cut_index, 6:7] + 1) * (38 - 32) / 2 + 32
        theta_12 = (model_output['model_in'][:, :cut_index, 7:8] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_12 = (model_output['model_in'][:, :cut_index, 8:9] + 1) * (25 - 18) / 2 + 18

        # unnormalize the state for agent 1
        dx_21 = (model_output['model_in'][:, cut_index:, 5:6] + 1) * (90 - 15) / 2 + 15
        dy_21 = (model_output['model_in'][:, cut_index:, 6:7] + 1) * (38 - 32) / 2 + 32
        theta_21 = (model_output['model_in'][:, cut_index:, 7:8] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_21 = (model_output['model_in'][:, cut_index:, 8:9] + 1) * (25 - 18) / 2 + 18

        # unnormalize the state for agent 2
        dx_22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (90 - 15) / 2 + 15
        dy_22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (38 - 32) / 2 + 32
        theta_22 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_22 = (model_output['model_in'][:, cut_index:, 4:5] + 1) * (25 - 18) / 2 + 18

        # # calculate instantaneous loss
        ct_1 = (epsilon - torch.sqrt(((R - dx_12) - dx_11) ** 2 + (dy_12 - dy_11) ** 2)).squeeze()
        ct_2 = (epsilon - torch.sqrt(((R - dx_22) - dx_21) ** 2 + (dy_22 - dy_21) ** 2)).squeeze()

        # calculate hamiltonian, -H = (dV/dx)^T * f - (dV/dz)^T * L
        ham_1 = lam11_1.squeeze() * v_11.squeeze() * torch.cos(theta_11.squeeze()) + \
                lam11_2.squeeze() * v_11.squeeze() * torch.sin(theta_11.squeeze()) + \
                lam11_3.squeeze() * w1.squeeze() + lam11_4.squeeze() * u1.squeeze() + \
                lam12_1.squeeze() * v_12.squeeze() * torch.cos(theta_12.squeeze()) + \
                lam12_2.squeeze() * v_12.squeeze() * torch.sin(theta_12.squeeze()) + \
                lam12_3.squeeze() * w2.squeeze() + lam12_4.squeeze() * u2.squeeze() - lam1_z.squeeze() * (u1**2).squeeze() - \
                lam1_z.squeeze() * (100 * w1 ** 2).squeeze()
        ham_2 = lam21_1.squeeze() * v_21.squeeze() * torch.cos(theta_21.squeeze()) + \
                lam21_2.squeeze() * v_21.squeeze() * torch.sin(theta_21.squeeze()) + \
                lam21_3.squeeze() * w1.squeeze() + lam21_4.squeeze() * u1.squeeze() + \
                lam22_1.squeeze() * v_22.squeeze() * torch.cos(theta_22.squeeze()) + \
                lam22_2.squeeze() * v_22.squeeze() * torch.sin(theta_22.squeeze()) + \
                lam22_3.squeeze() * w2.squeeze() + lam22_4.squeeze() * u2.squeeze() - lam2_z.squeeze() * (u2**2).squeeze() - \
                lam2_z.squeeze() * (100 * w2 ** 2).squeeze()

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
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()/hjpde_weight,
                'weight': hjpde_weight}

    return intersection_hji

def initialize_intersection_HJI_pinn(dataset, Weight):
    def intersection_hji(model_output, gt):
        weight1, weight2 = Weight
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
        lam11_1 = dvdx_1[:, :1] / ((90 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((38 - 32) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((0.18 - (-0.15)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((25 - 18) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 4:5] / ((90 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 5:6] / ((38 - 32) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 6:7] / ((0.18 - (-0.15)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 7:] / ((25 - 18) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 4:5] / ((90 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 5:6] / ((38 - 32) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 6:7] / ((0.18 - (-0.15)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 7:] / ((25 - 18) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((90 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((38 - 32) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((0.18 - (-0.15)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((25 - 18) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R = torch.tensor([70.], dtype=torch.float32).to(device)  # road length
        threshold = torch.tensor([1.5], dtype=torch.float32).to(device)  # collision penalty threshold
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action
        # H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        u1 = 0.5 * lam11_4
        w1 = lam11_3 / 200

        # Agent 2's action
        u2 = 0.5 * lam22_4
        w2 = lam22_3 / 200

        # set up bounds for u1 and u2
        max_acc_u = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc_u = torch.tensor([-5.], dtype=torch.float32).to(device)
        max_acc_w = torch.tensor([1.], dtype=torch.float32).to(device)
        min_acc_w = torch.tensor([-1.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc_u)] = max_acc_u
        u1[torch.where(u1 < min_acc_u)] = min_acc_u
        u2[torch.where(u2 > max_acc_u)] = max_acc_u
        u2[torch.where(u2 < min_acc_u)] = min_acc_u

        w1[torch.where(w1 > max_acc_w)] = max_acc_w
        w1[torch.where(w1 < min_acc_w)] = min_acc_w
        w2[torch.where(w2 > max_acc_w)] = max_acc_w
        w2[torch.where(w2 < min_acc_w)] = min_acc_w

        # unnormalize the state for agent 1
        dx_11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (90 - 15) / 2 + 15
        dy_11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (38 - 32) / 2 + 32
        theta_11 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_11 = (model_output['model_in'][:, :cut_index, 4:5] + 1) * (25 - 18) / 2 + 18

        # unnormalize the state for agent 2
        dx_12 = (model_output['model_in'][:, :cut_index, 5:6] + 1) * (90 - 15) / 2 + 15
        dy_12 = (model_output['model_in'][:, :cut_index, 6:7] + 1) * (38 - 32) / 2 + 32
        theta_12 = (model_output['model_in'][:, :cut_index, 7:8] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_12 = (model_output['model_in'][:, :cut_index, 8:] + 1) * (25 - 18) / 2 + 18

        # calculate the collision area lower and upper bounds
        dist_diff1 = (-(torch.sqrt(((R - dx_12) - dx_11) ** 2 + (dy_12 - dy_11) ** 2) - threshold) * 5).squeeze().reshape(-1, 1).to(device)
        sigmoid1 = torch.sigmoid(dist_diff1)
        loss_instant1 = beta * sigmoid1

        # unnormalize the state for agent 1
        dx_21 = (model_output['model_in'][:, cut_index:, 5:6] + 1) * (90 - 15) / 2 + 15
        dy_21 = (model_output['model_in'][:, cut_index:, 6:7] + 1) * (38 - 32) / 2 + 32
        theta_21 = (model_output['model_in'][:, cut_index:, 7:8] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_21 = (model_output['model_in'][:, cut_index:, 8:] + 1) * (25 - 18) / 2 + 18

        # unnormalize the state for agent 2
        dx_22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (90 - 15) / 2 + 15
        dy_22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (38 - 32) / 2 + 32
        theta_22 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_22 = (model_output['model_in'][:, cut_index:, 4:5] + 1) * (25 - 18) / 2 + 18

        # calculate the collision area lower and upper bounds
        dist_diff2 = (-(torch.sqrt(((R - dx_22) - dx_21) ** 2 + (dy_22 - dy_21) ** 2) - threshold) * 5).squeeze().reshape(-1, 1).to(device)
        sigmoid2 = torch.sigmoid(dist_diff2)
        loss_instant2 = beta * sigmoid2

        # calculate instantaneous loss
        loss_fun_1 = 100 * w1 ** 2 + u1 ** 2 + loss_instant1
        loss_fun_2 = 100 * w2 ** 2 + u2 ** 2 + loss_instant2

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v_11.squeeze() * torch.cos(theta_11.squeeze()) - \
                lam11_2.squeeze() * v_11.squeeze() * torch.sin(theta_11.squeeze()) - \
                lam11_3.squeeze() * w1.squeeze() - lam11_4.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v_12.squeeze() * torch.cos(theta_12.squeeze()) - \
                lam12_2.squeeze() * v_12.squeeze() * torch.sin(theta_12.squeeze()) - \
                lam12_3.squeeze() * w2.squeeze() - lam12_4.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v_21.squeeze() * torch.cos(theta_21.squeeze()) - \
                lam21_2.squeeze() * v_21.squeeze() * torch.sin(theta_21.squeeze()) - \
                lam21_3.squeeze() * w1.squeeze() - lam21_4.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v_22.squeeze() * torch.cos(theta_22.squeeze()) - \
                lam22_2.squeeze() * v_22.squeeze() * torch.sin(theta_22.squeeze()) - \
                lam22_3.squeeze() * w2.squeeze() - lam21_4.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

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
        dirichlet_1 = y1[dirichlet_mask] - 1 * source_boundary_values[:, :y1.shape[1]][dirichlet_mask]
        dirichlet_2 = y2[dirichlet_mask] - 1 * source_boundary_values[:, y2.shape[1]:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        weight1 = 15
        weight2 = 1000
        # A factor of (weight1, weight2) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() / weight1,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / weight2}
    return intersection_hji

def initialize_intersection_HJI_supervised(dataset, Weight):
    def intersection_hji(model_output, gt):
        weight1, weight2 = Weight
        groundtruth_values = gt['groundtruth_values']
        groundtruth_costates = gt['groundtruth_costates']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]   # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]   # (meta_batch_size, num_points, 1); agent 2's value
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((90 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((38 - 32) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((0.18 - (-0.15)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((25 - 18) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 4:5] / ((90 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 5:6] / ((38 - 32) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 6:7] / ((0.18 - (-0.15)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 7:] / ((25 - 18) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 4:5] / ((90 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 5:6] / ((38 - 32) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 6:7] / ((0.18 - (-0.15)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 7:] / ((25 - 18) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((90 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((38 - 32) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((0.18 - (-0.15)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((25 - 18) / 2)  # lambda_22

        # supervised learning for values
        value1_difference = y1 - groundtruth_values[:, :y1.shape[1]]
        value2_difference = y2 - groundtruth_values[:, y2.shape[1]:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1, lam11_2, lam11_3, lam11_4, lam12_1, lam12_2, lam12_3, lam12_4), dim=1)
        costate2_prediction = torch.cat((lam21_1, lam21_2, lam21_3, lam21_4, lam22_1, lam22_2, lam22_3, lam22_4), dim=1)
        costate1_difference = costate1_prediction - groundtruth_costates[:, :y1.shape[1]].squeeze()
        costate2_difference = costate2_prediction - groundtruth_costates[:, y2.shape[1]:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # A factor of (2e5, 100) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / weight1,
                'costates_difference': torch.abs(costates_difference).sum() / weight2}
    return intersection_hji

def initialize_intersection_HJI_hyrid(dataset, Weight):
    def intersection_hji(model_output, gt):
        weight1, weight2, weight3, weight4 = Weight
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
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((90 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((38 - 32) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((0.18 - (-0.15)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((25 - 18) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 4:5] / ((90 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 5:6] / ((38 - 32) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 6:7] / ((0.18 - (-0.15)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 7:] / ((25 - 18) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 4:5] / ((90 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 5:6] / ((38 - 32) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 6:7] / ((0.18 - (-0.15)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 7:] / ((25 - 18) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((90 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((38 - 32) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((0.18 - (-0.15)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((25 - 18) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R = torch.tensor([70.], dtype=torch.float32).to(device)  # road length
        threshold = torch.tensor([1.5], dtype=torch.float32).to(device)  # collision penalty threshold
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action
        # H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        u1 = 0.5 * lam11_4
        w1 = lam11_3 / 200

        # Agent 2's action
        u2 = 0.5 * lam22_4
        w2 = lam22_3 / 200

        # set up bounds for u1 and u2
        max_acc_u = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc_u = torch.tensor([-5.], dtype=torch.float32).to(device)
        max_acc_w = torch.tensor([1.], dtype=torch.float32).to(device)
        min_acc_w = torch.tensor([-1.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc_u)] = max_acc_u
        u1[torch.where(u1 < min_acc_u)] = min_acc_u
        u2[torch.where(u2 > max_acc_u)] = max_acc_u
        u2[torch.where(u2 < min_acc_u)] = min_acc_u

        w1[torch.where(w1 > max_acc_w)] = max_acc_w
        w1[torch.where(w1 < min_acc_w)] = min_acc_w
        w2[torch.where(w2 > max_acc_w)] = max_acc_w
        w2[torch.where(w2 < min_acc_w)] = min_acc_w

        # unnormalize the state for agent 1
        dx_11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (90 - 15) / 2 + 15
        dy_11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (38 - 32) / 2 + 32
        theta_11 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_11 = (model_output['model_in'][:, :cut_index, 4:5] + 1) * (25 - 18) / 2 + 18

        # unnormalize the state for agent 2
        dx_12 = (model_output['model_in'][:, :cut_index, 5:6] + 1) * (90 - 15) / 2 + 15
        dy_12 = (model_output['model_in'][:, :cut_index, 6:7] + 1) * (38 - 32) / 2 + 32
        theta_12 = (model_output['model_in'][:, :cut_index, 7:8] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_12 = (model_output['model_in'][:, :cut_index, 8:] + 1) * (25 - 18) / 2 + 18

        # calculate the collision area lower and upper bounds
        dist_diff1 = (-(torch.sqrt(((R - dx_12) - dx_11) ** 2 + (dy_12 - dy_11) ** 2) - threshold) * 5).squeeze().reshape(-1, 1).to(device)
        sigmoid1 = torch.sigmoid(dist_diff1)
        loss_instant1 = beta * sigmoid1

        # unnormalize the state for agent 1
        dx_21 = (model_output['model_in'][:, cut_index:, 5:6] + 1) * (90 - 15) / 2 + 15
        dy_21 = (model_output['model_in'][:, cut_index:, 6:7] + 1) * (38 - 32) / 2 + 32
        theta_21 = (model_output['model_in'][:, cut_index:, 7:8] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_21 = (model_output['model_in'][:, cut_index:, 8:] + 1) * (25 - 18) / 2 + 18

        # unnormalize the state for agent 2
        dx_22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (90 - 15) / 2 + 15
        dy_22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (38 - 32) / 2 + 32
        theta_22 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_22 = (model_output['model_in'][:, cut_index:, 4:5] + 1) * (25 - 18) / 2 + 18

        # calculate the collision area lower and upper bounds
        dist_diff2 = (-(torch.sqrt(((R - dx_22) - dx_21) ** 2 + (dy_22 - dy_21) ** 2) - threshold) * 5).squeeze().reshape(-1, 1).to(device)
        sigmoid2 = torch.sigmoid(dist_diff2)
        loss_instant2 = beta * sigmoid2

        # calculate instantaneous loss
        loss_fun_1 = 100 * w1 ** 2 + u1 ** 2 + loss_instant1
        loss_fun_2 = 100 * w2 ** 2 + u2 ** 2 + loss_instant2

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v_11.squeeze() * torch.cos(theta_11.squeeze()) - \
                lam11_2.squeeze() * v_11.squeeze() * torch.sin(theta_11.squeeze()) - \
                lam11_3.squeeze() * w1.squeeze() - lam11_4.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v_12.squeeze() * torch.cos(theta_12.squeeze()) - \
                lam12_2.squeeze() * v_12.squeeze() * torch.sin(theta_12.squeeze()) - \
                lam12_3.squeeze() * w2.squeeze() - lam12_4.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v_21.squeeze() * torch.cos(theta_21.squeeze()) - \
                lam21_2.squeeze() * v_21.squeeze() * torch.sin(theta_21.squeeze()) - \
                lam21_3.squeeze() * w1.squeeze() - lam21_4.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v_22.squeeze() * torch.cos(theta_22.squeeze()) - \
                lam22_2.squeeze() * v_22.squeeze() * torch.sin(theta_22.squeeze()) - \
                lam22_3.squeeze() * w2.squeeze() - lam22_4.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        diff_constraint_hom_1 = dvdt_1 + ham_1
        diff_constraint_hom_2 = dvdt_2 + ham_2
        diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # supervised learning for values
        value1_difference = y1[:, :supervised_index] - groundtruth_values[:, :supervised_index]
        value2_difference = y2[:, :supervised_index] - groundtruth_values[:, supervised_index:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1[:supervised_index, :],
                                         lam11_2[:supervised_index, :],
                                         lam11_3[:supervised_index, :],
                                         lam11_4[:supervised_index, :],
                                         lam12_1[:supervised_index, :],
                                         lam12_2[:supervised_index, :],
                                         lam12_3[:supervised_index, :],
                                         lam12_4[:supervised_index, :]), dim=1)
        costate2_prediction = torch.cat((lam21_1[:supervised_index, :],
                                         lam21_2[:supervised_index, :],
                                         lam21_3[:supervised_index, :],
                                         lam21_4[:supervised_index, :],
                                         lam22_1[:supervised_index, :],
                                         lam22_2[:supervised_index, :],
                                         lam22_3[:supervised_index, :],
                                         lam22_4[:supervised_index, :]), dim=1)
        costate1_difference = costate1_prediction - groundtruth_costates[:, :supervised_index].squeeze()
        costate2_difference = costate2_prediction - groundtruth_costates[:, supervised_index:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # boundary condition check
        dirichlet_1 = y1[:, supervised_index:][dirichlet_mask] - source_boundary_values[:, :hji_index][dirichlet_mask]
        dirichlet_2 = y2[:, supervised_index:][dirichlet_mask] - source_boundary_values[:, hji_index:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (weight1, weight2, weight3, weight4) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / weight1,
                'costates_difference': torch.abs(costates_difference).sum() / weight2,
                'dirichlet': torch.abs(dirichlet).sum() / weight3,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / weight4}
    return intersection_hji

def initialize_intersection_HJI_valuehardening(dataset, gamma, Weight, alpha):
    def intersection_hji(model_output, gt):
        weight1, weight2 = Weight
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((90 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((38 - 32) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((0.18 - (-0.15)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((25 - 18) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 4:5] / ((90 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 5:6] / ((38 - 32) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 6:7] / ((0.18 - (-0.15)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 7:] / ((25 - 18) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 4:5] / ((90 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 5:6] / ((38 - 32) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 6:7] / ((0.18 - (-0.15)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 7:] / ((25 - 18) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((90 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((38 - 32) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((0.18 - (-0.15)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((25 - 18) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R = torch.tensor([70.], dtype=torch.float32).to(device)  # road length
        threshold = torch.tensor([1.5], dtype=torch.float32).to(device)  # collision penalty threshold
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action
        # H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        u1 = 0.5 * lam11_4 * (1/alpha)
        w1 = lam11_3 / (200 * alpha)

        # Agent 2's action
        u2 = 0.5 * lam22_4 * (1/alpha)
        w2 = lam22_3 / (200 * alpha)

        # set up bounds for u1 and u2
        max_acc_u = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc_u = torch.tensor([-5.], dtype=torch.float32).to(device)
        max_acc_w = torch.tensor([1.], dtype=torch.float32).to(device)
        min_acc_w = torch.tensor([-1.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc_u)] = max_acc_u
        u1[torch.where(u1 < min_acc_u)] = min_acc_u
        u2[torch.where(u2 > max_acc_u)] = max_acc_u
        u2[torch.where(u2 < min_acc_u)] = min_acc_u

        w1[torch.where(w1 > max_acc_w)] = max_acc_w
        w1[torch.where(w1 < min_acc_w)] = min_acc_w
        w2[torch.where(w2 > max_acc_w)] = max_acc_w
        w2[torch.where(w2 < min_acc_w)] = min_acc_w

        # unnormalize the state for agent 1
        dx_11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (90 - 15) / 2 + 15
        dy_11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (38 - 32) / 2 + 32
        theta_11 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_11 = (model_output['model_in'][:, :cut_index, 4:5] + 1) * (25 - 18) / 2 + 18

        # unnormalize the state for agent 2
        dx_12 = (model_output['model_in'][:, :cut_index, 5:6] + 1) * (90 - 15) / 2 + 15
        dy_12 = (model_output['model_in'][:, :cut_index, 6:7] + 1) * (38 - 32) / 2 + 32
        theta_12 = (model_output['model_in'][:, :cut_index, 7:8] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_12 = (model_output['model_in'][:, :cut_index, 8:] + 1) * (25 - 18) / 2 + 18

        # calculate the collision area lower and upper bounds
        dist_diff1 = (-(torch.sqrt(((R - dx_12) - dx_11) ** 2 + (dy_12 - dy_11) ** 2) - threshold) * gamma).squeeze().reshape(-1, 1).to(device)
        sigmoid1 = torch.sigmoid(dist_diff1)
        loss_instant1 = beta * sigmoid1

        # unnormalize the state for agent 1
        dx_21 = (model_output['model_in'][:, cut_index:, 5:6] + 1) * (90 - 15) / 2 + 15
        dy_21 = (model_output['model_in'][:, cut_index:, 6:7] + 1) * (38 - 32) / 2 + 32
        theta_21 = (model_output['model_in'][:, cut_index:, 7:8] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_21 = (model_output['model_in'][:, cut_index:, 8:] + 1) * (25 - 18) / 2 + 18

        # unnormalize the state for agent 2
        dx_22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (90 - 15) / 2 + 15
        dy_22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (38 - 32) / 2 + 32
        theta_22 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (0.18 - (-0.15)) / 2 + (-0.15)
        v_22 = (model_output['model_in'][:, cut_index:, 4:5] + 1) * (25 - 18) / 2 + 18

        # calculate the collision area lower and upper bounds
        dist_diff2 = (-(torch.sqrt(((R - dx_22) - dx_21) ** 2 + (dy_22 - dy_21) ** 2) - threshold) * gamma).squeeze().reshape(-1, 1).to(device)
        sigmoid2 = torch.sigmoid(dist_diff2)
        loss_instant2 = beta * sigmoid2

        # calculate instantaneous loss
        loss_fun_1 = alpha * (100 * w1 ** 2 + u1 ** 2 + loss_instant1)
        loss_fun_2 = alpha * (100 * w2 ** 2 + u2 ** 2 + loss_instant2)

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v_11.squeeze() * torch.cos(theta_11.squeeze()) - \
                lam11_2.squeeze() * v_11.squeeze() * torch.sin(theta_11.squeeze()) - \
                lam11_3.squeeze() * w1.squeeze() - lam11_4.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v_12.squeeze() * torch.cos(theta_12.squeeze()) - \
                lam12_2.squeeze() * v_12.squeeze() * torch.sin(theta_12.squeeze()) - \
                lam12_3.squeeze() * w2.squeeze() - lam12_4.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v_21.squeeze() * torch.cos(theta_21.squeeze()) - \
                lam21_2.squeeze() * v_21.squeeze() * torch.sin(theta_21.squeeze()) - \
                lam21_3.squeeze() * w1.squeeze() - lam21_4.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v_22.squeeze() * torch.cos(theta_22.squeeze()) - \
                lam22_2.squeeze() * v_22.squeeze() * torch.sin(theta_22.squeeze()) - \
                lam22_3.squeeze() * w2.squeeze() - lam21_4.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

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
        dirichlet_1 = y1[dirichlet_mask] - alpha * source_boundary_values[:, :y1.shape[1]][dirichlet_mask]
        dirichlet_2 = y2[dirichlet_mask] - alpha * source_boundary_values[:, y2.shape[1]:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (weight1, weight2) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() / weight1,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / weight2}
    return intersection_hji