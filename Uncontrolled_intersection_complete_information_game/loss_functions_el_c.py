import torch
import diff_operators
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_intersection_HJI_supervised(dataset, Weight, Costate):
    def intersection_hji(model_output, gt):
        weight1, weight2 = Weight
        groundtruth_values = gt['groundtruth_values']
        groundtruth_constraints = gt['groundtruth_constraints']
        groundtruth_costates = gt['groundtruth_costates']
        x = model_output['model_in']
        yA = model_output['model_outA']
        yB = model_output['model_outB']
        y = torch.max(yA, yB)
        sl_index = groundtruth_values.shape[1] // 2
        cut_index = x.shape[1] // 2

        y1A = yA[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2A = yA[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        y1B = yB[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2B = yB[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        value1B = y1B + x[:, :cut_index, -1:]
        value2B = y2B + x[:, cut_index:, -1:]

        # calculate the partial gradient of V w.r.t. time and state
        # jac, _ = diff_operators.jacobian(yB, x)
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

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:4] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # supervised learning for values
        value1_difference = value1B - groundtruth_values[:, :sl_index]
        value2_difference = value2B - groundtruth_values[:, sl_index:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1, lam11_2, lam12_1, lam12_2), dim=1)
        costate2_prediction = torch.cat((lam21_1, lam21_2, lam22_1, lam22_2), dim=1)
        costate1_difference = costate1_prediction - groundtruth_costates[:, :sl_index].squeeze()
        costate2_difference = costate2_prediction - groundtruth_costates[:, sl_index:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)
        if Costate == True:
            pass
        else:
            costates_difference = torch.Tensor([0])

        constraint1_difference = y1A - groundtruth_constraints[:, :sl_index]
        constraint2_difference = y2A - groundtruth_constraints[:, sl_index:]
        constraint_difference = torch.cat((constraint1_difference, constraint2_difference), dim=0)

        dirichletA = torch.Tensor([0])
        dirichletB = torch.Tensor([0])
        hjpde_weight = 1

        # A factor of (weight1, weight2) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / 2,
                'constraint_difference': torch.abs(constraint_difference).sum() / weight2,
                'costates_difference': torch.abs(costates_difference).sum() / weight2,
                'dirichletA': torch.abs(dirichletA).sum() / weight1,
                'dirichletB': torch.abs(dirichletB).sum() / weight1,
                'weight': hjpde_weight}
    return intersection_hji


def initialize_intersection_HJI_EL(dataset, Weight, Costate):
    def intersection_hji(model_output, gt):
        weight1, weight2 = Weight
        groundtruth_values = gt['groundtruth_values']
        groundtruth_constraints = gt['groundtruth_constraints']
        groundtruth_costates = gt['groundtruth_costates']
        source_boundary_valuesA = gt['source_boundary_valuesA']
        source_boundary_valuesB = gt['source_boundary_valuesB']
        dirichlet_mask = gt['dirichlet_mask']
        x = model_output['model_in']
        yA = model_output['model_outA']
        yB = model_output['model_outB']
        y = torch.max(yA, yB)
        cut_index = x.shape[1] // 2
        sl_index = groundtruth_values.shape[1] // 2
        el_index = source_boundary_valuesA.shape[1] // 2

        y1 = y[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2 = y[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value

        y1A = yA[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2A = yA[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        y1B = yB[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2B = yB[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        value1B = y1B + x[:, :cut_index, -1:]
        value2B = y2B + x[:, cut_index:, -1:]

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
        epsilon = (torch.tensor([4.5], dtype=torch.float32)).to(device)  # collision ratio (a,a):4.5; (na,na),(a,na),(na,a):7.5

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

        # # calculate instantaneous loss, (a,a): (36.5,36.5); (na,na): (35,35); (a,na); (na,a)
        ct_1 = (epsilon - (torch.abs((d11 - 36.5) + (d12 - 36.5)) + torch.abs((d11 - 36.5) - (d12 - 36.5)))).squeeze()  
        ct_2 = (epsilon - (torch.abs((d21 - 36.5) + (d22 - 36.5)) + torch.abs((d21 - 36.5) - (d22 - 36.5)))).squeeze()

        # calculate hamiltonian, -H = (dV/dx)^T * f - (dV/dz)^T * L
        ham_1 = lam11_1.squeeze() * v11.squeeze() + lam11_2.squeeze() * u1.squeeze() + \
                lam12_1.squeeze() * v12.squeeze() + lam12_2.squeeze() * u2.squeeze() - lam1_z.squeeze() * (u1**2).squeeze()
        ham_2 = lam21_1.squeeze() * v21.squeeze() + lam21_2.squeeze() * u1.squeeze() + \
                lam22_1.squeeze() * v22.squeeze() + lam22_2.squeeze() * u2.squeeze() - lam2_z.squeeze() * (u2**2).squeeze()

        # boundary condition check
        dirichletA_1 = y1A[:, sl_index:][dirichlet_mask] - source_boundary_valuesA[:, :el_index][dirichlet_mask]
        dirichletA_2 = y2A[:, sl_index:][dirichlet_mask] - source_boundary_valuesA[:, el_index:][dirichlet_mask]
        dirichletA = torch.cat((dirichletA_1, dirichletA_2), dim=0)

        dirichletB_1 = y1B[:, sl_index:][dirichlet_mask] - source_boundary_valuesB[:, :el_index][dirichlet_mask]
        dirichletB_2 = y2B[:, sl_index:][dirichlet_mask] - source_boundary_valuesB[:, el_index:][dirichlet_mask]
        dirichletB = torch.cat((dirichletB_1, dirichletB_2), dim=0)

        # supervised learning for values
        value1_difference = value1B[:, :sl_index] - groundtruth_values[:, :sl_index]
        value2_difference = value2B[:, :sl_index] - groundtruth_values[:, sl_index:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        costate1_prediction = torch.cat((lam11_1, lam11_2, lam12_1, lam12_2), dim=1)
        costate2_prediction = torch.cat((lam21_1, lam21_2, lam22_1, lam22_2), dim=1)
        costate1_difference = costate1_prediction[:sl_index, :] - groundtruth_costates[:, :sl_index].squeeze()
        costate2_difference = costate2_prediction[:sl_index, :] - groundtruth_costates[:, sl_index:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)
        if Costate == True:
            pass
        else:
            costates_difference = torch.Tensor([0])

        constraint1_difference = y1A[:, :sl_index] - groundtruth_constraints[:, :sl_index]
        constraint2_difference = y2A[:, :sl_index] - groundtruth_constraints[:, sl_index:]
        constraint_difference = torch.cat((constraint1_difference, constraint2_difference), dim=0)

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check, HJI = dV/dt - H = -dV/dt + (dV/dx)^T * f - (dV/dz)^T * L because we invert the time
        # num_boundary = dirichletB.shape[0] // 2
        diff_constraint_hom_1 = torch.max(ct_1 - y1.squeeze(), -dvdt_1 + ham_1)
        diff_constraint_hom_2 = torch.max(ct_2 - y2.squeeze(), -dvdt_2 + ham_2)
        diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)
        weight_ratio = torch.abs(diff_constraint_hom).sum() * weight1 / torch.abs(dirichletB).sum()
        weight_ratio = weight_ratio.detach()
        hjpde_weight = float(weight_ratio)

        # A factor of (2e5, 100) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum()/2,
                'costates_difference': torch.abs(costates_difference).sum()/weight2,
                'constraint_difference': torch.abs(constraint_difference).sum()/weight2,
                'dirichletA': torch.abs(dirichletA).sum()/weight1,
                'dirichletB': torch.abs(dirichletB).sum()/weight1,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()/hjpde_weight,
                'weight': hjpde_weight}

    return intersection_hji