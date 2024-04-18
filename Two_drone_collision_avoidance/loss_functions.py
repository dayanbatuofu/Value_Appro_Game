import torch
import diff_operators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        lam11_1 = dvdx_1[:, :1] / ((15.5 - 0) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((15.5 - 0) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((2 - (-1.8)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((4.5 - 0.3) / 2)  # lambda_11
        lam11_5 = dvdx_1[:, 4:5] / ((4.5 - 0.3) / 2)  # lambda_11
        lam11_6 = dvdx_1[:, 5:6] / ((1.8 - (-1.8)) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 6:7] / ((15.5 - 0) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 7:8] / ((15.5 - 0) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 8:9] / ((2 - (-1.8)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 9:10] / ((4.5 - 0.3) / 2)  # lambda_12
        lam12_5 = dvdx_1[:, 10:11] / ((4.5 - 0.3) / 2)  # lambda_12
        lam12_6 = dvdx_1[:, 11:12] / ((1.8 - (-1.8)) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 6:7] / ((15.5 - 0) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 7:8] / ((15.5 - 0) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 8:9] / ((2 - (-1.8)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 9:10] / ((4.5 - 0.3) / 2)  # lambda_21
        lam21_5 = dvdx_2[:, 10:11] / ((4.5 - 0.3) / 2)  # lambda_21
        lam21_6 = dvdx_2[:, 11:12] / ((1.8 - (-1.8)) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((15.5 - 0) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((15.5 - 0) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((2 - (-1.8)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((4.5 - 0.3) / 2)  # lambda_22
        lam22_5 = dvdx_2[:, 4:5] / ((4.5 - 0.3) / 2)  # lambda_22
        lam22_6 = dvdx_2[:, 5:6] / ((1.8 - (-1.8)) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([5.], dtype=torch.float32).to(device)  # road length at y direction
        R2 = torch.tensor([5.], dtype=torch.float32).to(device)  # road length at x direction
        threshold = torch.tensor([0.9], dtype=torch.float32).to(device)  # collision penalty threshold
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio
        gravity = torch.tensor([9.81], dtype=torch.float32).to(device)  # gravity acceleration

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action
        # H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        theta1 = torch.atan(lam11_4 * gravity / 200)
        phi1 = torch.atan(-lam11_5 * gravity / 200)
        thrust1 = lam11_6 / 2 + gravity

        # Agent 2's action
        theta2 = torch.atan(lam22_4 * gravity / 200)
        phi2 = torch.atan(-lam22_5 * gravity / 200)
        thrust2 = lam22_6 / 2 + gravity

        # set up bounds for u1 and u2
        max_acc_theta = torch.tensor([0.05], dtype=torch.float32).to(device)
        min_acc_theta = torch.tensor([-0.05], dtype=torch.float32).to(device)
        max_acc_phi = torch.tensor([0.05], dtype=torch.float32).to(device)
        min_acc_phi = torch.tensor([-0.05], dtype=torch.float32).to(device)
        max_acc_thrust = torch.tensor([11.81], dtype=torch.float32).to(device)
        min_acc_thrust = torch.tensor([7.81], dtype=torch.float32).to(device)

        theta1[torch.where(theta1 > max_acc_theta)] = max_acc_theta
        theta1[torch.where(theta1 < min_acc_theta)] = min_acc_theta
        theta2[torch.where(theta2 > max_acc_theta)] = max_acc_theta
        theta2[torch.where(theta2 < min_acc_theta)] = min_acc_theta

        phi1[torch.where(phi1 > max_acc_phi)] = max_acc_phi
        phi1[torch.where(phi1 < min_acc_phi)] = min_acc_phi
        phi2[torch.where(phi2 > max_acc_phi)] = max_acc_phi
        phi2[torch.where(phi2 < min_acc_phi)] = min_acc_phi

        thrust1[torch.where(thrust1 > max_acc_thrust)] = max_acc_thrust
        thrust1[torch.where(thrust1 < min_acc_thrust)] = min_acc_thrust
        thrust2[torch.where(thrust2 > max_acc_thrust)] = max_acc_thrust
        thrust2[torch.where(thrust2 < min_acc_thrust)] = min_acc_thrust

        # detach and let the action as the number, not trainable variable
        # u1.requires_grad = False
        # u2.requires_grad = False

        # unnormalize the state for agent 1
        dx_11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_11 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (2 - (-1.8)) / 2 + (-1.8)
        vx_11 = (model_output['model_in'][:, :cut_index, 4:5] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_11 = (model_output['model_in'][:, :cut_index, 5:6] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_11 = (model_output['model_in'][:, :cut_index, 6:7] + 1) * (1.8 - (-1.8)) / 2 + (-1.8)

        # unnormalize the state for agent 2
        dx_12 = (model_output['model_in'][:, :cut_index, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_12 = (model_output['model_in'][:, :cut_index, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_12 = (model_output['model_in'][:, :cut_index, 9:10] + 1) * (2 - (-1.8)) / 2 + (-1.8)
        vx_12 = (model_output['model_in'][:, :cut_index, 10:11] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_12 = (model_output['model_in'][:, :cut_index, 11:12] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_12 = (model_output['model_in'][:, :cut_index, 12:13] + 1) * (1.8 - (-1.8)) / 2 + (-1.8)

        # calculate the collision area lower and upper bounds
        dist_diff1 = (-(torch.sqrt(((R1 - dx_12) - dx_11) ** 2 + ((R2 - dy_12) - dy_11) ** 2 + (dz_12 - dz_11) ** 2) -
                        threshold) * 5).squeeze().reshape(-1, 1).to(device)
        sigmoid1 = torch.sigmoid(dist_diff1)
        loss_instant1 = beta * sigmoid1

        # unnormalize the state for agent 1
        dx_21 = (model_output['model_in'][:, cut_index:, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_21 = (model_output['model_in'][:, cut_index:, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_21 = (model_output['model_in'][:, cut_index:, 9:10] + 1) * (2 - (-1.8)) / 2 + (-1.8)
        vx_21 = (model_output['model_in'][:, cut_index:, 10:11] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_21 = (model_output['model_in'][:, cut_index:, 11:12] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_21 = (model_output['model_in'][:, cut_index:, 12:13] + 1) * (1.8 - (-1.8)) / 2 + (-1.8)

        # unnormalize the state for agent 2
        dx_22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_22 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (2 - (-1.8)) / 2 + (-1.8)
        vx_22 = (model_output['model_in'][:, cut_index:, 4:5] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_22 = (model_output['model_in'][:, cut_index:, 5:6] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_22 = (model_output['model_in'][:, cut_index:, 6:7] + 1) * (1.8 - (-1.8)) / 2 + (-1.8)

        # calculate the collision area lower and upper bounds
        dist_diff2 = (-(torch.sqrt(((R1 - dx_22) - dx_21) ** 2 + ((R2 - dy_22) - dy_21) ** 2 + (dz_22 - dz_21) ** 2) -
                        threshold) * 5).squeeze().reshape(-1, 1).to(device)
        sigmoid2 = torch.sigmoid(dist_diff2)
        loss_instant2 = beta * sigmoid2

        # calculate instantaneous loss
        loss_fun_1 = 100 * torch.tan(theta1) ** 2 + 100 * torch.tan(phi1) ** 2 + (thrust1 - gravity) ** 2 + loss_instant1
        loss_fun_2 = 100 * torch.tan(theta2) ** 2 + 100 * torch.tan(phi2) ** 2 + (thrust2 - gravity) ** 2 + loss_instant2

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * vx_11.squeeze() - lam11_2.squeeze() * vy_11.squeeze() - \
                lam11_3.squeeze() * vz_11.squeeze() - lam11_4.squeeze() * torch.tan(theta1).squeeze() * gravity + \
                lam11_5.squeeze() * torch.tan(phi1).squeeze() * gravity - lam11_6.squeeze() * (thrust1 - gravity).squeeze() - \
                lam12_1.squeeze() * vx_12.squeeze() - lam12_2.squeeze() * vy_12.squeeze() - \
                lam12_3.squeeze() * vz_12.squeeze() - lam12_4.squeeze() * torch.tan(theta2).squeeze() * gravity + \
                lam12_5.squeeze() * torch.tan(phi2).squeeze() * gravity - lam12_6.squeeze() * (thrust2 - gravity).squeeze() + \
                loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * vx_21.squeeze() - lam21_2.squeeze() * vy_21.squeeze() - \
                lam21_3.squeeze() * vz_21.squeeze() - lam21_4.squeeze() * torch.tan(theta1).squeeze() * gravity + \
                lam21_5.squeeze() * torch.tan(phi1).squeeze() * gravity - lam21_6.squeeze() * (thrust1 - gravity).squeeze() - \
                lam22_1.squeeze() * vx_22.squeeze() - lam22_2.squeeze() * vy_22.squeeze() - \
                lam22_3.squeeze() * vz_22.squeeze() - lam22_4.squeeze() * torch.tan(theta2).squeeze() * gravity + \
                lam22_5.squeeze() * torch.tan(phi2).squeeze() * gravity - lam22_6.squeeze() * (thrust2 - gravity).squeeze() + \
                loss_fun_2.squeeze()

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

        # A factor of (weight1, weight2) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() / weight1,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / weight2}
    return intersection_hji

def initialize_intersection_HJI_supervised(dataset, Weight, alpha):
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
        lam11_1 = dvdx_1[:, :1] / ((15.5 - 0) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((15.5 - 0) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((2 - (-1.8)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((4.5 - 0.3) / 2)  # lambda_11
        lam11_5 = dvdx_1[:, 4:5] / ((4.5 - 0.3) / 2)  # lambda_11
        lam11_6 = dvdx_1[:, 5:6] / ((1.8 - (-1.8)) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 6:7] / ((15.5 - 0) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 7:8] / ((15.5 - 0) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 8:9] / ((2 - (-1.8)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 9:10] / ((4.5 - 0.3) / 2)  # lambda_12
        lam12_5 = dvdx_1[:, 10:11] / ((4.5 - 0.3) / 2)  # lambda_12
        lam12_6 = dvdx_1[:, 11:12] / ((1.8 - (-1.8)) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 6:7] / ((15.5 - 0) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 7:8] / ((15.5 - 0) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 8:9] / ((2 - (-1.8)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 9:10] / ((4.5 - 0.3) / 2)  # lambda_21
        lam21_5 = dvdx_2[:, 10:11] / ((4.5 - 0.3) / 2)  # lambda_21
        lam21_6 = dvdx_2[:, 11:12] / ((1.8 - (-1.8)) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((15.5 - 0) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((15.5 - 0) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((2 - (-1.8)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((4.5 - 0.3) / 2)  # lambda_22
        lam22_5 = dvdx_2[:, 4:5] / ((4.5 - 0.3) / 2)  # lambda_22
        lam22_6 = dvdx_2[:, 5:6] / ((1.8 - (-1.8)) / 2)  # lambda_22

        # supervised learning for values
        value1_difference = y1 - alpha * groundtruth_values[:, :y1.shape[1]]
        value2_difference = y2 - alpha * groundtruth_values[:, y2.shape[1]:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1, lam11_2, lam11_3, lam11_4, lam11_5, lam11_6,
                                         lam12_1, lam12_2, lam12_3, lam12_4, lam12_5, lam12_6), dim=1)
        costate2_prediction = torch.cat((lam21_1, lam21_2, lam21_3, lam21_4, lam21_5, lam21_6,
                                         lam22_1, lam22_2, lam22_3, lam22_4, lam22_5, lam22_6), dim=1)
        costate1_difference = costate1_prediction - alpha * groundtruth_costates[:, :y1.shape[1]].squeeze()
        costate2_difference = costate2_prediction - alpha * groundtruth_costates[:, y2.shape[1]:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # A factor of (2e5, 100) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / weight1,
                'costates_difference': torch.abs(costates_difference).sum() / weight2}
    return intersection_hji

def initialize_intersection_HJI_hyrid(dataset, Weight, alpha):
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
        lam11_1 = dvdx_1[:, :1] / ((15.5 - 0) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((15.5 - 0) / 2)  # lambda_11
        lam11_3 = dvdx_1[:, 2:3] / ((2 - (-1.8)) / 2)  # lambda_11
        lam11_4 = dvdx_1[:, 3:4] / ((4.5 - 0.3) / 2)  # lambda_11
        lam11_5 = dvdx_1[:, 4:5] / ((4.5 - 0.3) / 2)  # lambda_11
        lam11_6 = dvdx_1[:, 5:6] / ((1.8 - (-1.8)) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 6:7] / ((15.5 - 0) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 7:8] / ((15.5 - 0) / 2)  # lambda_12
        lam12_3 = dvdx_1[:, 8:9] / ((2 - (-1.8)) / 2)  # lambda_12
        lam12_4 = dvdx_1[:, 9:10] / ((4.5 - 0.3) / 2)  # lambda_12
        lam12_5 = dvdx_1[:, 10:11] / ((4.5 - 0.3) / 2)  # lambda_12
        lam12_6 = dvdx_1[:, 11:12] / ((1.8 - (-1.8)) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 6:7] / ((15.5 - 0) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 7:8] / ((15.5 - 0) / 2)  # lambda_21
        lam21_3 = dvdx_2[:, 8:9] / ((2 - (-1.8)) / 2)  # lambda_21
        lam21_4 = dvdx_2[:, 9:10] / ((4.5 - 0.3) / 2)  # lambda_21
        lam21_5 = dvdx_2[:, 10:11] / ((4.5 - 0.3) / 2)  # lambda_21
        lam21_6 = dvdx_2[:, 11:12] / ((1.8 - (-1.8)) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((15.5 - 0) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((15.5 - 0) / 2)  # lambda_22
        lam22_3 = dvdx_2[:, 2:3] / ((2 - (-1.8)) / 2)  # lambda_22
        lam22_4 = dvdx_2[:, 3:4] / ((4.5 - 0.3) / 2)  # lambda_22
        lam22_5 = dvdx_2[:, 4:5] / ((4.5 - 0.3) / 2)  # lambda_22
        lam22_6 = dvdx_2[:, 5:6] / ((1.8 - (-1.8)) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([5.], dtype=torch.float32).to(device)  # road length at y direction
        R2 = torch.tensor([5.], dtype=torch.float32).to(device)  # road length at x direction
        threshold = torch.tensor([0.9], dtype=torch.float32).to(device)  # collision penalty threshold
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio
        gravity = torch.tensor([9.81], dtype=torch.float32).to(device)  # gravity acceleration

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action
        # H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        theta1 = torch.atan(lam11_4 * gravity / (200 * alpha))
        phi1 = torch.atan(-lam11_5 * gravity / (200 * alpha))
        thrust1 = lam11_6 / (2 * alpha) + gravity

        # Agent 2's action
        theta2 = torch.atan(lam22_4 * gravity / (200 * alpha))
        phi2 = torch.atan(-lam22_5 * gravity / (200 * alpha))
        thrust2 = lam22_6 / (2 * alpha) + gravity

        # set up bounds for u1 and u2
        max_acc_theta = torch.tensor([0.05], dtype=torch.float32).to(device)
        min_acc_theta = torch.tensor([-0.05], dtype=torch.float32).to(device)
        max_acc_phi = torch.tensor([0.05], dtype=torch.float32).to(device)
        min_acc_phi = torch.tensor([-0.05], dtype=torch.float32).to(device)
        max_acc_thrust = torch.tensor([11.81], dtype=torch.float32).to(device)
        min_acc_thrust = torch.tensor([7.81], dtype=torch.float32).to(device)

        theta1[torch.where(theta1 > max_acc_theta)] = max_acc_theta
        theta1[torch.where(theta1 < min_acc_theta)] = min_acc_theta
        theta2[torch.where(theta2 > max_acc_theta)] = max_acc_theta
        theta2[torch.where(theta2 < min_acc_theta)] = min_acc_theta

        phi1[torch.where(phi1 > max_acc_phi)] = max_acc_phi
        phi1[torch.where(phi1 < min_acc_phi)] = min_acc_phi
        phi2[torch.where(phi2 > max_acc_phi)] = max_acc_phi
        phi2[torch.where(phi2 < min_acc_phi)] = min_acc_phi

        thrust1[torch.where(thrust1 > max_acc_thrust)] = max_acc_thrust
        thrust1[torch.where(thrust1 < min_acc_thrust)] = min_acc_thrust
        thrust2[torch.where(thrust2 > max_acc_thrust)] = max_acc_thrust
        thrust2[torch.where(thrust2 < min_acc_thrust)] = min_acc_thrust

        # detach and let the action as the number, not trainable variable
        # u1.requires_grad = False
        # u2.requires_grad = False

        # unnormalize the state for agent 1
        dx_11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_11 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (2 - (-1.8)) / 2 + (-1.8)
        vx_11 = (model_output['model_in'][:, :cut_index, 4:5] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_11 = (model_output['model_in'][:, :cut_index, 5:6] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_11 = (model_output['model_in'][:, :cut_index, 6:7] + 1) * (1.8 - (-1.8)) / 2 + (-1.8)

        # unnormalize the state for agent 2
        dx_12 = (model_output['model_in'][:, :cut_index, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_12 = (model_output['model_in'][:, :cut_index, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_12 = (model_output['model_in'][:, :cut_index, 9:10] + 1) * (2 - (-1.8)) / 2 + (-1.8)
        vx_12 = (model_output['model_in'][:, :cut_index, 10:11] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_12 = (model_output['model_in'][:, :cut_index, 11:12] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_12 = (model_output['model_in'][:, :cut_index, 12:13] + 1) * (1.8 - (-1.8)) / 2 + (-1.8)

        # calculate the collision area lower and upper bounds
        dist_diff1 = (-(torch.sqrt(((R1 - dx_12) - dx_11) ** 2 + ((R2 - dy_12) - dy_11) ** 2 + (dz_12 - dz_11) ** 2) -
                        threshold) * 5).squeeze().reshape(-1, 1).to(device)
        sigmoid1 = torch.sigmoid(dist_diff1)
        loss_instant1 = beta * sigmoid1

        # unnormalize the state for agent 1
        dx_21 = (model_output['model_in'][:, cut_index:, 7:8] + 1) * (15.5 - 0) / 2 + 0
        dy_21 = (model_output['model_in'][:, cut_index:, 8:9] + 1) * (15.5 - 0) / 2 + 0
        dz_21 = (model_output['model_in'][:, cut_index:, 9:10] + 1) * (2 - (-1.8)) / 2 + (-1.8)
        vx_21 = (model_output['model_in'][:, cut_index:, 10:11] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_21 = (model_output['model_in'][:, cut_index:, 11:12] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_21 = (model_output['model_in'][:, cut_index:, 12:13] + 1) * (1.8 - (-1.8)) / 2 + (-1.8)

        # unnormalize the state for agent 2
        dx_22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (15.5 - 0) / 2 + 0
        dy_22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (15.5 - 0) / 2 + 0
        dz_22 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (2 - (-1.8)) / 2 + (-1.8)
        vx_22 = (model_output['model_in'][:, cut_index:, 4:5] + 1) * (4.5 - 0.3) / 2 + 0.3
        vy_22 = (model_output['model_in'][:, cut_index:, 5:6] + 1) * (4.5 - 0.3) / 2 + 0.3
        vz_22 = (model_output['model_in'][:, cut_index:, 6:7] + 1) * (1.8 - (-1.8)) / 2 + (-1.8)

        # calculate the collision area lower and upper bounds
        dist_diff2 = (-(torch.sqrt(((R1 - dx_22) - dx_21) ** 2 + ((R2 - dy_22) - dy_21) ** 2 + (dz_22 - dz_21) ** 2) -
                        threshold) * 5).squeeze().reshape(-1, 1).to(device)
        sigmoid2 = torch.sigmoid(dist_diff2)
        loss_instant2 = beta * sigmoid2

        # calculate instantaneous loss
        loss_fun_1 = alpha * (100 * torch.tan(theta1) ** 2 + 100 * torch.tan(phi1) ** 2 + (thrust1 - gravity) ** 2 + loss_instant1)
        loss_fun_2 = alpha * (100 * torch.tan(theta2) ** 2 + 100 * torch.tan(phi2) ** 2 + (thrust2 - gravity) ** 2 + loss_instant2)

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * vx_11.squeeze() - lam11_2.squeeze() * vy_11.squeeze() - \
                lam11_3.squeeze() * vz_11.squeeze() - lam11_4.squeeze() * torch.tan(theta1).squeeze() * gravity + \
                lam11_5.squeeze() * torch.tan(phi1).squeeze() * gravity - lam11_6.squeeze() * (thrust1 - gravity).squeeze() - \
                lam12_1.squeeze() * vx_12.squeeze() - lam12_2.squeeze() * vy_12.squeeze() - \
                lam12_3.squeeze() * vz_12.squeeze() - lam12_4.squeeze() * torch.tan(theta2).squeeze() * gravity + \
                lam12_5.squeeze() * torch.tan(phi2).squeeze() * gravity - lam12_6.squeeze() * (thrust2 - gravity).squeeze() + \
                loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * vx_21.squeeze() - lam21_2.squeeze() * vy_21.squeeze() - \
                lam21_3.squeeze() * vz_21.squeeze() - lam21_4.squeeze() * torch.tan(theta1).squeeze() * gravity + \
                lam21_5.squeeze() * torch.tan(phi1).squeeze() * gravity - lam21_6.squeeze() * (thrust1 - gravity).squeeze() - \
                lam22_1.squeeze() * vx_22.squeeze() - lam22_2.squeeze() * vy_22.squeeze() - \
                lam22_3.squeeze() * vz_22.squeeze() - lam22_4.squeeze() * torch.tan(theta2).squeeze() * gravity + \
                lam22_5.squeeze() * torch.tan(phi2).squeeze() * gravity - lam22_6.squeeze() * (thrust2 - gravity).squeeze() + \
                loss_fun_2.squeeze()

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
                                         lam11_3[:supervised_index, :],
                                         lam11_4[:supervised_index, :],
                                         lam11_5[:supervised_index, :],
                                         lam11_6[:supervised_index, :],
                                         lam12_1[:supervised_index, :],
                                         lam12_2[:supervised_index, :],
                                         lam12_3[:supervised_index, :],
                                         lam12_4[:supervised_index, :],
                                         lam12_5[:supervised_index, :],
                                         lam12_6[:supervised_index, :]), dim=1)
        costate2_prediction = torch.cat((lam21_1[:supervised_index, :],
                                         lam21_2[:supervised_index, :],
                                         lam21_3[:supervised_index, :],
                                         lam21_4[:supervised_index, :],
                                         lam21_5[:supervised_index, :],
                                         lam21_6[:supervised_index, :],
                                         lam22_1[:supervised_index, :],
                                         lam22_2[:supervised_index, :],
                                         lam22_3[:supervised_index, :],
                                         lam22_4[:supervised_index, :],
                                         lam22_5[:supervised_index, :],
                                         lam22_6[:supervised_index, :]), dim=1)
        costate1_difference = costate1_prediction - alpha * groundtruth_costates[:, :supervised_index].squeeze()
        costate2_difference = costate2_prediction - alpha * groundtruth_costates[:, supervised_index:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # boundary condition check
        dirichlet_1 = y1[:, supervised_index:][dirichlet_mask] - source_boundary_values[:, :hji_index][dirichlet_mask]
        dirichlet_2 = y2[:, supervised_index:][dirichlet_mask] - source_boundary_values[:, hji_index:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (weight1, weight2, weight3, weight4) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / weight1,
                'costates_difference': torch.abs(costates_difference).sum() / weight2,
                'dirichlet': torch.norm(torch.abs(dirichlet)) / weight3,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / weight4}
    return intersection_hji