# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules_adaptive, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optimal_z_generation(X, t, model):
    def dichotomy1(a, b, threshold, X, t, model):
        dx_1 = np.array([[2.0 * (X[0] - 15) / (90 - 15) - 1]])
        dy_1 = np.array([[2.0 * (X[1] - 32) / (38 - 32) - 1]])
        theta_1 = np.array([[2.0 * (X[2] - (-0.15)) / (0.18 - (-0.15)) - 1]])
        v_1 = np.array([[2.0 * (X[3] - 18) / (25 - 18) - 1]])
        dx_2 = np.array([[2.0 * (X[4] - 15) / (90 - 15) - 1]])
        dy_2 = np.array([[2.0 * (X[5] - 32) / (38 - 32) - 1]])
        theta_2 = np.array([[2.0 * (X[6] - (-0.15)) / (0.18 - (-0.15)) - 1]])
        v_2 = np.array([[2.0 * (X[7] - 18) / (25 - 18) - 1]])

        z1 = np.array([[a]])
        X1 = np.vstack((dx_1, dy_1, theta_1, v_1, dx_2, dy_2, theta_2, v_2, z1))
        X1 = torch.tensor(X1, dtype=torch.float32, requires_grad=True).T
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        coords_1 = torch.cat((t, X1), dim=1)
        coords = coords_1.unsqueeze(0)
        model_in = {'coords': coords.to(device)}
        model_output = model(model_in)
        yA = model_output['model_outA']
        yB = model_output['model_outB']
        y1 = torch.max(yA, yB)

        if yA > 0:
            z_star = 50  # 60
        elif yB < 0:
            z_star = z1
        else:
            z_star = z1 + y1.detach().cpu().numpy().squeeze(0)

        return z_star

    def dichotomy2(a, b, threshold, X, t, model):
        dx_1 = np.array([[2.0 * (X[0] - 15) / (90 - 15) - 1]])
        dy_1 = np.array([[2.0 * (X[1] - 32) / (38 - 32) - 1]])
        theta_1 = np.array([[2.0 * (X[2] - (-0.15)) / (0.18 - (-0.15)) - 1]])
        v_1 = np.array([[2.0 * (X[3] - 18) / (25 - 18) - 1]])
        dx_2 = np.array([[2.0 * (X[4] - 15) / (90 - 15) - 1]])
        dy_2 = np.array([[2.0 * (X[5] - 32) / (38 - 32) - 1]])
        theta_2 = np.array([[2.0 * (X[6] - (-0.15)) / (0.18 - (-0.15)) - 1]])
        v_2 = np.array([[2.0 * (X[7] - 18) / (25 - 18) - 1]])

        z2 = np.array([[a]])
        X2 = np.vstack((dx_2, dy_2, theta_2, v_2, dx_1, dy_1, theta_1, v_1, z2))
        X2 = torch.tensor(X2, dtype=torch.float32, requires_grad=True).T
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        coords_2 = torch.cat((t, X2), dim=1)
        coords = coords_2.unsqueeze(0)
        model_in = {'coords': coords.to(device)}
        model_output = model(model_in)
        yA = model_output['model_outA']
        yB = model_output['model_outB']
        y2 = torch.max(yA, yB)

        if yA > 0:
            z_star = 50  # 60
        elif yB < 0:
            z_star = z2
        else:
            z_star = z2 + y2.detach().cpu().numpy().squeeze(0)

        return z_star

    z_min = -9e-5  # -1.05e-4, -9e-5
    z_max = 10
    threshold = 0

    z1_0 = dichotomy1(z_min, z_max, threshold, X, t, model)
    z2_0 = dichotomy2(z_min, z_max, threshold, X, t, model)

    return z1_0, z2_0

def value_action(X, t, model, N_choice):
    # normalize the state for agent 1, agent 2
    dx_1 = 2.0 * (X[0, :] - 15) / (90 - 15) - 1.
    dy_1 = 2.0 * (X[1, :] - 32) / (38 - 32) - 1.
    theta_1 = 2.0 * (X[2, :] - (-0.15)) / (0.18 - (-0.15)) - 1.
    v_1 = 2.0 * (X[3, :] - 18) / (25 - 18) - 1.
    z1 = X[4, :]
    dx_2 = 2.0 * (X[5, :] - 15) / (90 - 15) - 1.
    dy_2 = 2.0 * (X[6, :] - 32) / (38 - 32) - 1.
    theta_2 = 2.0 * (X[7, :] - (-0.15)) / (0.18 - (-0.15)) - 1.
    v_2 = 2.0 * (X[8, :] - 18) / (25 - 18) - 1.
    z2 = X[9, :]

    X1 = np.vstack((dx_1, dy_1, theta_1, v_1, dx_2, dy_2, theta_2, v_2, z1))
    X2 = np.vstack((dx_2, dy_2, theta_2, v_2, dx_1, dy_1, theta_1, v_1, z2))

    X1 = torch.tensor(X1, dtype=torch.float32, requires_grad=True).T
    X2 = torch.tensor(X2, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    coords_1 = torch.cat((t, X1), dim=1)
    coords_2 = torch.cat((t, X2), dim=1)
    coords = torch.cat((coords_1, coords_2), dim=0).unsqueeze(0)
    model_in = {'coords': coords.to(device)}
    model_output = model(model_in)

    x = model_output['model_in']
    yA = model_output['model_outA']
    yB = model_output['model_outB']
    y = torch.max(yA, yB)
    cut_index = x.shape[1] // 2
    y1 = y[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
    y2 = y[:, cut_index:]  # agent 2's value

    yA1 = yA[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
    yA2 = yA[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
    yB1 = yB[:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
    yB2 = yB[:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value

    jac, _ = diff_operators.jacobian(y, x)
    dv_1 = jac[:, :cut_index, :]
    dv_2 = jac[:, cut_index:, :]

    # agent 1: partial gradient of V w.r.t. time and state
    dvdt_1 = dv_1[..., 0, 0].squeeze()
    dvdx_1 = dv_1[..., 0, 1:].squeeze().reshape(1, dv_1.shape[-1] - 1)

    # unnormalize the costate for agent 1, consider V = exp(u)
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
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1)

    # unnormalize the costate for agent 2, consider V = exp(u)
    lam21_1 = dvdx_2[:, 4:5] / ((90 - 15) / 2)  # lambda_21
    lam21_2 = dvdx_2[:, 5:6] / ((38 - 32) / 2)  # lambda_21
    lam21_3 = dvdx_2[:, 6:7] / ((0.18 - (-0.15)) / 2)  # lambda_21
    lam21_4 = dvdx_2[:, 7:8] / ((25 - 18) / 2)  # lambda_21
    lam22_1 = dvdx_2[:, :1] / ((90 - 15) / 2)  # lambda_22
    lam22_2 = dvdx_2[:, 1:2] / ((38 - 32) / 2)  # lambda_22
    lam22_3 = dvdx_2[:, 2:3] / ((0.18 - (-0.15)) / 2)  # lambda_22
    lam22_4 = dvdx_2[:, 3:4] / ((25 - 18) / 2)  # lambda_22
    lam2_z = dvdx_2[:, 8:]  # lambda2_z

    # set up bounds for u1 and u2
    max_acc_u = torch.tensor([10.], dtype=torch.float32).to(device)
    min_acc_u = torch.tensor([-5.], dtype=torch.float32).to(device)
    max_acc_w = torch.tensor([1.], dtype=torch.float32).to(device)
    min_acc_w = torch.tensor([-1.], dtype=torch.float32).to(device)

    index_P1_1 = torch.where(lam1_z == -1)[0]   # -1
    index_P2_1 = torch.where(lam2_z == -1)[0]

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

    R = torch.tensor([70.], dtype=torch.float32).to(device)
    epsilon = (torch.tensor([1.5], dtype=torch.float32)).to(device)  # collision ratio

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

    # # calculate the collision area lower and upper bounds
    ct_1 = (epsilon - torch.sqrt(((R - dx_12) - dx_11) ** 2 + (dy_12 - dy_11) ** 2)).squeeze()  # 35
    ct_2 = (epsilon - torch.sqrt(((R - dx_22) - dx_21) ** 2 + (dy_22 - dy_21) ** 2)).squeeze()  # 35

    # calculate hamiltonian, -H = (dV/dx)^T * f - (dV/dz)^T * L
    ham_1 = -lam11_1.squeeze() * v_11.squeeze() * torch.cos(theta_11.squeeze()) - \
            lam11_2.squeeze() * v_11.squeeze() * torch.sin(theta_11.squeeze()) - \
            lam11_3.squeeze() * w1.squeeze() - lam11_4.squeeze() * u1.squeeze() - \
            lam12_1.squeeze() * v_12.squeeze() * torch.cos(theta_12.squeeze()) - \
            lam12_2.squeeze() * v_12.squeeze() * torch.sin(theta_12.squeeze()) - \
            lam12_3.squeeze() * w2.squeeze() - lam12_4.squeeze() * u2.squeeze() - lam1_z.squeeze() * (u1**2).squeeze() - \
            lam1_z.squeeze() * (100 * w1 ** 2).squeeze()
    ham_2 = -lam21_1.squeeze() * v_21.squeeze() * torch.cos(theta_21.squeeze()) - \
            lam21_2.squeeze() * v_21.squeeze() * torch.sin(theta_21.squeeze()) - \
            lam21_3.squeeze() * w1.squeeze() - lam21_4.squeeze() * u1.squeeze() - \
            lam22_1.squeeze() * v_22.squeeze() * torch.cos(theta_22.squeeze()) - \
            lam22_2.squeeze() * v_22.squeeze() * torch.sin(theta_22.squeeze()) - \
            lam22_3.squeeze() * w2.squeeze() - lam22_4.squeeze() * u2.squeeze() - lam2_z.squeeze() * (u2**2).squeeze() - \
            lam2_z.squeeze() * (100 * w2 ** 2).squeeze()

    diff_constraint_hom_1 = torch.max(ct_1 - y1.squeeze(), -dvdt_1 + ham_1)
    diff_constraint_hom_2 = torch.max(ct_2 - y2.squeeze(), -dvdt_2 + ham_2)

    return u1, u2, w1, w2, y1, y2, ct_1-y1.squeeze(), ct_2-y2.squeeze(), ham_1, ham_2, \
           -dvdt_1+ham_1, -dvdt_2+ham_2, diff_constraint_hom_1, diff_constraint_hom_2, yA1, yB1, yA2, yB2

def dynamic(X_nn, dt, action):
    u1, u2, w1, w2 = action
    theta1 = X_nn[2, :] + w1 * dt
    theta2 = X_nn[7, :] + w2 * dt
    v1 = X_nn[3, :] + u1 * dt
    v2 = X_nn[8, :] + u2 * dt
    dx1 = X_nn[0, :] + v1 * np.cos(theta1) * dt
    dy1 = X_nn[1, :] + v1 * np.sin(theta1) * dt
    dx2 = X_nn[5, :] + v2 * np.cos(theta2) * dt
    dy2 = X_nn[6, :] + v2 * np.sin(theta2) * dt

    return dx1, dy1, theta1, v1, dx2, dy2, theta2, v2

def discrete_data(data, dt, N_choice, N):
    dx1 = data['dx1']
    dy1 = data['dy1']
    theta1 = data['theta1']
    dx2 = data['dx2']
    dy2 = data['dy2']
    theta2 = data['theta2']
    v1 = data['v1']
    v2 = data['v2']
    u1 = data['u1']
    u2 = data['u2']
    w1 = data['w1']
    w2 = data['w2']
    z1 = data['z1']
    z2 = data['z2']
    yV1 = data['V1']
    yV2 = data['V2']
    yA1 = data['yA1']
    yA2 = data['yA2']
    yB1 = data['yB1']
    yB2 = data['yB2']
    ct_1 = data['ct_1']
    ct_2 = data['ct_2']
    ham_1 = data['ham_1']
    ham_2 = data['ham_2']
    hji_1 = data['hji_1']
    hji_2 = data['hji_2']
    diff_hji1 = data['diff_hji1']
    diff_hji2 = data['diff_hji2']
    time_horizon = N

    R = 70
    alpha = 1e-06
    beta = 10000
    threshold = 1.5
    t_step = dt

    U1 = torch.tensor(u1, requires_grad=True, dtype=torch.float32)
    U2 = torch.tensor(u2, requires_grad=True, dtype=torch.float32)
    W1 = torch.tensor(w1, requires_grad=True, dtype=torch.float32)
    W2 = torch.tensor(w2, requires_grad=True, dtype=torch.float32)

    V1 = np.zeros((len(U1[:, 0]), len(U1[0])))
    Loss1 = np.zeros((len(U1[:, 0]), len(U1[0])))
    Loss1_tmp = np.zeros((len(U1[:, 0]), len(U1[0])))
    V2 = np.zeros((len(U2[:, 0]), len(U2[0])))
    Loss2 = np.zeros((len(U2[:, 0]), len(U2[0])))
    Loss2_tmp = np.zeros((len(U2[:, 0]), len(U2[0])))

    for i in range(len(U1[:, 0])):
        for j in range(time_horizon):
            x1 = torch.tensor(dx1[i][j], requires_grad=True, dtype=torch.float32)
            y1 = torch.tensor(dy1[i][j], requires_grad=True, dtype=torch.float32)
            x2 = torch.tensor(dx2[i][j], requires_grad=True, dtype=torch.float32)
            y2 = torch.tensor(dy2[i][j], requires_grad=True, dtype=torch.float32)

            dist_diff = (-(torch.sqrt(((R - x2) - x1) ** 2 + (y2 - y1) ** 2) - threshold) * 5)

            Loss1_tmp[i][j] = (100 * W1[i][j] ** 2 + U1[i][j] ** 2 + beta * torch.sigmoid(dist_diff)) * t_step
            Loss2_tmp[i][j] = (100 * W2[i][j] ** 2 + U2[i][j] ** 2 + beta * torch.sigmoid(dist_diff)) * t_step

    U1 = U1.detach().cpu().numpy()
    U2 = U2.detach().cpu().numpy()
    W1 = W1.detach().cpu().numpy()
    W2 = W2.detach().cpu().numpy()

    for i in range(len(U1[:, 0])):
        for j in range(time_horizon):
            Loss1[i][j] = np.sum(Loss1_tmp[i][j:])
            Loss2[i][j] = np.sum(Loss2_tmp[i][j:])

    for i in range(len(U1[:, 0])):
        for j in range(time_horizon):
            V1[i][j] = alpha * dx1[i][-1] - (v1[i][-1] - 18) ** 2 - (dy1[i][-1] - 35) ** 2 - Loss1[i][j]
            V2[i][j] = alpha * dx2[i][-1] - (v2[i][-1] - 18) ** 2 - (dy2[i][-1] - 35) ** 2 - Loss2[i][j]

    data = {'t': data['t'],
            'X': np.vstack((dx1.reshape(1, -1),
                            dy1.reshape(1, -1),
                            theta1.reshape(1, -1),
                            v1.reshape(1, -1),
                            dx2.reshape(1, -1),
                            dy2.reshape(1, -1),
                            theta2.reshape(1, -1),
                            v2.reshape(1, -1))),
            'V': np.vstack((V1.reshape(1, -1),
                            V2.reshape(1, -1))),
            'U': np.vstack((U1.reshape(1, -1),
                            U2.reshape(1, -1))),
            'Omega': np.vstack((W1.reshape(1, -1),
                                W2.reshape(1, -1))),
            'z': np.vstack((z1.reshape(1, -1),
                            z2.reshape(1, -1))),
            'y': np.vstack((yV1.reshape(1, -1),
                            yV2.reshape(1, -1))),
            'yA': np.vstack((yA1.reshape(1, -1),
                             yA2.reshape(1, -1))),
            'yB': np.vstack((yB1.reshape(1, -1),
                             yB2.reshape(1, -1))),
            'ct': np.vstack((ct_1.reshape(1, -1),
                             ct_2.reshape(1, -1))),
            'ham': np.vstack((ham_1.reshape(1, -1),
                              ham_2.reshape(1, -1))),
            'hji': np.vstack((hji_1.reshape(1, -1),
                              hji_2.reshape(1, -1))),
            'diff_hji': np.vstack((diff_hji1.reshape(1, -1),
                                   diff_hji2.reshape(1, -1)))}

    return data

if __name__ == '__main__':

    logging_root = './logs'
    N_neurons = 64

    policy = ['a_a', 'a_na', 'na_a', 'na_na']
    N_choice = 0

    ckpt_path = './model/tanh/model_epigraphical_narrowroad_tanh.pth'
    activation = 'tanh'

    # Initialize and load the model
    model = modules_adaptive.SingleBVPNet(in_features=9, out_features=1, type=activation, mode='mlp',
                                          final_layer_factor=1., hidden_features=64, num_hidden_layers=3)
    model.to(device)
    checkpoint = torch.load(ckpt_path)
    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    model.eval()

    "initial state space including 600 trajectories"
    path = './test_data/data_test_narrowroad_600.mat'
    test_data = scio.loadmat(path)

    t = test_data['t']
    X = test_data['X']
    test_data.update({'t0': test_data['t']})
    idx0 = np.nonzero(np.equal(test_data.pop('t0'), 0))[1]
    idxT = idx0 + 150

    print(len(idx0))
    X0 = np.zeros((len(idx0), 8))
    for n in range(1, len(idx0) + 1):
        X0[n - 1, :] = X[:, idx0[n - 1]]

    X0 = X0.T

    print(len(idxT))
    XT = np.zeros((len(idxT), 8))
    for n in range(1, len(idxT) + 1):
        XT[n - 1, :] = X[:, idxT[n - 1]]

    XT = XT.T

    N = 151
    Time = np.linspace(0, 3, num=N)
    dt = Time[1] - Time[0]
    Time = np.flip(Time)  # invert time to fit for network input setting

    dx1 = np.zeros((len(idx0), Time.shape[0]))
    dy1 = np.zeros((len(idx0), Time.shape[0]))
    theta1 = np.zeros((len(idx0), Time.shape[0]))
    v1 = np.zeros((len(idx0), Time.shape[0]))
    u1 = np.zeros((len(idx0), Time.shape[0]))
    w1 = np.zeros((len(idx0), Time.shape[0]))
    dx2 = np.zeros((len(idx0), Time.shape[0]))
    dy2 = np.zeros((len(idx0), Time.shape[0]))
    theta2 = np.zeros((len(idx0), Time.shape[0]))
    v2 = np.zeros((len(idx0), Time.shape[0]))
    u2 = np.zeros((len(idx0), Time.shape[0]))
    w2 = np.zeros((len(idx0), Time.shape[0]))

    z1 = np.zeros((len(idx0), Time.shape[0]))
    V1 = np.zeros((len(idx0), Time.shape[0]))
    yA1 = np.zeros((len(idx0), Time.shape[0]))
    yB1 = np.zeros((len(idx0), Time.shape[0]))
    z2 = np.zeros((len(idx0), Time.shape[0]))
    V2 = np.zeros((len(idx0), Time.shape[0]))
    yA2 = np.zeros((len(idx0), Time.shape[0]))
    yB2 = np.zeros((len(idx0), Time.shape[0]))
    ct_1 = np.zeros((len(idx0), Time.shape[0]))
    ct_2 = np.zeros((len(idx0), Time.shape[0]))
    ham_1 = np.zeros((len(idx0), Time.shape[0]))
    ham_2 = np.zeros((len(idx0), Time.shape[0]))
    hji_1 = np.zeros((len(idx0), Time.shape[0]))
    hji_2 = np.zeros((len(idx0), Time.shape[0]))
    diff_hji1 = np.zeros((len(idx0), Time.shape[0]))
    diff_hji2 = np.zeros((len(idx0), Time.shape[0]))

    for n in range(len(idx0)):
        dx1[n][0] = X0[0, n]
        dy1[n][0] = X0[1, n]
        theta1[n][0] = X0[2, n]
        v1[n][0] = X0[3, n]
        dx2[n][0] = X0[4, n]
        dy2[n][0] = X0[5, n]
        theta2[n][0] = X0[6, n]
        v2[n][0] = X0[7, n]

    # find optimal auxiliary state z*
    for i in range(X0.shape[1]):
        t0 = np.array([[Time[0]]])
        z1[i][0], z2[i][0] = optimal_z_generation(X0[:, i], t0, model)
        print(i)

    start_time = time.time()

    # closed-loop trajectory generation
    for i in range(X0.shape[1]):
        for j in range(1, Time.shape[0] + 1):
            X_nn = np.array([[dx1[i][j - 1]],
                             [dy1[i][j - 1]],
                             [theta1[i][j - 1]],
                             [v1[i][j - 1]],
                             [z1[i][j - 1]],
                             [dx2[i][j - 1]],
                             [dy2[i][j - 1]],
                             [theta2[i][j - 1]],
                             [v2[i][j - 1]],
                             [z2[i][j - 1]]])
            t_nn = np.array([[Time[j - 1]]])
            u1[i][j - 1], u2[i][j - 1], w1[i][j - 1], w2[i][j - 1], V1[i][j - 1], V2[i][j - 1], ct_1[i][j - 1], ct_2[i][j - 1], \
            ham_1[i][j - 1], ham_2[i][j - 1], hji_1[i][j - 1], hji_2[i][j - 1], diff_hji1[i][j - 1], diff_hji2[i][j - 1], \
            yA1[i][j - 1], yB1[i][j - 1], yA2[i][j - 1], yB2[i][j - 1] = value_action(X_nn, t_nn, model, N_choice)
            if j == Time.shape[0]:
                break
            else:
                dx1[i][j], dy1[i][j], theta1[i][j], v1[i][j], dx2[i][j], dy2[i][j], theta2[i][j], v2[i][j] \
                    = dynamic(X_nn, dt, (u1[i][j - 1], u2[i][j - 1], w1[i][j - 1], w2[i][j - 1]))
                X_new = np.array([dx1[i][j],
                                  dy1[i][j],
                                  theta1[i][j],
                                  v1[i][j],
                                  dx2[i][j],
                                  dy2[i][j],
                                  theta2[i][j],
                                  v2[i][j]])
                t_new = np.array([[Time[j]]])
                z1[i][j], z2[i][j] = optimal_z_generation(X_new, t_new, model)
        print(i)

    print()
    time_spend = time.time() - start_time
    print('Total solution time: %1.1f' % (time_spend), 'sec')
    print()

    data = {'dx1': dx1,
            'dy1': dy1,
            'theta1': theta1,
            'v1': v1,
            'dx2': dx2,
            'dy2': dy2,
            'theta2': theta2,
            'v2': v2,
            'u1': u1,
            'u2': u2,
            'w1': w1,
            'w2': w2,
            'V1': V1,
            'V2': V2,
            'yA1': yA1,
            'yB1': yB1,
            'yA2': yA2,
            'yB2': yB2,
            'z1': z1,
            'z2': z2,
            'ct_1': ct_1,
            'ct_2': ct_2,
            'ham_1': ham_1,
            'ham_2': ham_2,
            'hji_1': hji_1,
            'hji_2': hji_2,
            'diff_hji1': diff_hji1,
            'diff_hji2': diff_hji2,
            't': t}

    final_data = discrete_data(data, dt, N_choice, N)

    save_data = 1  # input('Save data? Enter 0 for no, 1 for yes:')
    if save_data:
        save_path = 'closed_loop/tanh/closedloop_traj_epigraphical_tanh.mat'
        scio.savemat(save_path, final_data)

