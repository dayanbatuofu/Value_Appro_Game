# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def value_action(X, t, model, alpha):
    # normalize the state for agent 1, agent 2
    dx_1 = 2.0 * (X[0, :] - 0) / (95 - 0) - 1.
    dy_1 = 2.0 * (X[1, :] - 31) / (37 - 31) - 1.
    theta_1 = 2.0 * (X[2, :] - (-0.15)) / (0.13 - (-0.15)) - 1.
    v_1 = 2.0 * (X[3, :] - 17) / (26 - 17) - 1.
    dx_2 = 2.0 * (X[4, :] - 0) / (95 - 0) - 1.
    dy_2 = 2.0 * (X[5, :] - 33) / (39 - 33) - 1.
    theta_2 = 2.0 * (X[6, :] - (-0.13)) / (0.15 - (-0.13)) - 1.
    v_2 = 2.0 * (X[7, :] - 17) / (26 - 17) - 1.
    label1 = torch.zeros(1, 1)
    label2 = torch.ones(1, 1)

    X1 = np.vstack((dx_1, dy_1, theta_1, v_1, dx_2, dy_2, theta_2, v_2, label1))
    X2 = np.vstack((dx_2, dy_2, theta_2, v_2, dx_1, dy_1, theta_1, v_1, label2))

    X1 = torch.tensor(X1, dtype=torch.float32, requires_grad=True).T
    X2 = torch.tensor(X2, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    coords_1 = torch.cat((t, X1), dim=1)
    coords_2 = torch.cat((t, X2), dim=1)
    coords = torch.cat((coords_1, coords_2), dim=0).unsqueeze(0)
    model_in = {'coords': coords.to(device)}
    model_output = model(model_in)

    x = model_output['model_in']
    y = model_output['model_out']
    cut_index = x.shape[1] // 2
    y1 = model_output['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
    y2 = model_output['model_out'][:, cut_index:]  # agent 2's value

    jac, _ = diff_operators.jacobian(y, x)
    dv_1 = jac[:, :cut_index, :]
    dv_2 = jac[:, cut_index:, :]

    # agent 1: partial gradient of V w.r.t. state
    dvdx_1 = dv_1[..., 0, 1:].squeeze().reshape(1, dv_1.shape[-1] - 1)

    # unnormalize the costate for agent 1, consider V = exp(u)
    lam11_1 = dvdx_1[:, :1] / ((95 - 0) / 2)  # lambda_11
    lam11_2 = dvdx_1[:, 1:2] / ((37 - 31) / 2)  # lambda_11
    lam11_3 = dvdx_1[:, 2:3] / ((0.13 - (-0.15)) / 2)  # lambda_11
    lam11_4 = dvdx_1[:, 3:4] / ((26 - 17) / 2)  # lambda_11
    lam12_1 = dvdx_1[:, 4:5] / ((95 - 0) / 2)  # lambda_12
    lam12_2 = dvdx_1[:, 5:6] / ((39 - 33) / 2)  # lambda_12
    lam12_3 = dvdx_1[:, 6:7] / ((0.15 - (-0.13)) / 2)  # lambda_12
    lam12_4 = dvdx_1[:, 7:8] / ((26 - 17) / 2)  # lambda_12

    # agent 2: partial gradient of V w.r.t. state
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1)

    # unnormalize the costate for agent 2, consider V = exp(u)
    lam21_1 = dvdx_2[:, 4:5] / ((95 - 0) / 2)  # lambda_21
    lam21_2 = dvdx_2[:, 5:6] / ((37 - 31) / 2)  # lambda_21
    lam21_3 = dvdx_2[:, 6:7] / ((0.13 - (-0.15)) / 2)  # lambda_21
    lam21_4 = dvdx_2[:, 7:8] / ((26 - 17) / 2)  # lambda_21
    lam22_1 = dvdx_2[:, :1] / ((95 - 0) / 2)  # lambda_22
    lam22_2 = dvdx_2[:, 1:2] / ((39 - 33) / 2)  # lambda_22
    lam22_3 = dvdx_2[:, 2:3] / ((0.15 - (-0.13)) / 2)  # lambda_22
    lam22_4 = dvdx_2[:, 3:4] / ((26 - 17) / 2)  # lambda_22

    max_acc_u = torch.tensor([10.], dtype=torch.float32).to(device)
    min_acc_u = torch.tensor([-5.], dtype=torch.float32).to(device)
    max_acc_w = torch.tensor([1.], dtype=torch.float32).to(device)
    min_acc_w = torch.tensor([-1.], dtype=torch.float32).to(device)

    # action for agent 1
    U1 = 0.5 * lam11_4 * alpha
    W1 = lam11_3 * alpha / 200

    # action for agent 2
    U2 = 0.5 * lam22_4 * alpha
    W2 = lam22_3 * alpha / 200

    U1[torch.where(U1 > max_acc_u)] = max_acc_u
    U1[torch.where(U1 < min_acc_u)] = min_acc_u
    U2[torch.where(U2 > max_acc_u)] = max_acc_u
    U2[torch.where(U2 < min_acc_u)] = min_acc_u

    W1[torch.where(W1 > max_acc_w)] = max_acc_w
    W1[torch.where(W1 < min_acc_w)] = min_acc_w
    W2[torch.where(W2 > max_acc_w)] = max_acc_w
    W2[torch.where(W2 < min_acc_w)] = min_acc_w

    return U1, U2, W1, W2

def dynamic(X_nn, dt, action):
    u1, u2, w1, w2 = action
    theta1 = X_nn[2, :] + w1 * dt
    theta2 = X_nn[6, :] + w2 * dt
    v1 = X_nn[3, :] + u1 * dt
    v2 = X_nn[7, :] + u2 * dt
    dx1 = X_nn[0, :] + v1 * np.cos(theta1) * dt
    dy1 = X_nn[1, :] + v1 * np.sin(theta1) * dt
    dx2 = X_nn[4, :] + v2 * np.cos(theta2) * dt
    dy2 = X_nn[5, :] + v2 * np.sin(theta2) * dt

    return dx1, dy1, theta1, v1, dx2, dy2, theta2, v2

def discrete_data(data, dt, N):
    dx1 = data['dx1']
    dy1 = data['dy1']
    theta1 = data['theta1']
    v1 = data['v1']
    dx2 = data['dx2']
    dy2 = data['dy2']
    theta2 = data['theta2']
    v2 = data['v2']
    u1 = data['u1']
    u2 = data['u2']
    w1 = data['w1']
    w2 = data['w2']
    time_horizon = N

    alpha = 1e-06
    beta = 10000
    threshold = 2.5

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

            dist_diff = (-(torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) - threshold) * 5)

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
            V1[i][j] = alpha * dx1[i][-1] - (v1[i][-1] - 18) ** 2 - (dy1[i][-1] - 37) ** 2 - Loss1[i][j] - 100 * (theta1[i][-1] - 0) ** 2
            V2[i][j] = alpha * dx2[i][-1] - (v2[i][-1] - 18) ** 2 - (dy2[i][-1] - 33) ** 2 - Loss2[i][j] - 100 * (theta2[i][-1] - 0) ** 2

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
            'U': np.vstack((U1.reshape(1, -1), U2.reshape(1, -1))),
            'Omega': np.vstack((W1.reshape(1, -1), W2.reshape(1, -1)))}

    return data

if __name__ == '__main__':

    logging_root = './logs'

    """
    value hardening uses alpha = 1/10
    self-supervised， hybrid and supervised uses alpha = 1
    """

    ckpt_path = './model/sine/model_hybrid_lane_sine.pth'
    # ckpt_path = './model/new/model_supervised_lane_sine.pth'
    # ckpt_path = './model/sine/model_pinn_lane_sine.pth'
    # ckpt_path = './model/sine/model_valuehardening_lane_sine.pth'
    activation = 'sine'
    alpha = 1

    # Initialize and load the model
    model = modules.SingleBVPNet(in_features=10, out_features=1, type=activation, mode='mlp',
                                 final_layer_factor=1., hidden_features=64, num_hidden_layers=3)
    model.to(device)
    checkpoint = torch.load(ckpt_path)
    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    model.eval()

    path = './test_data/data_test_lane_600.mat'
    test_data = scio.loadmat(path)

    t = test_data['t']
    X = test_data['X']
    test_data.update({'t0': test_data['t']})
    idx0 = np.nonzero(np.equal(test_data.pop('t0'), 0))[1]

    print(len(idx0))
    X0 = np.zeros((len(idx0), 8))
    for n in range(1, len(idx0) + 1):
        X0[n - 1, :] = X[:, idx0[n - 1]]

    X0 = X0.T

    N = 201
    Time = np.linspace(0, 4, num=N)
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

    for n in range(len(idx0)):
        dx1[n][0] = X0[0, n]
        dy1[n][0] = X0[1, n]
        theta1[n][0] = X0[2, n]
        v1[n][0] = X0[3, n]
        dx2[n][0] = X0[4, n]
        dy2[n][0] = X0[5, n]
        theta2[n][0] = X0[6, n]
        v2[n][0] = X0[7, n]

    start_time = time.time()

    # closed-loop trajectory generation
    for i in range(X0.shape[1]):
        for j in range(1, Time.shape[0] + 1):
            X_nn = np.array([[dx1[i][j - 1]],
                             [dy1[i][j - 1]],
                             [theta1[i][j - 1]],
                             [v1[i][j - 1]],
                             [dx2[i][j - 1]],
                             [dy2[i][j - 1]],
                             [theta2[i][j - 1]],
                             [v2[i][j - 1]]])
            t_nn = np.array([[Time[j - 1]]])
            u1[i][j - 1], u2[i][j - 1], w1[i][j - 1], w2[i][j - 1] = value_action(X_nn, t_nn, model, alpha)
            if j == Time.shape[0]:
                break
            else:
                dx1[i][j], dy1[i][j], theta1[i][j], v1[i][j], dx2[i][j], dy2[i][j], theta2[i][j], v2[i][j] \
                    = dynamic(X_nn, dt, (u1[i][j - 1], u2[i][j - 1], w1[i][j - 1], w2[i][j - 1]))
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
            't': t}

    final_data = discrete_data(data, dt, N)

    save_data = 1  # input('Save data? Enter 0 for no, 1 for yes:')
    if save_data:
        # save_path = 'closed_loop/sine/closedloop_traj_hybrid_lane_sine.mat'
        # save_path = 'closed_loop/new/closedloop_traj_supervised_lane_sine.mat'
        # save_path = 'closed_loop/sine/closedloop_traj_pinn_lane_sine.mat'
        save_path = 'closed_loop/sine/closedloop_traj_valuehardening_lane_sine.mat'
        scio.savemat(save_path, final_data)

