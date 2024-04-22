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

def value_action(X, t, model, N_choice, alpha):
    # normalize the state for agent 1, agent 2
    d1 = 2.0 * (X[0, :] - 15) / (105 - 15) - 1.
    v1 = 2.0 * (X[1, :] - 15) / (32 - 15) - 1.
    d2 = 2.0 * (X[2, :] - 15) / (105 - 15) - 1.
    v2 = 2.0 * (X[3, :] - 15) / (32 - 15) - 1.

    X = np.vstack((d1, v1, d2, v2))

    X = torch.tensor(X, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    coords_1 = torch.cat((t, X), dim=1)
    coords_2 = torch.cat((t, (torch.cat((X[:, 2:], X[:, :2]), dim=1))), dim=1)
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
    lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
    lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
    lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
    lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

    # agent 2: partial gradient of V w.r.t. state
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1)

    # unnormalize the costate for agent 2, consider V = exp(u)
    lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
    lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
    lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
    lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

    max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
    min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

    if N_choice == 0:
        # action for agent 1
        U1 = 0.5 * lam11_2 * alpha
        # action for agent 2
        U2 = 0.5 * lam22_2 * alpha
    else:
        # action for agent 1
        U1 = 0.5 * lam11_2 * alpha
        # action for agent 2
        U2 = 0.5 * lam22_2 * alpha

    U1[torch.where(U1 > max_acc)] = max_acc
    U1[torch.where(U1 < min_acc)] = min_acc
    U2[torch.where(U2 > max_acc)] = max_acc
    U2[torch.where(U2 < min_acc)] = min_acc

    return U1, U2

def dynamic(X_nn, dt, action):
    u1, u2 = action
    v1 = X_nn[1, :] + u1 * dt
    v2 = X_nn[3, :] + u2 * dt
    d1 = X_nn[0, :] + v1 * dt
    d2 = X_nn[2, :] + v2 * dt

    return d1, v1, d2, v2

def discrete_data(data, dt, N_choice, N):
    d1 = data['d1']
    d2 = data['d2']
    v1 = data['v1']
    v2 = data['v2']
    u1 = data['u1']
    u2 = data['u2']
    time_horizon = N

    R1 = 70
    R2 = 70
    L1 = 3
    L2 = 3
    W1 = 1.5
    W2 = 1.5
    alpha = 1e-06
    beta = 10000

    if N_choice == 0:
        theta1, theta2 = 1, 1
    if N_choice == 1:
        theta1, theta2 = 1, 5
    if N_choice == 2:
        theta1, theta2 = 5, 1
    if N_choice == 3:
        theta1, theta2 = 5, 5

    t_step = dt

    U1 = torch.tensor(u1, requires_grad=True, dtype=torch.float32)
    U2 = torch.tensor(u2, requires_grad=True, dtype=torch.float32)

    V1 = np.zeros((len(U1[:, 0]), len(U1[0])))
    Loss1 = np.zeros((len(U1[:, 0]), len(U1[0])))
    Loss1_tmp = np.zeros((len(U1[:, 0]), len(U1[0])))
    V2 = np.zeros((len(U2[:, 0]), len(U2[0])))
    Loss2 = np.zeros((len(U2[:, 0]), len(U2[0])))
    Loss2_tmp = np.zeros((len(U2[:, 0]), len(U2[0])))

    for i in range(len(U1[:, 0])):
        for j in range(time_horizon):
            x1 = torch.tensor(d1[i][j], requires_grad=True, dtype=torch.float32)
            x2 = torch.tensor(d2[i][j], requires_grad=True, dtype=torch.float32)
            x1_in = (x1 - R1 / 2 + theta1 * W2 / 2) * 5
            x1_out = -(x1 - R1 / 2 - W2 / 2 - L1) * 5
            x2_in = (x2 - R2 / 2 + theta2 * W1 / 2) * 5
            x2_out = -(x2 - R2 / 2 - W1 / 2 - L2) * 5
            Loss1_tmp[i][j] = (U1[i][j] ** 2 + beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                x2_in) * torch.sigmoid(x2_out)) * t_step
            Loss2_tmp[i][j] = (U2[i][j] ** 2 + beta * torch.sigmoid(x1_in) * torch.sigmoid(x1_out) * torch.sigmoid(
                x2_in) * torch.sigmoid(x2_out)) * t_step

    U1 = U1.detach().cpu().numpy()
    U2 = U2.detach().cpu().numpy()

    for i in range(len(U1[:, 0])):
        for j in range(time_horizon):
            Loss1[i][j] = np.sum(Loss1_tmp[i][j:])
            Loss2[i][j] = np.sum(Loss2_tmp[i][j:])

    for i in range(len(U1[:, 0])):
        for j in range(time_horizon):
            V1[i][j] = alpha * d1[i][-1] - (v1[i][-1] - 18) ** 2 - Loss1[i][j]
            V2[i][j] = alpha * d2[i][-1] - (v2[i][-1] - 18) ** 2 - Loss2[i][j]

    data = {'t': data['t'],
            'X': np.vstack((d1.reshape(1, -1),
                            v1.reshape(1, -1),
                            d2.reshape(1, -1),
                            v2.reshape(1, -1))),
            'V': np.vstack((V1.reshape(1, -1),
                            V2.reshape(1, -1))),
            'U': np.vstack((U1.reshape(1, -1), U2.reshape(1, -1)))}

    return data

if __name__ == '__main__':

    logging_root = './logs'
    N_neurons = 64

    policy = ['a_a']
    N_choice = 0

    """
    self-supervised uses alpha = 10 for (a,a), (a,na), (na,a), (na,na)
    hybrid and supervised uses alpha = 1 for (a,a) and alpha = 10 for (a,na), (na,a), (na,na)
    """
    if N_choice == 0:
        alpha = 1
    else:
        alpha = 10

    ckpt_path = './model/gelu/model_hybrid_' + str(policy[N_choice]) + '_gelu.pth'
    # ckpt_path = './model/gelu/model_supervised_' + str(policy[N_choice]) + '_gelu.pth'
    # ckpt_path = './model/gelu/model_pinn_' + str(policy[N_choice]) + '_gelu.pth'
    activation = 'gelu'

    # Initialize and load the model
    model = modules.SingleBVPNet(in_features=5, out_features=1, type=activation, mode='mlp',
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
    path = './test_data/data_test_' + str(policy[N_choice]) + '_600.mat'

    "expanded state space including 500 trajectories"
    # path = './test_data/data_test_' + str(policy[N_choice]) + '_500.mat'
    test_data = scio.loadmat(path)

    t = test_data['t']
    X = test_data['X']
    test_data.update({'t0': test_data['t']})
    idx0 = np.nonzero(np.equal(test_data.pop('t0'), 0))[1]

    print(len(idx0))
    X0 = np.zeros((len(idx0), 4))
    for n in range(1, len(idx0) + 1):
        X0[n - 1, :] = X[:, idx0[n - 1]]

    X0 = X0.T

    N = 151
    Time = np.linspace(0, 3, num=N)
    dt = Time[1] - Time[0]
    Time = np.flip(Time)  # invert time to fit for network input setting

    d1 = np.zeros((len(idx0), Time.shape[0]))
    v1 = np.zeros((len(idx0), Time.shape[0]))
    u1 = np.zeros((len(idx0), Time.shape[0]))
    d2 = np.zeros((len(idx0), Time.shape[0]))
    v2 = np.zeros((len(idx0), Time.shape[0]))
    u2 = np.zeros((len(idx0), Time.shape[0]))

    for n in range(len(idx0)):
        d1[n][0] = X0[0, n]
        v1[n][0] = X0[1, n]
        d2[n][0] = X0[2, n]
        v2[n][0] = X0[3, n]

    start_time = time.time()

    # closed-loop trajectory generation
    for i in range(X0.shape[1]):
        last_action = (u1[i][0], u2[i][0])

        for j in range(1, Time.shape[0] + 1):
            X_nn = np.array([[d1[i][j - 1]], [v1[i][j - 1]], [d2[i][j - 1]], [v2[i][j - 1]]])
            t_nn = np.array([[Time[j - 1]]])
            u1[i][j - 1], u2[i][j - 1] = value_action(X_nn, t_nn, model, N_choice, alpha)
            if j == Time.shape[0]:
                break
            else:
                d1[i][j], v1[i][j], d2[i][j], v2[i][j] = dynamic(X_nn, dt, (u1[i][j - 1], u2[i][j - 1]))
                last_action = (u1[i][j - 1], u2[i][j - 1])

        print(i)

    print()
    time_spend = time.time() - start_time
    print('Total solution time: %1.1f' % (time_spend), 'sec')
    print()

    data = {'d1': d1,
            'd2': d2,
            'v1': v1,
            'v2': v2,
            'u1': u1,
            'u2': u2,
            't': t}

    final_data = discrete_data(data, dt, N_choice, N)

    save_data = 1  # input('Save data? Enter 0 for no, 1 for yes:')
    if save_data:
        save_path = 'closed_loop/gelu/closedloop_traj_hybrid_initial_' + str(policy[N_choice]) + '_gelu.mat'
        # save_path = 'closed_loop/gelu/closedloop_traj_supervised_initial_' + str(policy[N_choice]) + '_gelu.mat'
        # save_path = 'closed_loop/gelu/closedloop_traj_pinn_initial_' + str(policy[N_choice]) + '_gelu.mat'
        # save_path = 'closed_loop/gelu/closedloop_traj_hybrid_expanded_' + str(policy[N_choice]) + '_gelu.mat'
        # save_path = 'closed_loop/gelu/closedloop_traj_supervised_expanded_' + str(policy[N_choice]) + '_gelu.mat'
        # save_path = 'closed_loop/gelu/closedloop_traj_pinn_expanded_' + str(policy[N_choice]) + '_gelu.mat'
        scio.savemat(save_path, final_data)

