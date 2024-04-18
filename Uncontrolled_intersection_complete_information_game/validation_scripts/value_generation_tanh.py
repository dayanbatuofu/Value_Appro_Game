# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname( os.path.abspath(__file__))))

import modules, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def value_function(X, t, alpha, model):
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
    y1 = model_output['model_out'][:, :cut_index] * alpha  # (meta_batch_size, num_points, 1); agent 1's value
    y2 = model_output['model_out'][:, cut_index:] * alpha  # agent 2's value

    jac, _ = diff_operators.jacobian(y, x)
    dv_1 = jac[:, :cut_index, :]
    dv_2 = jac[:, cut_index:, :]

    # agent 1: partial gradient of V w.r.t. state
    dvdx_1 = dv_1[..., 0, 1:].squeeze().reshape(1, dv_1.shape[-1] - 1)

    # unnormalize the costate for agent 1, consider V = exp(u)
    lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2) * alpha  # lambda_11
    lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2) * alpha  # lambda_11
    lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2) * alpha  # lambda_12
    lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2) * alpha  # lambda_12

    # agent 2: partial gradient of V w.r.t. state
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1)

    # unnormalize the costate for agent 2, consider V = exp(u)
    lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2) * alpha  # lambda_21
    lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2) * alpha  # lambda_21
    lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2) * alpha  # lambda_22
    lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2) * alpha  # lambda_22

    # H = lambda^T * f - L
    # Agent 1's action
    u1 = (0.5 * dvdx_1[:, 1:2] / ((32 - 15) / 2)) * alpha

    # Agent 2's action
    u2 = (0.5 * dvdx_2[:, 1:2] / ((32 - 15) / 2)) * alpha

    # set up bounds for u1 and u2
    max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
    min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

    u1[torch.where(u1 > max_acc)] = max_acc
    u1[torch.where(u1 < min_acc)] = min_acc
    u2[torch.where(u2 > max_acc)] = max_acc
    u2[torch.where(u2 < min_acc)] = min_acc

    return y1, y2, lam11_1, lam11_2, lam12_1, lam12_2, lam21_1, lam21_2, lam22_1, lam22_2, u1, u2

if __name__ == '__main__':

    logging_root = './logs'
    Num = 1
    N_neurons = 64
    policy = ['a_a', 'a_na', 'na_a', 'na_na']
    N_choice = 0

    """
    self-supervised and value hardening uses alpha = 10 for (a,a), (a,na), (na,a), (na,na)
    hybrid and supervised uses alpha = 1 for (a,a) and alpha = 10 for (a,na), (na,a), (na,na)
    """
    if N_choice == 0:
        alpha = 1
    else:
        alpha = 10

    ckpt_path = './model_tanh/model_hybrid_' + str(policy[N_choice]) + '_tanh.pth'
    # ckpt_path = './model_tanh/model_supervised_' + str(policy[N_choice]) + '_tanh.pth'
    # ckpt_path = './model_tanh/model_selfsupervised_' + str(policy[N_choice]) + '_tanh.pth'
    # ckpt_path = './model_tanh/model_valuehardening_' + str(policy[N_choice]) + '_tanh.pth'
    activation = 'tanh'

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
    path = 'data_test_' + str(policy[N_choice]) + '_600_18.mat'

    "expanded state space including 500 trajectories"
    # path = 'data_test_' + str(policy[N_choice]) + '_500_18.mat'
    test_data = scio.loadmat(path)

    t = test_data['t']
    X = test_data['X']
    test_data.update({'t0': test_data['t']})
    idx0 = np.nonzero(np.equal(test_data.pop('t0'), 0))[1]
    print(len(idx0))

    N = 151
    Time = np.linspace(0, 3, num=N)
    dt = Time[1] - Time[0]
    Time = np.flip(Time)  # invert time to fit for network input setting

    V1 = np.zeros((len(idx0), N))
    V2 = np.zeros((len(idx0), N))
    A11_1 = np.zeros((len(idx0), N))
    A11_2 = np.zeros((len(idx0), N))
    A12_1 = np.zeros((len(idx0), N))
    A12_2 = np.zeros((len(idx0), N))
    A21_1 = np.zeros((len(idx0), N))
    A21_2 = np.zeros((len(idx0), N))
    A22_1 = np.zeros((len(idx0), N))
    A22_2 = np.zeros((len(idx0), N))
    U1 = np.zeros((len(idx0), N))
    U2 = np.zeros((len(idx0), N))

    start_time = time.time()

    # value and costate generation
    for i in range(1, len(idx0) + 1):
        if i == len(idx0):
            for j in range(1, N + 1):
                d1 = X[0, idx0[i - 1]:][j - 1]
                v1 = X[1, idx0[i - 1]:][j - 1]
                d2 = X[2, idx0[i - 1]:][j - 1]
                v2 = X[3, idx0[i - 1]:][j - 1]
                X_nn = np.vstack((d1, v1, d2, v2))
                t_nn = np.array([[Time[j - 1]]])
                y1, y2, lam11_1, lam11_2, lam12_1, lam12_2, lam21_1, lam21_2, \
                lam22_1, lam22_2, u1, u2 = value_function(X_nn, t_nn, alpha, model)
                V1[i - 1][j - 1] = y1
                V2[i - 1][j - 1] = y2
                A11_1[i - 1][j - 1] = lam11_1
                A11_2[i - 1][j - 1] = lam11_2
                A12_1[i - 1][j - 1] = lam12_1
                A12_2[i - 1][j - 1] = lam12_2
                A21_1[i - 1][j - 1] = lam21_1
                A21_2[i - 1][j - 1] = lam21_2
                A22_1[i - 1][j - 1] = lam22_1
                A22_2[i - 1][j - 1] = lam22_2
                U1[i - 1][j - 1] = u1
                U2[i - 1][j - 1] = u2

        else:
            for j in range(1, N + 1):
                d1 = X[0, idx0[i - 1]: idx0[i]][j - 1]
                v1 = X[1, idx0[i - 1]: idx0[i]][j - 1]
                d2 = X[2, idx0[i - 1]: idx0[i]][j - 1]
                v2 = X[3, idx0[i - 1]: idx0[i]][j - 1]
                X_nn = np.vstack((d1, v1, d2, v2))
                t_nn = np.array([[Time[j - 1]]])
                y1, y2, lam11_1, lam11_2, lam12_1, lam12_2, lam21_1, lam21_2, \
                lam22_1, lam22_2, u1, u2 = value_function(X_nn, t_nn, alpha, model)
                V1[i - 1][j - 1] = y1
                V2[i - 1][j - 1] = y2
                A11_1[i - 1][j - 1] = lam11_1
                A11_2[i - 1][j - 1] = lam11_2
                A12_1[i - 1][j - 1] = lam12_1
                A12_2[i - 1][j - 1] = lam12_2
                A21_1[i - 1][j - 1] = lam21_1
                A21_2[i - 1][j - 1] = lam21_2
                A22_1[i - 1][j - 1] = lam22_1
                A22_2[i - 1][j - 1] = lam22_2
                U1[i - 1][j - 1] = u1
                U2[i - 1][j - 1] = u2

        print(i)

    print()
    time_spend = time.time() - start_time
    print('Total solution time: %1.1f' % (time_spend), 'sec')
    print()

    V1 = V1.flatten()
    V2 = V2.flatten()
    A11_1 = A11_1.flatten()
    A11_2 = A11_2.flatten()
    A12_1 = A12_1.flatten()
    A12_2 = A12_2.flatten()
    A21_1 = A21_1.flatten()
    A21_2 = A21_2.flatten()
    A22_1 = A22_1.flatten()
    A22_2 = A22_2.flatten()
    U1 = U1.flatten()
    U2 = U2.flatten()

    X_OUT = X
    V_OUT = np.vstack((V1, V2))
    t_OUT = t
    A_OUT = np.vstack((A11_1, A11_2, A12_1, A12_2,
                       A21_1, A21_2, A22_1, A22_2))
    U_OUT = np.vstack((U1, U2))

    data = {'X': X_OUT,
            't': t_OUT,
            'V': V_OUT,
            'A': A_OUT,
            'U': U_OUT}

    save_data = 1  # input('Save data? Enter 0 for no, 1 for yes:')
    if save_data:
        save_path = 'value/tanh/value_generation_hybrid_initial_' + str(policy[N_choice]) + '_tanh.mat'
        # save_path = 'value/tanh/value_generation_supervised_initial_' + str(policy[N_choice]) + '_tanh.mat'
        # save_path = 'value/tanh/value_generation_selfsupervised_initial_' + str(policy[N_choice]) + '_tanh.mat'
        # save_path = 'value/tanh/value_generation_valuehardening_initial_' + str(policy[N_choice]) + '_tanh.mat'
        # save_path = 'value/tanh/value_generation_hybrid_expanded_' + str(policy[N_choice]) + '_tanh.mat'
        # save_path = 'value/tanh/value_generation_supervised_expanded_' + str(policy[N_choice]) + '_tanh.mat'
        # save_path = 'value/tanh/value_generation_selfsupervised_expanded_' + str(policy[N_choice]) + '_tanh.mat'
        # save_path = 'value/tanh/value_generation_valuehardening_expanded_' + str(policy[N_choice]) + '_tanh.mat'
        scio.savemat(save_path, data)

