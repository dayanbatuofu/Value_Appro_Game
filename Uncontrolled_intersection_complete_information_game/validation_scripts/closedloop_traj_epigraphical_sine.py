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
        d1 = np.array([[2.0 * (X[0] - 15) / (105 - 15) - 1]])
        v1 = np.array([[2.0 * (X[1] - 15) / (32 - 15) - 1]])
        d2 = np.array([[2.0 * (X[2] - 15) / (105 - 15) - 1]])
        v2 = np.array([[2.0 * (X[3] - 15) / (32 - 15) - 1]])

        z1 = np.array([[a]])
        X1 = np.vstack((d1, v1, d2, v2, z1))
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
            z_star = 400
        elif yB < 0:
            z_star = z1
        else:
            z_star = z1 + y1.detach().cpu().numpy().squeeze(0)

        return z_star

    def dichotomy2(a, b, threshold, X, t, model):
        d1 = np.array([[2.0 * (X[0] - 15) / (105 - 15) - 1]])
        v1 = np.array([[2.0 * (X[1] - 15) / (32 - 15) - 1]])
        d2 = np.array([[2.0 * (X[2] - 15) / (105 - 15) - 1]])
        v2 = np.array([[2.0 * (X[3] - 15) / (32 - 15) - 1]])

        z2 = np.array([[a]])
        X2 = np.vstack((d2, v2, d1, v1, z2))
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
            z_star = 400
        elif yB < 0:
            z_star = z2
        else:
            z_star = z2 + y2.detach().cpu().numpy().squeeze(0)

        return z_star

    z_min = -1.05e-4   # -1.05e-4
    z_max = 300
    threshold = 0

    z1_0 = dichotomy1(z_min, z_max, threshold, X, t, model)
    z2_0 = dichotomy2(z_min, z_max, threshold, X, t, model)

    return z1_0, z2_0

def value_action(X, t, model, N_choice):
    # normalize the state for agent 1, agent 2
    d1 = 2.0 * (X[0, :] - 15) / (105 - 15) - 1.
    v1 = 2.0 * (X[1, :] - 15) / (32 - 15) - 1.
    z1 = X[2, :]
    d2 = 2.0 * (X[3, :] - 15) / (105 - 15) - 1.
    v2 = 2.0 * (X[4, :] - 15) / (32 - 15) - 1.
    z2 = X[5, :]

    if N_choice == 0 or 3:
        X1 = np.vstack((d1, v1, d2, v2, z1))
        X2 = np.vstack((d2, v2, d1, v1, z2))
    else:
        label1 = torch.zeros(1, 1)
        label2 = torch.ones(1, 1)

        X1 = np.vstack((d1, v1, d2, v2, label1, z1))
        X2 = np.vstack((d2, v2, d1, v1, label2, z2))

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
    lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
    lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
    lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
    lam12_2 = dvdx_1[:, 3:4] / ((32 - 15) / 2)  # lambda_12
    lam1_z = dvdx_1[:, -1:]  # lambda1_z

    # agent 2: partial gradient of V w.r.t. time and state
    dvdt_2 = dv_2[..., 0, 0].squeeze()
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1)

    # unnormalize the costate for agent 2, consider V = exp(u)
    lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
    lam21_2 = dvdx_2[:, 3:4] / ((32 - 15) / 2)  # lambda_21
    lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
    lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22
    lam2_z = dvdx_2[:, -1:]  # lambda2_z

    # set up bounds for u1 and u2
    max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
    min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

    index_P1_1 = torch.where(lam1_z == -1)[0]   # -1
    index_P2_1 = torch.where(lam2_z == -1)[0]

    # u1 = 0.5 * lam11_2 / lam1_z
    u1 = 1 * lam11_2
    u1[torch.where(u1 > 0)] = 1
    u1[torch.where(u1 < 0)] = -1
    u1[torch.where(u1 == 1)] = min_acc
    u1[torch.where(u1 == -1)] = max_acc
    u1[index_P1_1] = (0.5 * lam11_2[index_P1_1] / lam1_z[index_P1_1])

    # Agent 2's action, be careful about the order of u2>0 and u2<0
    # u2 = 0.5 * lam22_2 / lam2_z
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

    if N_choice == 0:
        epsilon = (torch.tensor([4.5], dtype=torch.float32)).to(device)  # collision ratio
    else:
        epsilon = (torch.tensor([7.5], dtype=torch.float32)).to(device)  # collision ratio

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

    # calculate the collision area lower and upper bounds
    if N_choice == 0:
        ct_1 = (epsilon - (torch.abs((d11 - 36.5) + (d12 - 36.5)) + torch.abs((d11 - 36.5) - (d12 - 36.5)))).squeeze()
        ct_2 = (epsilon - (torch.abs((d21 - 36.5) + (d22 - 36.5)) + torch.abs((d21 - 36.5) - (d22 - 36.5)))).squeeze()
    if N_choice == 1:
        ct_1 = (epsilon - (torch.abs((5/3)*(d11 - 36.5) + (d12 - 35)) + torch.abs((5/3)*(d11 - 36.5) - (d12 - 35)))).squeeze()
        ct_2 = (epsilon - (torch.abs((5/3)*(d21 - 36.5) + (d22 - 35)) + torch.abs((5/3)*(d21 - 36.5) - (d22 - 35)))).squeeze()
    if N_choice == 2:
        ct_1 = (epsilon - (torch.abs((d11 - 35) + (5/3)*(d12 - 36.5)) + torch.abs((d11 - 35) - (5/3)*(d12 - 36.5)))).squeeze()
        ct_2 = (epsilon - (torch.abs((d21 - 35) + (5/3)*(d22 - 36.5)) + torch.abs((d21 - 35) - (5/3)*(d22 - 36.5)))).squeeze()
    if N_choice == 3:
        ct_1 = (epsilon - (torch.abs((d11 - 35) + (d12 - 35)) + torch.abs((d11 - 35) - (d12 - 35)))).squeeze()
        ct_2 = (epsilon - (torch.abs((d21 - 35) + (d22 - 35)) + torch.abs((d21 - 35) - (d22 - 35)))).squeeze()

    # calculate hamiltonian, -H = (dV/dx)^T * f - (dV/dz)^T * L
    ham_1 = lam11_1.squeeze() * v11.squeeze() + lam11_2.squeeze() * u1.squeeze() + lam12_1.squeeze() * v12.squeeze() \
            + lam12_2.squeeze() * u2.squeeze() - lam1_z.squeeze() * (u1 ** 2).squeeze()
    ham_2 = lam21_1.squeeze() * v21.squeeze() + lam21_2.squeeze() * u1.squeeze() + lam22_1.squeeze() * v22.squeeze() \
            + lam22_2.squeeze() * u2.squeeze() - lam2_z.squeeze() * (u2 ** 2).squeeze()

    diff_constraint_hom_1 = torch.max(ct_1 - y1.squeeze(), -dvdt_1 + ham_1)
    diff_constraint_hom_2 = torch.max(ct_2 - y2.squeeze(), -dvdt_2 + ham_2)

    return u1, u2, y1, y2, ct_1-y1.squeeze(), ct_2-y2.squeeze(), ham_1, ham_2, \
           -dvdt_1+ham_1, -dvdt_2+ham_2, diff_constraint_hom_1, diff_constraint_hom_2, yA1, yB1, yA2, yB2

def dynamic(X_nn, dt, action):
    u1, u2 = action
    v1 = X_nn[1, :] + u1 * dt
    v2 = X_nn[4, :] + u2 * dt
    d1 = X_nn[0, :] + v1 * dt
    d2 = X_nn[3, :] + v2 * dt

    return d1, v1, d2, v2

def discrete_data(data, dt, N_choice, N):
    d1 = data['d1']
    d2 = data['d2']
    v1 = data['v1']
    v2 = data['v2']
    u1 = data['u1']
    u2 = data['u2']
    z1 = data['z1']
    z2 = data['z2']
    y1 = data['V1']
    y2 = data['V2']
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
            'U': np.vstack((U1.reshape(1, -1),
                            U2.reshape(1, -1))),
            'z': np.vstack((z1.reshape(1, -1),
                            z2.reshape(1, -1))),
            'y': np.vstack((y1.reshape(1, -1),
                            y2.reshape(1, -1))),
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
    N_choice = 3

    ckpt_path = './model/sine/model_epigraphical_' + str(policy[N_choice]) + '_sine.pth'
    activation = 'sine'

    # Initialize and load the model
    if N_choice == 0 or 3:
        input_dim = 5
    else:
        input_dim = 6
    model = modules_adaptive.SingleBVPNet(in_features=input_dim, out_features=1, type=activation, mode='mlp',
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
    idxT = idx0 + 150

    print(len(idx0))
    X0 = np.zeros((len(idx0), 4))
    for n in range(1, len(idx0) + 1):
        X0[n - 1, :] = X[:, idx0[n - 1]]

    X0 = X0.T

    print(len(idxT))
    XT = np.zeros((len(idxT), 4))
    for n in range(1, len(idxT) + 1):
        XT[n - 1, :] = X[:, idxT[n - 1]]

    XT = XT.T

    N = 151
    Time = np.linspace(0, 3, num=N)
    dt = Time[1] - Time[0]
    Time = np.flip(Time)  # invert time to fit for network input setting

    d1 = np.zeros((len(idx0), Time.shape[0]))
    v1 = np.zeros((len(idx0), Time.shape[0]))
    z1 = np.zeros((len(idx0), Time.shape[0]))
    u1 = np.zeros((len(idx0), Time.shape[0]))
    V1 = np.zeros((len(idx0), Time.shape[0]))
    yA1 = np.zeros((len(idx0), Time.shape[0]))
    yB1 = np.zeros((len(idx0), Time.shape[0]))
    d2 = np.zeros((len(idx0), Time.shape[0]))
    v2 = np.zeros((len(idx0), Time.shape[0]))
    z2 = np.zeros((len(idx0), Time.shape[0]))
    u2 = np.zeros((len(idx0), Time.shape[0]))
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
        d1[n][0] = X0[0, n]
        v1[n][0] = X0[1, n]
        d2[n][0] = X0[2, n]
        v2[n][0] = X0[3, n]

    # find optimal auxiliary state z*
    for i in range(X0.shape[1]):
        t0 = np.array([[Time[0]]])
        z1[i][0], z2[i][0] = optimal_z_generation(X0[:, i], t0, model)
        print(i)

    start_time = time.time()

    # closed-loop trajectory generation
    for i in range(X0.shape[1]):
        for j in range(1, Time.shape[0] + 1):
            X_nn = np.array([[d1[i][j - 1]], [v1[i][j - 1]], [z1[i][j - 1]], [d2[i][j - 1]], [v2[i][j - 1]], [z2[i][j - 1]]])
            t_nn = np.array([[Time[j - 1]]])
            u1[i][j - 1], u2[i][j - 1], V1[i][j - 1], V2[i][j - 1], ct_1[i][j - 1], ct_2[i][j - 1], \
            ham_1[i][j - 1], ham_2[i][j - 1], hji_1[i][j - 1], hji_2[i][j - 1], diff_hji1[i][j - 1], diff_hji2[i][j - 1], \
            yA1[i][j - 1], yB1[i][j - 1], yA2[i][j - 1], yB2[i][j - 1] = value_action(X_nn, t_nn, model, N_choice)
            if j == Time.shape[0]:
                break
            else:
                d1[i][j], v1[i][j], d2[i][j], v2[i][j] = dynamic(X_nn, dt, (u1[i][j - 1], u2[i][j - 1]))
                X_new = np.array([d1[i][j], v1[i][j], d2[i][j], v2[i][j]])
                t_new = np.array([[Time[j]]])
                z1[i][j], z2[i][j] = optimal_z_generation(X_new, t_new, model)
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
        save_path = 'closed_loop/sine/closedloop_traj_epigraphical_initial_' + str(policy[N_choice]) + '_sine.mat'
        # save_path = 'closed_loop/sine/closedloop_traj_epigraphical_expanded_' + str(policy[N_choice]) + '_sine.mat'
        scio.savemat(save_path, final_data)

