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
    dx_1 = 2.0 * (X[0, :] - 0) / (15.5 - 0) - 1.
    dy_1 = 2.0 * (X[1, :] - 0) / (15.5 - 0) - 1.
    dz_1 = 2.0 * (X[2, :] - (-1.8)) / (2 - (-1.8)) - 1.
    vx_1 = 2.0 * (X[3, :] - 0.3) / (4.5 - 0.3) - 1.
    vy_1 = 2.0 * (X[4, :] - 0.3) / (4.5 - 0.3) - 1.
    vz_1 = 2.0 * (X[5, :] - (-1.8)) / (1.8 - (-1.8)) - 1.
    dx_2 = 2.0 * (X[6, :] - 0) / (15.5 - 0) - 1.
    dy_2 = 2.0 * (X[7, :] - 0) / (15.5 - 0) - 1.
    dz_2 = 2.0 * (X[8, :] - (-1.8)) / (2 - (-1.8)) - 1.
    vx_2 = 2.0 * (X[9, :] - 0.3) / (4.5 - 0.3) - 1.
    vy_2 = 2.0 * (X[10, :] - 0.3) / (4.5 - 0.3) - 1.
    vz_2 = 2.0 * (X[11, :] - (-1.8)) / (1.8 - (-1.8)) - 1.

    X1 = np.vstack((dx_1, dy_1, dz_1, vx_1, vy_1, vz_1, dx_2, dy_2, dz_2, vx_2, vy_2, vz_2))
    X2 = np.vstack((dx_2, dy_2, dz_2, vx_2, vy_2, vz_2, dx_1, dy_1, dz_1, vx_1, vy_1, vz_1))

    X1 = torch.tensor(X1, dtype=torch.float32, requires_grad=True).T
    X2 = torch.tensor(X2, dtype=torch.float32, requires_grad=True).T
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    coords_1 = torch.cat((t, X1), dim=1)
    coords_2 = torch.cat((t, X2), dim=1)
    coords = torch.cat((coords_1, coords_2), dim=0).unsqueeze(0)

    start_time = time.time()
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

    time_spend = time.time() - start_time
    print(time_spend)

    # agent 1: partial gradient of V w.r.t. state
    dvdx_1 = dv_1[..., 0, 1:].squeeze().reshape(1, dv_1.shape[-1] - 1)

    # unnormalize the costate for agent 1, consider V = exp(u)
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

    # agent 2: partial gradient of V w.r.t. state
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1)

    # unnormalize the costate for agent 2, consider V = exp(u)
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

    max_acc_theta = torch.tensor([0.05], dtype=torch.float32).to(device)
    min_acc_theta = torch.tensor([-0.05], dtype=torch.float32).to(device)
    max_acc_phi = torch.tensor([0.05], dtype=torch.float32).to(device)
    min_acc_phi = torch.tensor([-0.05], dtype=torch.float32).to(device)
    max_acc_thrust = torch.tensor([11.81], dtype=torch.float32).to(device)
    min_acc_thrust = torch.tensor([7.81], dtype=torch.float32).to(device)
    gravity = torch.tensor([9.81], dtype=torch.float32).to(device)  # gravity acceleration

    # action for agent 1
    theta1 = torch.atan(lam11_4 * gravity / 200 * alpha)
    phi1 = torch.atan(-lam11_5 * gravity / 200 * alpha)
    thrust1 = lam11_6 / 2 * alpha + gravity

    # action for agent 2
    theta2 = torch.atan(lam22_4 * gravity / 200 * alpha)
    phi2 = torch.atan(-lam22_5 * gravity / 200 * alpha)
    thrust2 = lam22_6 / 2 * alpha + gravity

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

    return theta1, theta2, phi1, phi2, thrust1, thrust2

def dynamic(X_nn, dt, action):
    theta1, theta2, phi1, phi2, thrust1, thrust2 = action
    gravity = 9.81
    vx1 = X_nn[3, :] + gravity * np.tan(theta1) * dt
    vx2 = X_nn[9, :] + gravity * np.tan(theta2) * dt
    vy1 = X_nn[4, :] - gravity * np.tan(phi1) * dt
    vy2 = X_nn[10, :] - gravity * np.tan(phi1) * dt
    vz1 = X_nn[5, :] + (thrust1 - gravity) * dt
    vz2 = X_nn[11, :] + (thrust2 - gravity) * dt
    dx1 = X_nn[0, :] + vx1 * dt
    dy1 = X_nn[1, :] + vy1 * dt
    dz1 = X_nn[2, :] + vz1 * dt
    dx2 = X_nn[6, :] + vx2 * dt
    dy2 = X_nn[7, :] + vy2 * dt
    dz2 = X_nn[8, :] + vz2 * dt

    return dx1, dy1, dz1, vx1, vy1, vz1, dx2, dy2, dz2, vx2, vy2, vz2

def discrete_data(data, dt, N):
    dx1 = data['dx1']
    dy1 = data['dy1']
    dz1 = data['dz1']
    vx1 = data['vx1']
    vy1 = data['vy1']
    vz1 = data['vz1']
    dx2 = data['dx2']
    dy2 = data['dy2']
    dz2 = data['dz2']
    vx2 = data['vx2']
    vy2 = data['vy2']
    vz2 = data['vz2']
    theta1 = data['theta1']
    theta2 = data['theta2']
    phi1 = data['phi1']
    phi2 = data['phi2']
    thrust1 = data['thrust1']
    thrust2 = data['thrust2']
    time_horizon = N

    R1 = 5
    R2 = 5
    alpha = 1e-06
    beta = 10000
    threshold = 0.9  # 1.5
    gravity = 9.81

    t_step = dt

    Theta1 = torch.tensor(theta1, requires_grad=True, dtype=torch.float32)
    Theta2 = torch.tensor(theta2, requires_grad=True, dtype=torch.float32)
    Phi1 = torch.tensor(phi1, requires_grad=True, dtype=torch.float32)
    Phi2 = torch.tensor(phi2, requires_grad=True, dtype=torch.float32)
    Thrust1 = torch.tensor(thrust1, requires_grad=True, dtype=torch.float32)
    Thrust2 = torch.tensor(thrust2, requires_grad=True, dtype=torch.float32)

    V1 = np.zeros((len(Theta1[:, 0]), len(Theta1[0])))
    Loss1 = np.zeros((len(Theta1[:, 0]), len(Theta1[0])))
    Loss1_tmp = np.zeros((len(Theta1[:, 0]), len(Theta1[0])))
    V2 = np.zeros((len(Theta2[:, 0]), len(Theta2[0])))
    Loss2 = np.zeros((len(Theta2[:, 0]), len(Theta2[0])))
    Loss2_tmp = np.zeros((len(Theta2[:, 0]), len(Theta2[0])))

    for i in range(len(Theta1[:, 0])):
        for j in range(time_horizon):
            x1 = torch.tensor(dx1[i][j], requires_grad=True, dtype=torch.float32)
            y1 = torch.tensor(dy1[i][j], requires_grad=True, dtype=torch.float32)
            z1 = torch.tensor(dz1[i][j], requires_grad=True, dtype=torch.float32)
            x2 = torch.tensor(dx2[i][j], requires_grad=True, dtype=torch.float32)
            y2 = torch.tensor(dy2[i][j], requires_grad=True, dtype=torch.float32)
            z2 = torch.tensor(dz2[i][j], requires_grad=True, dtype=torch.float32)

            dist_diff = (-(torch.sqrt(((R1 - x2) - x1) ** 2 + ((R2 - y2) - y1) ** 2 + (z2 - z1) ** 2) - threshold) * 5)

            Loss1_tmp[i][j] = (100 * torch.tan(Theta1[i][j]) ** 2 + 100 * torch.tan(Phi1[i][j]) ** 2 +
                               (Thrust1[i][j] - gravity) ** 2 + beta * torch.sigmoid(dist_diff)) * t_step
            Loss2_tmp[i][j] = (100 * torch.tan(Theta2[i][j]) ** 2 + 100 * torch.tan(Phi1[i][j]) ** 2 +
                               (Thrust2[i][j] - gravity) ** 2 + beta * torch.sigmoid(dist_diff)) * t_step

    Theta1 = Theta1.detach().cpu().numpy()
    Theta2 = Theta2.detach().cpu().numpy()
    Phi1 = Phi1.detach().cpu().numpy()
    Phi2 = Phi2.detach().cpu().numpy()
    Thrust1 = Thrust1.detach().cpu().numpy()
    Thrust2 = Thrust2.detach().cpu().numpy()

    for i in range(len(Theta1[:, 0])):
        for j in range(time_horizon):
            Loss1[i][j] = np.sum(Loss1_tmp[i][j:])
            Loss2[i][j] = np.sum(Loss2_tmp[i][j:])

    for i in range(len(Theta1[:, 0])):
        for j in range(time_horizon):
            V1[i][j] = alpha * dx1[i][-1] + alpha * dy1[i][-1] - (dz1[i][-1] - 0) ** 2 - (vx1[i][-1] - 0) ** 2 - \
                       (vy1[i][-1] - 0) ** 2 - (vz1[i][-1] - 0) ** 2 - Loss1[i][j]
            V2[i][j] = alpha * dx2[i][-1] + alpha * dy2[i][-1] - (dz2[i][-1] - 0) ** 2 - (vx2[i][-1] - 0) ** 2 - \
                       (vy2[i][-1] - 0) ** 2 - (vz2[i][-1] - 0) ** 2 - Loss2[i][j]

    data = {'t': data['t'],
            'X': np.vstack((dx1.reshape(1, -1),
                            dy1.reshape(1, -1),
                            dz1.reshape(1, -1),
                            vx1.reshape(1, -1),
                            vy1.reshape(1, -1),
                            vz1.reshape(1, -1),
                            dx2.reshape(1, -1),
                            dy2.reshape(1, -1),
                            dz2.reshape(1, -1),
                            vx2.reshape(1, -1),
                            vy2.reshape(1, -1),
                            vz2.reshape(1, -1),)),
            'V': np.vstack((V1.reshape(1, -1),
                            V2.reshape(1, -1))),
            'Theta': np.vstack((Theta1.reshape(1, -1), Theta2.reshape(1, -1))),
            'Phi': np.vstack((Phi1.reshape(1, -1), Phi2.reshape(1, -1))),
            'Thrust': np.vstack((Thrust1.reshape(1, -1), Thrust2.reshape(1, -1)))}

    return data

if __name__ == '__main__':

    logging_root = './logs'

    ckpt_path = './model/tanh/model_hybrid_drone_tanh.pth'
    # ckpt_path = './model/tanh/model_supervised_drone_tanh.pth'
    # ckpt_path = './model/tanh/model_pinn_drone_tanh.pth'
    # ckpt_path = './model/tanh/model_valuehardening_drone_tanh.pth'
    activation = 'tanh'

    # Initialize and load the model
    model = modules.SingleBVPNet(in_features=13, out_features=1, type=activation, mode='mlp',
                                 final_layer_factor=1., hidden_features=64, num_hidden_layers=3)
    model.to(device)
    checkpoint = torch.load(ckpt_path)
    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    model.eval()

    alpha = 10

    path = './test_data/data_test_drone_600.mat'
    test_data = scio.loadmat(path)

    t = test_data['t']
    X = test_data['X']
    test_data.update({'t0': test_data['t']})
    idx0 = np.nonzero(np.equal(test_data.pop('t0'), 0))[1]

    print(len(idx0))
    X0 = np.zeros((len(idx0), 12))
    for n in range(1, len(idx0) + 1):
        X0[n - 1, :] = X[:, idx0[n - 1]]

    X0 = X0.T

    N = 201
    Time = np.linspace(0, 4, num=N)
    dt = Time[1] - Time[0]
    Time = np.flip(Time)  # invert time to fit for network input setting

    dx1 = np.zeros((len(idx0), Time.shape[0]))
    dy1 = np.zeros((len(idx0), Time.shape[0]))
    dz1 = np.zeros((len(idx0), Time.shape[0]))
    vx1 = np.zeros((len(idx0), Time.shape[0]))
    vy1 = np.zeros((len(idx0), Time.shape[0]))
    vz1 = np.zeros((len(idx0), Time.shape[0]))
    theta1 = np.zeros((len(idx0), Time.shape[0]))
    phi1 = np.zeros((len(idx0), Time.shape[0]))
    thrust1 = np.zeros((len(idx0), Time.shape[0]))
    dx2 = np.zeros((len(idx0), Time.shape[0]))
    dy2 = np.zeros((len(idx0), Time.shape[0]))
    dz2 = np.zeros((len(idx0), Time.shape[0]))
    vx2 = np.zeros((len(idx0), Time.shape[0]))
    vy2 = np.zeros((len(idx0), Time.shape[0]))
    vz2 = np.zeros((len(idx0), Time.shape[0]))
    theta2 = np.zeros((len(idx0), Time.shape[0]))
    phi2 = np.zeros((len(idx0), Time.shape[0]))
    thrust2 = np.zeros((len(idx0), Time.shape[0]))

    for n in range(len(idx0)):
        dx1[n][0] = X0[0, n]
        dy1[n][0] = X0[1, n]
        dz1[n][0] = X0[2, n]
        vx1[n][0] = X0[3, n]
        vy1[n][0] = X0[4, n]
        vz1[n][0] = X0[5, n]
        dx2[n][0] = X0[6, n]
        dy2[n][0] = X0[7, n]
        dz2[n][0] = X0[8, n]
        vx2[n][0] = X0[9, n]
        vy2[n][0] = X0[10, n]
        vz2[n][0] = X0[11, n]

    start_time = time.time()

    # closed-loop trajectory generation
    for i in range(X0.shape[1]):
        for j in range(1, Time.shape[0] + 1):
            X_nn = np.array([[dx1[i][j - 1]],
                             [dy1[i][j - 1]],
                             [dz1[i][j - 1]],
                             [vx1[i][j - 1]],
                             [vy1[i][j - 1]],
                             [vz1[i][j - 1]],
                             [dx2[i][j - 1]],
                             [dy2[i][j - 1]],
                             [dz2[i][j - 1]],
                             [vx2[i][j - 1]],
                             [vy2[i][j - 1]],
                             [vz2[i][j - 1]]])
            t_nn = np.array([[Time[j - 1]]])
            theta1[i][j - 1], theta2[i][j - 1], phi1[i][j - 1], phi2[i][j - 1], \
            thrust1[i][j - 1], thrust2[i][j - 1] = value_action(X_nn, t_nn, model, alpha)
            if j == Time.shape[0]:
                break
            else:
                dx1[i][j], dy1[i][j], dz1[i][j], vx1[i][j], vy1[i][j], vz1[i][j], \
                dx2[i][j], dy2[i][j], dz2[i][j], vx2[i][j], vy2[i][j], vz2[i][j] = \
                dynamic(X_nn, dt, (theta1[i][j - 1], theta2[i][j - 1], phi1[i][j - 1], phi2[i][j - 1], thrust1[i][j - 1], thrust2[i][j - 1]))
        print(i)

    print()
    time_spend = time.time() - start_time
    print('Total solution time: %1.1f' % (time_spend), 'sec')
    print()

    data = {'dx1': dx1,
            'dy1': dy1,
            'dz1': dz1,
            'vx1': vx1,
            'vy1': vy1,
            'vz1': vz1,
            'dx2': dx2,
            'dy2': dy2,
            'dz2': dz2,
            'vx2': vx2,
            'vy2': vy2,
            'vz2': vz2,
            'theta1': theta1,
            'theta2': theta2,
            'phi1': phi1,
            'phi2': phi2,
            'thrust1': thrust1,
            'thrust2': thrust2,
            't': t}

    final_data = discrete_data(data, dt, N)

    save_data = 1  # input('Save data? Enter 0 for no, 1 for yes:')
    if save_data:
        save_path = 'closed_loop/tanh/closedloop_traj_hybrid_drone_tanh.mat'
        # save_path = 'closed_loop/tanh/closedloop_traj_supervised_drone_tanh.mat'
        # save_path = 'closed_loop/tanh/closedloop_traj_pinn_drone_tanh.mat'
        # save_path = 'closed_loop/tanh/closedloop_traj_valuehardening_drone_tanh.mat'
        scio.savemat(save_path, final_data)

