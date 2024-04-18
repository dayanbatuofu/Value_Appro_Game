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
        d1 = np.array([2.0 * (X[0] - 15) / (105 - 15) - 1])
        v1 = np.array([2.0 * (X[1] - 15) / (32 - 15) - 1])
        d2 = np.array([2.0 * (X[2] - 15) / (105 - 15) - 1])
        v2 = np.array([2.0 * (X[3] - 15) / (32 - 15) - 1])

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
        z_star = z1 + y1.detach().cpu().numpy().squeeze(0)

        z1 = np.array([[b]])
        X1 = np.vstack((d1, v1, d2, v2, z1))
        X1 = torch.tensor(X1, dtype=torch.float32, requires_grad=True).T
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        coords_1 = torch.cat((t, X1), dim=1)
        coords = coords_1.unsqueeze(0)
        model_in = {'coords': coords.to(device)}
        model_output = model(model_in)
        yA = model_output['model_outA']
        yB = model_output['model_outB']
        y1_tmp = torch.max(yA, yB)

        if y1_tmp > 0:
            z_star = 400
        else:
            z_star = z_star

        return z_star

    def dichotomy2(a, b, threshold, X, t, model):
        d1 = np.array([2.0 * (X[0] - 15) / (105 - 15) - 1])
        v1 = np.array([2.0 * (X[1] - 15) / (32 - 15) - 1])
        d2 = np.array([2.0 * (X[2] - 15) / (105 - 15) - 1])
        v2 = np.array([2.0 * (X[3] - 15) / (32 - 15) - 1])

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
        z_star = z2 + y2.detach().cpu().numpy().squeeze(0)

        z2 = np.array([[b]])
        X2 = np.vstack((d2, v2, d1, v1, z2))
        X2 = torch.tensor(X2, dtype=torch.float32, requires_grad=True).T
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        coords_2 = torch.cat((t, X2), dim=1)
        coords = coords_2.unsqueeze(0)
        model_in = {'coords': coords.to(device)}
        model_output = model(model_in)
        yA = model_output['model_outA']
        yB = model_output['model_outB']
        y2_tmp = torch.max(yA, yB)

        if y2_tmp > 0:
            z_star = 400
        else:
            z_star = z_star

        return z_star

    z_min = -1.05e-4
    z_max = 300
    threshold = 0

    z1_0 = dichotomy1(z_min, z_max, threshold, X, t, model)
    z2_0 = dichotomy2(z_min, z_max, threshold, X, t, model)

    return z1_0, z2_0

def value_action(X, t, model):
    # normalize the state for agent 1, agent 2
    d1 = np.array([2.0 * (X[0] - 15) / (105 - 15) - 1])
    v1 = np.array([2.0 * (X[1] - 15) / (32 - 15) - 1])
    z1 = np.array([X[2]])
    d2 = np.array([2.0 * (X[3] - 15) / (105 - 15) - 1])
    v2 = np.array([2.0 * (X[4] - 15) / (32 - 15) - 1])
    z2 = np.array([X[5]])

    X1 = np.vstack((d1, v1, d2, v2, z1))
    X2 = np.vstack((d2, v2, d1, v1, z2))

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
    lam1_z = dvdx_1[:, 4:]  # lambda1_z

    # agent 2: partial gradient of V w.r.t. time and state
    dvdt_2 = dv_2[..., 0, 0].squeeze()
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1)

    # unnormalize the costate for agent 2, consider V = exp(u)
    lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
    lam21_2 = dvdx_2[:, 3:4] / ((32 - 15) / 2)  # lambda_21
    lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
    lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22
    lam2_z = dvdx_2[:, 4:]  # lambda2_z

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

    epsilon = (torch.tensor([4.5], dtype=torch.float32)).to(device)  # collision ratio

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

    ct_1 = (epsilon - (torch.abs((d11 - 36.5) + (d12 - 36.5)) + torch.abs((d11 - 36.5) - (d12 - 36.5)))).squeeze()  # 35
    ct_2 = (epsilon - (torch.abs((d21 - 36.5) + (d22 - 36.5)) + torch.abs((d21 - 36.5) - (d22 - 36.5)))).squeeze()  # 35

    # calculate hamiltonian, -H = (dV/dx)^T * f - (dV/dz)^T * L
    ham_1 = lam11_1.squeeze() * v11.squeeze() + lam11_2.squeeze() * u1.squeeze() + \
            lam12_1.squeeze() * v12.squeeze() + lam12_2.squeeze() * u2.squeeze() - lam1_z.squeeze() * (u1 ** 2).squeeze()
    ham_2 = lam21_1.squeeze() * v21.squeeze() + lam21_2.squeeze() * u1.squeeze() + \
            lam22_1.squeeze() * v22.squeeze() + lam22_2.squeeze() * u2.squeeze() - lam2_z.squeeze() * (u2 ** 2).squeeze()

    diff_constraint_hom_1 = torch.max(ct_1 - y1.squeeze(), -dvdt_1 + ham_1)
    diff_constraint_hom_2 = torch.max(ct_2 - y2.squeeze(), -dvdt_2 + ham_2)

    return ct_1-y1.squeeze(), ct_2-y2.squeeze(), -dvdt_1+ham_1, -dvdt_2+ham_2, diff_constraint_hom_1, diff_constraint_hom_2, \
           yA1, yB1, yA2, yB2

ckpt_path = './model/tanh/model_el_a_a_tanh.pth'
activation = 'tanh'

# Initialize and load the model
model = modules_adaptive.SingleBVPNet(in_features=5, out_features=1, type=activation, mode='mlp',
                                      final_layer_factor=1., hidden_features=64, num_hidden_layers=3)
model.to(device)
checkpoint = torch.load(ckpt_path)
try:
    model_weights = checkpoint['model']
except:
    model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

N = 101

x1_axis = np.zeros(N)
x2_axis = np.zeros(N)

for i in range(N):
    x1_axis[i] = 15 + 0.25 * i
    x2_axis[i] = 15 + 0.25 * i

x1, x2 = np.meshgrid(x1_axis, x2_axis)

start_time = time.time()
count = 0

Time = np.linspace(0, 3, num=6)
Time = np.flip(Time)

V1 = np.zeros((Time.shape[0], N, N))
V2 = np.zeros((Time.shape[0], N, N))
yA1 = np.zeros((Time.shape[0], N, N))
yB1 = np.zeros((Time.shape[0], N, N))
yA2 = np.zeros((Time.shape[0], N, N))
yB2 = np.zeros((Time.shape[0], N, N))
ct_1 = np.zeros((Time.shape[0], N, N))
ct_2 = np.zeros((Time.shape[0], N, N))
hji_1 = np.zeros((Time.shape[0], N, N))
hji_2 = np.zeros((Time.shape[0], N, N))
diff_hji1 = np.zeros((Time.shape[0], N, N))
diff_hji2 = np.zeros((Time.shape[0], N, N))

for k in range(Time.shape[0]):
    for i in range(N):
        for j in range(N):
            d1 = x1[i][j]
            v1 = 20.
            d2 = x2[i][j]
            v2 = 20.
            X_nn = np.vstack((d1, v1, d2, v2))
            t_nn = np.array([[Time[-k-1]]])
            y1, y2 = optimal_z_generation(X_nn, t_nn, model)
            V1[k][i][j] = y1
            V2[k][i][j] = y2
            X_nn = np.vstack((d1, v1, y1, d2, v2, y2))
            ct_1[k][i][j], ct_2[k][i][j], hji_1[k][i][j], hji_2[k][i][j], diff_hji1[k][i][j], diff_hji2[k][i][j], \
            yA1[k][i][j], yB1[k][i][j], yA2[k][i][j], yB2[k][i][j] = value_action(X_nn, t_nn, model)
            count += 1
            print(count)
    count = 0
    print('------------------' + str(k) + 'iteration------------------')

print()
time_spend = time.time() - start_time
print('Total solution time: %1.1f' % (time_spend), 'sec')
print()

data = {'d1': x1,
        'd2': x2,
        'V1': V1,
        'V2': V2,
        'yA1': yA1,
        'yA2': yA2,
        'yB1': yB1,
        'yB2': yB2,
        'ct_1': ct_1,
        'ct_2': ct_2,
        'hji_1': hji_1,
        'hji_2': hji_2,
        'diff_hji1': diff_hji1,
        'diff_hji2': diff_hji2}

save_data = input('Save data? Enter 0 for no, 1 for yes:')
if save_data:
    save_path = 'value/tanh/valuecontour_el_a_a_tanh.mat'
    scio.savemat(save_path, data)