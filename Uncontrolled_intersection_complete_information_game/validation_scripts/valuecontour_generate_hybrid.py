import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators
import time
import torch
import numpy as np
import scipy.io as scio

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def value_function(X, t, model, alpha):
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
    y = model_output['model_out'] * alpha
    cut_index = x.shape[1] // 2

    y1 = y[:, :cut_index, :]
    y2 = y[:, cut_index:, :]

    # hybrid learning
    if y1 <= -80:  # 80
        y1_contour = 100
    else:
        y1_contour = 10
    if y2 <= -80:  # 80
        y2_contour = 100
    else:
        y2_contour = 10

    return y1, y2, y1_contour, y2_contour

logging_root = './logs'

ckpt_path = './model/tanh/model_hybrid_a_a_tanh.pth'
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

N_choice = 0

"""
self-supervised and value hardening uses alpha = 10 for (a,a), (a,na), (na,a), (na,na)
hybrid and supervised uses alpha = 1 for (a,a) and alpha = 10 for (a,na), (na,a), (na,na)
"""
alpha = 1

numpoints = 101

x1_axis = np.zeros(numpoints)
x2_axis = np.zeros(numpoints)

for i in range(numpoints):
    x1_axis[i] = 15 + 0.25 * i
    x2_axis[i] = 15 + 0.25 * i

x1, x2 = np.meshgrid(x1_axis, x2_axis)

start_time = time.time()
count = 0

Time = np.linspace(0, 3, num=6)

V1 = np.zeros((Time.shape[0], numpoints, numpoints))
V2 = np.zeros((Time.shape[0], numpoints, numpoints))
V1_class = np.zeros((Time.shape[0], numpoints, numpoints))
V2_class = np.zeros((Time.shape[0], numpoints, numpoints))

for k in range(Time.shape[0]):
    for i in range(numpoints):
        for j in range(numpoints):
            d1 = x1[i][j]
            v1 = 20.
            d2 = x2[i][j]
            v2 = 20.
            X_nn = np.vstack((d1, v1, d2, v2))
            t_nn = np.array([[Time[-k-1]]])
            V1[k][i][j], V2[k][i][j], V1_class[k][i][j], V2_class[k][i][j] = value_function(X_nn, t_nn, model, alpha)
            count += 1
        print(count)

print()
time_spend = time.time() - start_time
print('Total solution time: %1.1f' % (time_spend), 'sec')
print()

data = {'V1_real': V1,
        'V2_real': V2,
        'V1': V1_class,
        'V2': V2_class,
        'd1': x1,
        'd2': x2}

save_data = 1  # input('Save data? Enter 0 for no, 1 for yes:')
if save_data:
    save_path = 'value/tanh/valuecontour_hl_a_a_tanh.mat'
    scio.savemat(save_path, data)

