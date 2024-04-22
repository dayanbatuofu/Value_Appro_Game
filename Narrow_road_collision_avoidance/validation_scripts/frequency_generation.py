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

def value_function(coords, alpha, model):
    coords = coords.unsqueeze(0)
    model_in = {'coords': coords.to(device)}
    model_output = model(model_in)

    x = model_output['model_in']
    y = model_output['model_out'] * alpha
    cut_index = x.shape[1] // 2
    y1 = model_output['model_out'][:, :cut_index].squeeze().detach().cpu().numpy()
    y2 = model_output['model_out'][:, cut_index:].squeeze().detach().cpu().numpy()

    y1 = y1 * alpha
    y2 = y2 * alpha

    return y1, y2

logging_root = './logs'

ckpt_path = './model/tanh/model_hybrid_narrowroad_tanh.pth'
# ckpt_path = './model/tanh/model_valuehardening_narrowroad_tanh.pth'
activation = 'tanh'

"""
value hardening uses alpha = 10
hybrid uses alpha = 1
"""

alpha = 1

# Initialize and load the model
model = modules.SingleBVPNet(in_features=9, out_features=1, type=activation, mode='mlp',
                             final_layer_factor=1., hidden_features=64, num_hidden_layers=3)
model.to(device)
checkpoint = torch.load(ckpt_path)
try:
    model_weights = checkpoint['model']
except:
    model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

numpoints = 101

dx1_axis = torch.linspace(-1, 1, steps=numpoints)
dx2_axis = torch.linspace(-1, 1, steps=numpoints)

dx1, dx2 = torch.meshgrid(dx1_axis, dx2_axis)

Time = torch.linspace(0, 3, steps=4)

V1 = np.zeros((Time.shape[0], numpoints, numpoints))
V2 = np.zeros((Time.shape[0], numpoints, numpoints))
fft_V1 = np.zeros((Time.shape[0], numpoints, numpoints))
fft_V2 = np.zeros((Time.shape[0], numpoints, numpoints))

dy1 = torch.tensor([0], dtype=torch.float32)
dy2 = torch.tensor([0], dtype=torch.float32)
theta1 = torch.tensor([0], dtype=torch.float32)
theta2 = torch.tensor([0], dtype=torch.float32)
v1 = torch.tensor([0], dtype=torch.float32)
v2 = torch.tensor([0], dtype=torch.float32)

start_time = time.time()
count = 0

for k in range(Time.shape[0]):
    for i in range(numpoints):
        for j in range(numpoints):
            coords1 = torch.stack((dx1[i][j].unsqueeze(0),
                                   dy1,
                                   theta1,
                                   v1,
                                   dx2[i][j].unsqueeze(0),
                                   dy2,
                                   theta2,
                                   v2), dim=1).to(device)
            coords2 = torch.stack((dx2[i][j].unsqueeze(0),
                                   dy2,
                                   theta2,
                                   v2,
                                   dx1[i][j].unsqueeze(0),
                                   dy1,
                                   theta1,
                                   v1), dim=1).to(device)
            t_nn = Time[k].clone().detach().reshape(1, 1).to(device)
            coords1 = torch.cat((t_nn, coords1), dim=1).to(device)
            coords2 = torch.cat((t_nn, coords2), dim=1).to(device)
            coords = torch.cat((coords1, coords2), dim=0)
            V1[k][i][j], V2[k][i][j] = value_function(coords, alpha, model)
            count += 1
            print(count)
    FV1 = np.fft.fftn(V1[k, :, :])
    FV2 = np.fft.fftn(V2[k, :, :])
    fft_V1[k, :, :] = np.log(np.abs(np.fft.fftshift(FV1)))
    fft_V2[k, :, :] = np.log(np.abs(np.fft.fftshift(FV2)))
    count = 0
    print('------------------' + str(k) + 'iteration------------------')

print()
time_spend = time.time() - start_time
print('Total solution time: %1.1f' % (time_spend), 'sec')
print()

data = {'V1': V1,
        'V2': V2,
        'fft_V1': fft_V1,
        'fft_V2': fft_V2,
        'd1': dx1.detach().numpy(),
        'd2': dx2.detach().numpy()}

save_data = 1  # input('Save data? Enter 0 for no, 1 for yes:')
if save_data:
    save_path = 'value/tanh/fft_value_hybrid_narrowroad_tanh.mat'
    # save_path = 'value/tanh/fft_value_valuehardening_narrowroad_tanh.mat'
    scio.savemat(save_path, data)

