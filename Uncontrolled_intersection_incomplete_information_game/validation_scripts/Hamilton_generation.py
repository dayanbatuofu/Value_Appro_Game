# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules, diff_operators
import torch
import numpy as np

def Hamilton_function(X, t, U, alpha, model, theta):
    theta1, theta2 = theta
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
    model_in = {'coords': coords.cuda()}
    model_output = model(model_in)

    x = model_output['model_in']
    y = model_output['model_out']
    cut_index = x.shape[1] // 2

    y1 = model_output['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
    y2 = model_output['model_out'][:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
    jac, _ = diff_operators.jacobian(y, x)
    dv_1 = jac[:, :cut_index, :]
    dv_2 = jac[:, cut_index:, :]

    # agent 1: partial gradient of V w.r.t. state
    dvdt_1 = (dv_1[..., 0, 0] / y1).squeeze() * alpha
    dvdx_1 = dv_1[..., 0, 1:].squeeze().reshape(1, dv_1.shape[-1] - 1) * alpha

    # unnormalize the costate for agent 1
    lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
    lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
    lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
    lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

    # agent 2: partial gradient of V w.r.t. state
    dvdt_2 = (dv_2[..., 0, 0] / y2).squeeze() * alpha
    dvdx_2 = dv_2[..., 0, 1:].squeeze().reshape(1, dv_2.shape[-1] - 1) * alpha

    # unnormalize the costate for agent 2
    lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
    lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
    lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
    lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

    # calculate the collision area for aggressive-aggressive case
    R1 = torch.tensor([70.], dtype=torch.float32).cuda()  # road length for agent 1
    R2 = torch.tensor([70.], dtype=torch.float32).cuda()  # road length for agent 2
    W1 = torch.tensor([1.5], dtype=torch.float32).cuda()  # car width for agent 1
    W2 = torch.tensor([1.5], dtype=torch.float32).cuda()  # car width for agent 1
    L1 = torch.tensor([3.], dtype=torch.float32).cuda()  # car length for agent 1
    L2 = torch.tensor([3.], dtype=torch.float32).cuda()  # car length for agent 1
    theta1 = torch.tensor([theta1], dtype=torch.float32).cuda()  # behavior for agent 1
    theta2 = torch.tensor([theta2], dtype=torch.float32).cuda()  # behavior for agent 2
    beta = torch.tensor([10000.], dtype=torch.float32).cuda()  # collision ratio

    # H = lambda^T * f - L
    # Agent 1's action
    u1 = torch.tensor(U[-2:-1, :]).cuda()

    # Agent 2's action
    u2 = torch.tensor(U[-1:, :]).cuda()

    # unnormalize the state for agent 1
    d1 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (105 - 15) / 2 + 15
    v1 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

    # unnormalize the state for agent 2
    d2 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (105 - 15) / 2 + 15
    v2 = (model_output['model_in'][:, :cut_index, 4:] + 1) * (32 - 15) / 2 + 15

    # calculate the collision area lower and upper bounds
    x1_in = ((d1 - R1 / 2 + theta1 * W2 / 2) * 5).squeeze().reshape(-1, 1).cuda()
    x1_out = (-(d1 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).cuda()
    x2_in = ((d2 - R2 / 2 + theta2 * W1 / 2) * 5).squeeze().reshape(-1, 1).cuda()
    x2_out = (-(d2 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).cuda()

    sigmoid1 = torch.sigmoid(x1_in) * torch.sigmoid(x1_out)
    sigmoid2 = torch.sigmoid(x2_in) * torch.sigmoid(x2_out)
    loss_instant = beta * sigmoid1 * sigmoid2

    # calculate instantaneous loss
    loss_fun_1 = (u1 ** 2 + loss_instant).cuda()
    loss_fun_2 = (u2 ** 2 + loss_instant).cuda()

    # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
    ham_1 = lam11_1.squeeze() * v1.squeeze() + lam11_2.squeeze() * u1.squeeze() + \
            lam12_1.squeeze() * v2.squeeze() + lam12_2.squeeze() * u2.squeeze() - loss_fun_1.squeeze()
    ham_2 = lam21_1.squeeze() * v1.squeeze() + lam21_2.squeeze() * u1.squeeze() + \
            lam22_1.squeeze() * v2.squeeze() + lam22_2.squeeze() * u2.squeeze() - loss_fun_2.squeeze()

    hji1 = -dvdt_1 + ham_1
    hji2 = -dvdt_2 + ham_2

    return ham_1.detach().cpu().numpy(), ham_2.detach().cpu().numpy()

def get_Hamilton(X, t, U, theta):
    theta1, theta2 = theta
    if theta1 == 5 and theta2 == 5:
        alpha = 10
        ckpt_path = 'model/model_hybrid_na_na.pth'
        # ckpt_path = 'model/model_supervised_na_na.pth'
    elif theta1 == 1 and theta2 == 1:
        alpha = 1
        ckpt_path = 'model/model_hybrid_a_a.pth'
        # ckpt_path = 'model/model_supervised_a_a.pth'
    elif theta1 == 1 and theta2 == 5:
        alpha = 10
        ckpt_path = 'model/model_hybrid_a_na.pth'
        # ckpt_path = 'model/model_supervised_a_na.pth'
    elif theta1 == 5 and theta2 == 1:
        alpha = 10
        ckpt_path = 'model/model_hybrid_na_a.pth'
        # ckpt_path = 'model/model_supervised_na_a.pth'
    else:
        print("WARNING!!! INCORRECT THETA INPUT")

    activation = 'tanh'
    model = modules.SingleBVPNet(in_features=5, out_features=1, type=activation, mode='mlp',
                                 final_layer_factor=1., hidden_features=64, num_hidden_layers=3)
    model.cuda()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, ckpt_path)
    checkpoint = torch.load(model_path)
    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    model.eval()
    ham1, ham2 = Hamilton_function(X, t, U, alpha, model, theta)
    return ham1, ham2


