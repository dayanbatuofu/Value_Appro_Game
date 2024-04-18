import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modules
import torch
import numpy as np
import matplotlib.pyplot as plt


def delta(x, a):
    return (1 / math.pi) * (a / (a ** 2 + x ** 2))


if __name__ == '__main__':

    logging_root = './logs'

    # Path to model
    ckpt_path_ssl = '../experiment_scripts/logs/ssl/checkpoints/model_final.pth'
    ckpt_path_h = '../experiment_scripts/logs/hybrid/checkpoints/model_final.pth'
    ckpt_path_super = '../experiment_scripts/logs/supervised/checkpoints/model_final.pth'

    activation = 'tanh'

    # Initialize and load the model
    model_h = modules.SingleBVPNet(in_features=1, out_features=1, type=activation, mode='mlp',
                                   final_layer_factor=1., hidden_features=4, num_hidden_layers=1)

    model_sup = modules.SingleBVPNet(in_features=1, out_features=1, type=activation, mode='mlp',
                                     final_layer_factor=1., hidden_features=4, num_hidden_layers=1)

    model_ssl = modules.SingleBVPNet(in_features=1, out_features=1, type=activation, mode='mlp',
                                     final_layer_factor=1., hidden_features=4, num_hidden_layers=1)

    chk_h = torch.load(ckpt_path_h)
    chk_sup = torch.load(ckpt_path_super)
    chk_ssl = torch.load(ckpt_path_ssl)

    try:
        model_w_h = chk_h['model']
        model_w_sup = chk_sup['model']
        model_w_ssl = chk_ssl['model']
    except:
        model_w_h = chk_h
        model_w_sup = chk_sup
        model_w_ssl = chk_ssl

    model_h.load_state_dict(model_w_h)
    model_sup.load_state_dict(model_w_sup)
    model_ssl.load_state_dict(model_w_ssl)

    model_h.eval()
    model_sup.eval()
    model_ssl.eval()

    test_coords = torch.linspace(-1, 1, 1000).reshape((-1, 1))
    model_in = {'coords': test_coords.cpu()}

    y_h = model_h(model_in)
    y_sup = model_sup(model_in)
    y_ssl = model_ssl(model_in)

    value_h = y_h['model_out']
    value_sup = y_sup['model_out']
    value_ssl = y_ssl['model_out']

    v_h = np.zeros((1000, 1))
    v_sup = np.zeros((1000, 1))
    v_ssl = np.zeros((1000, 1))

    x_ssl = np.zeros((1000, 1))

    v_h = value_h.detach().cpu()
    v_sup = value_sup.detach().cpu()
    v_ssl = value_ssl.detach().cpu()

    x_ssl = test_coords.detach().cpu()

    import math

    font = {'family': 'Times New Roman', 'weight': 'heavy', 'size': 20}
    plt.rc('font', **font)

    alpha = 1e-6
    N = 10000
    C = -np.arctan(1 / alpha) * (1 / math.pi)
    X = np.linspace(-1, 1, num=N)
    V = np.zeros((1, N)).flatten()

    for i in range(len(X)):
        V[i] = np.arctan(X[i] / alpha) * (1 / math.pi) + C
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))

    axs.plot(X, V, c='black', label='True Value', alpha=1, linewidth=5)
    axs.plot(x_ssl, v_ssl, c='gray', label='SSL Value', alpha=1, linewidth=5, linestyle='--')
    axs.plot(x_ssl, v_sup, c='purple', label='SL Value', alpha=1, linewidth=5, linestyle='-')
    axs.plot(x_ssl, v_h, c='orange', label='Hybrid Value', alpha=1, linewidth=5, linestyle='-.')
    axs.plot([-0.75, 0.25], [-1, 0], 'ro', markersize=20)
    axs.legend(loc='lower right')
    axs.set_xlabel('X', fontweight='bold')
    axs.set_ylabel('Value', fontweight='bold')
    legend_prop = {'weight': 'bold'}
    axs.legend(loc='lower right', fontsize='small', prop=legend_prop, bbox_to_anchor=(0.515, 0., 0.5, 0.5),
               handlelength=0.55)
    fig.tight_layout()
    plt.show()
