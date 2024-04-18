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
    ssl0 = '../experiment_scripts/logs/value_hardening/alpha_0/checkpoints/model_final.pth'  # initial
    ssl1 ='../experiment_scripts/logs/value_hardening/alpha_39/checkpoints/model_final.pth'  # intermediate
    ssl2 ='../experiment_scripts/logs/value_hardening/alpha_79/checkpoints/model_final.pth'  # final

    activation = 'tanh'

    # Initialize and load the model
    model_ssl0 = modules.SingleBVPNet(in_features=1, out_features=1, type=activation, mode='mlp',
                                   final_layer_factor=1., hidden_features=6, num_hidden_layers=1)

    model_ssl1 = modules.SingleBVPNet(in_features=1, out_features=1, type=activation, mode='mlp',
                                     final_layer_factor=1., hidden_features=6, num_hidden_layers=1)

    model_ssl2 = modules.SingleBVPNet(in_features=1, out_features=1, type=activation, mode='mlp',
                                     final_layer_factor=1., hidden_features=6, num_hidden_layers=1)

    chk_ssl0 = torch.load(ssl0)
    chk_ssl1 = torch.load(ssl1)
    chk_ssl2 = torch.load(ssl2)
    try:
        # model_weights = checkpoint['model']
        # model_w_sup = chk_sup['model']
        model_w_ssl0 = chk_ssl0['model']
        model_w_ssl1 = chk_ssl1['model']
        model_w_ssl2 = chk_ssl2['model']
    except:
        model_w_ssl0 = chk_ssl0
        model_w_ssl1 = chk_ssl1
        model_w_ssl2 = chk_ssl2

    model_ssl0.load_state_dict(model_w_ssl0)
    model_ssl1.load_state_dict(model_w_ssl1)
    model_ssl2.load_state_dict(model_w_ssl2)

    model_ssl0.eval()
    model_ssl1.eval()
    model_ssl2.eval()

    test_coords = torch.linspace(-1, 1, 1000).reshape((-1, 1))
    model_in = {'coords': test_coords.cpu()}

    y_ssl0 = model_ssl0(model_in)
    y_ssl1 = model_ssl1(model_in)
    y_ssl2 = model_ssl2(model_in)

    value_ssl0 = y_ssl0['model_out']
    value_ssl1 = y_ssl1['model_out']
    value_ssl2 = y_ssl2['model_out']

    v_ssl0 = np.zeros((1000, 1))
    v_ssl1 = np.zeros((1000, 1))
    v_ssl = np.zeros((1000, 1))

    x_ssl = np.zeros((1000, 1))

    v_ssl0 = value_ssl0.detach().cpu()
    v_ssl1 = value_ssl1.detach().cpu()
    v_ssl2 = value_ssl2.detach().cpu()
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

    axs.plot(X, V, c='black', label='True Value', alpha=0.8, linewidth=5)
    axs.plot(x_ssl, v_ssl2, c='purple', label='Final Value', alpha=1, linewidth=5, linestyle='-')
    axs.plot(x_ssl, v_ssl1, c='purple', label='Intermediate Value', alpha=0.45, linewidth=5, linestyle='-')
    axs.plot(x_ssl, v_ssl0, c='purple', label='Initial Value', alpha=0.2, linewidth=5, linestyle='-')

    axs.legend(loc='lower right')
    axs.set_xlabel('X', fontweight='bold')
    axs.set_ylabel('Value', fontweight='bold')
    legend_prop = {'weight': 'bold'}
    axs.legend(loc='lower right', fontsize='small', prop=legend_prop, bbox_to_anchor=(0.515, 0., 0.5, 0.5),
               handlelength=0.55)
    fig.tight_layout()
    plt.show()
