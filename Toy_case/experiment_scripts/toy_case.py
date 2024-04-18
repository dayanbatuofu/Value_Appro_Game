import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataio, loss_functions, modules
import training_selfsupervised, training_supervised, training_hybrid
from torch.utils.data import DataLoader
import configargparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs/tests', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-3, help='learning rate. default=2e-5')

"""
training epoch --num_epochs:
20000 for hybrid 
1000 for supervised 
10000 for self-supervised
3000 for value hardening
"""
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='tanh', required=False, choices=['tanh', 'relu', 'sine', 'gelu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=3, required=False, help='End time of the simulation')
p.add_argument('--num_hl', type=int, default=1, required=False, help='The number of hidden layers')
p.add_argument('--num_nl', type=int, default=4, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')

"""
training epoch ---pretrain_iters: 
1000 for hybrid
"""
p.add_argument('--pretrain_iters', type=int, default=1000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculul training')

"""
training epoch --counter_end:
1000 for hybrid 
"""
p.add_argument('--counter_end', type=int, default=1000, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--pretrain', action='store_true', default=True, required=False, help='Pretrain dirichlet conditions')

p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
opt = p.parse_args()

# Set the source coordinates for the target set and the obstacle sets

source_coords = [0., 0., 0., 0.]
if opt.counter_start == -1:
  opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
  opt.counter_end = opt.num_epochs

# set what to evaluate to True
Selfsupervised = True
Supervised = False
Hybrid = False
Valuehardening = False


if Selfsupervised == True:

    selfsupervised_dataset = dataio.SSL_1D(numpoints=2, seed=opt.seed)

    selfsupervised_dataloader = DataLoader(selfsupervised_dataset, shuffle=True, batch_size=opt.batch_size,
                                           pin_memory=True,  num_workers=0)

    model = modules.SingleBVPNet(in_features=1, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    loss_fn_selfsupervised = loss_functions.SSL_1D(selfsupervised_dataset)

    path = 'ssl/'
    root_path = os.path.join(opt.logging_root, path)

    training_selfsupervised.train(model=model, train_dataloader=selfsupervised_dataloader, epochs=opt.num_epochs,
                                  lr=opt.lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                                  model_dir=root_path, loss_fn=loss_fn_selfsupervised, clip_grad=opt.clip_grad,
                                  use_lbfgs=opt.use_lbfgs, validation_fn=None, start_epoch=opt.checkpoint_toload)

if Supervised == True:
    supervised_dataset = dataio.Sup_1D(seed=opt.seed)
    supervised_dataloader = DataLoader(supervised_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                                       num_workers=0)
    model = modules.SingleBVPNet(in_features=1, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    loss_fn_supervised = loss_functions.Sup_1D(supervised_dataset)

    path = 'supervised'
    root_path = os.path.join(opt.logging_root, path)

    training_supervised.train(model=model, train_dataloader=supervised_dataloader,
                   epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                   epochs_til_checkpoint=opt.epochs_til_ckpt, model_dir=root_path,
                   loss_fn=loss_fn_supervised, clip_grad=opt.clip_grad,
                   use_lbfgs=opt.use_lbfgs, validation_fn=None, start_epoch=opt.checkpoint_toload,
                   pretrain=False, pretrain_iters=opt.num_epochs)

if Hybrid == True:
    supervised_dataset = dataio.Sup_1D(seed=opt.seed)
    supervised_dataloader = DataLoader(supervised_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                                       num_workers=0)

    hybrid_dataset = dataio.Hybrid_1D(numpoints=2, seed=opt.seed)

    hybrid_dataloader = DataLoader(hybrid_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                                   num_workers=0)

    model = modules.SingleBVPNet(in_features=1, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    loss_fn_supervised = loss_functions.Sup_1D(supervised_dataset)
    loss_fn_hybrid = loss_functions.Hybrid_1D(hybrid_dataset)

    path = 'hybrid'
    root_path = os.path.join(opt.logging_root, path)

    training_hybrid.train(model=model, train_dataloader=hybrid_dataloader, train_dataloader_supervised=supervised_dataloader,
                   epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                   epochs_til_checkpoint=opt.epochs_til_ckpt, model_dir=root_path, loss_fn=loss_fn_hybrid,
                   loss_fn_supervised=loss_fn_supervised, clip_grad=opt.clip_grad, use_lbfgs=opt.use_lbfgs,
                   validation_fn=None, start_epoch=opt.checkpoint_toload, pretrain=True, pretrain_iters=opt.pretrain_iters)


if Valuehardening == True:

    # value hardening network hyper param
    opt.num_hl = 1
    opt.num_nl = 6

    model = modules.SingleBVPNet(in_features=1, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    # alpha for the delta function approximation
    alpha = np.linspace(5e-1, 1e-6, 80)

    for i in range(len(alpha)):
        if i == 0:
            load_dir = None
            start_epoch = 0
        else:
            load_dir = os.path.join(opt.logging_root,
                                    f'value_hardening/alpha_{i - 1}/')
            start_epoch = opt.num_epochs - 1

        root_path = os.path.join(opt.logging_root, f'value_hardening/alpha_{i}/')

        print(f'\n Training with alpha: {alpha[i]}\n')

        ssl_dataset = dataio.SSL_1D_vh(numpoints=300, alpha=alpha[i], seed=opt.seed)

        ssl_dataloader = DataLoader(ssl_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

        loss_fn = loss_functions.SSL_1D_vh(ssl_dataset, alpha[i])

        training_selfsupervised.train(model=model, train_dataloader=ssl_dataloader, epochs=opt.num_epochs, lr=opt.lr,
                                      steps_til_summary=opt.steps_til_summary,
                                      epochs_til_checkpoint=opt.epochs_til_ckpt,
                                      model_dir=root_path, loss_fn=loss_fn, clip_grad=opt.clip_grad,
                                      use_lbfgs=opt.use_lbfgs, validation_fn=None, load_dir=load_dir,
                                      start_epoch=start_epoch)
