# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, utils, loss_functions, modules, modules_adaptive
import training_pinn, training_supervised, training_hybrid, training_valuehardening, training_epigraphical

from torch.utils.data import DataLoader
import configargparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
"""
learning rate:
1e-3 for epigraphical learning 
2e-5 for hybrid, supervised, pinn, value hardening
"""
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')

"""
training epoch --num_epochs:
200000 for hybrid 
100000 for supervised 
270000 for pinn
5400 for value hardening
350000 for epigraphical learning
"""
p.add_argument('--num_epochs', type=int, default=200000,
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
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--num_nl', type=int, default=64, required=False, help='Number of neurons per hidden layer.')

"""
training epoch ---pretrain_iters: 
100000 for hybrid 
100000 for supervised 
10000 for pinn
1000 for value hardening
50000 for epigraphical learning
"""
p.add_argument('--pretrain_iters', type=int, default=100000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculul training')

"""
training epoch --counter_end:
100000 for hybrid 
0 for supervised 
260000 for pinn
4400 for value hardening
300000 for epigraphical learning
"""
p.add_argument('--counter_end', type=int, default=100000, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=1000, required=False, help='Number of source samples at each time step')

p.add_argument('--collisionR', type=float, default=0.25, required=False, help='Collision radius between vehicles')
p.add_argument('--minWith', type=str, default='target', required=False, choices=['none', 'zero', 'target'], help='BRS vs BRT computation')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
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

Epigraphical = False
PINN = False
Supervised = False
Hybrid = True
Valuehardening = False

policy = ['a_a', 'a_na', 'na_a', 'na_na']
N_choice = 0

if Epigraphical == True:
    if N_choice == 0:
        Weight = (10, 1)
        Theta = (1, 1)
    if N_choice == 1:
        Weight = (10, 1)
        Theta = (1, 5)
    if N_choice == 2:
        Weight = (10, 1)
        Theta = (5, 1)
    if N_choice == 3:
        Weight = (10, 1)
        Theta = (5, 5)

    Epigraphical_dataset = dataio.IntersectionHJI_EL(N_choice, numpoints=100000, pretrain=opt.pretrain, tMin=opt.tMin,
                                                     tMax=opt.tMax, counter_start=opt.counter_start,
                                                     counter_end=opt.counter_end, pretrain_iters=opt.pretrain_iters, seed=opt.seed,
                                                     num_src_samples=opt.num_src_samples)

    Epigraphical_dataloader = DataLoader(Epigraphical_dataset, shuffle=True, batch_size=opt.batch_size,
                                         pin_memory=True, num_workers=0)

    model = modules_adaptive.SingleBVPNet(in_features=5, out_features=1, type=opt.model, mode=opt.mode,
                                          final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    loss_fn_epigraphical = loss_functions.initialize_intersection_HJI_EL(Epigraphical_dataset, Weight, N_choice)

    path = 'experiment_HJI_' + opt.model + '_el_' + str(policy[N_choice]) + '/'
    root_path = os.path.join(opt.logging_root, path)

    training_epigraphical.train(model=model, train_dataloader=Epigraphical_dataloader, epochs=opt.num_epochs,
                                lr=opt.lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                                model_dir=root_path, loss_fn=loss_fn_epigraphical, clip_grad=opt.clip_grad,
                                use_lbfgs=opt.use_lbfgs, validation_fn=None, start_epoch=opt.checkpoint_toload, N_choice=N_choice)

if PINN == True:
    if N_choice == 0:
        Weight = (10, 200)
        Theta = (1, 1)
    if N_choice == 1:
        Weight = (5, 100)
        Theta = (1, 5)
    if N_choice == 2:
        Weight = (5, 100)
        Theta = (5, 1)
    if N_choice == 3:
        Weight = (2.5, 200)
        Theta = (5, 5)

    pinn_dataset = dataio.IntersectionHJI_PINN(numpoints=61000, pretrain=opt.pretrain, tMin=opt.tMin,
                                               tMax=opt.tMax, counter_start=opt.counter_start,
                                               counter_end=opt.counter_end, pretrain_iters=opt.pretrain_iters,
                                               seed=opt.seed, num_src_samples=opt.num_src_samples)

    pinn_dataloader = DataLoader(pinn_dataset, shuffle=True, batch_size=opt.batch_size,
                                 pin_memory=True,  num_workers=0)

    model = modules.SingleBVPNet(in_features=5, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    loss_fn_pinn = loss_functions.initialize_intersection_HJI_pinn(pinn_dataset, Weight, Theta)

    path = 'experiment_HJI_' + opt.model + '_pinn_' + str(policy[N_choice]) + '/'
    root_path = os.path.join(opt.logging_root, path)

    training_pinn.train(model=model, train_dataloader=pinn_dataloader, epochs=opt.num_epochs,
                        lr=opt.lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                        model_dir=root_path, loss_fn=loss_fn_pinn, clip_grad=opt.clip_grad,
                        use_lbfgs=opt.use_lbfgs, validation_fn=None, start_epoch=opt.checkpoint_toload)


if Supervised == True:
    Hybrid_use = False
    if N_choice == 0:
        Weight = (40, 120)
        Theta = (1, 1)
        alpha = 1
    if N_choice == 1:
        Weight = (40, 120)
        Theta = (1, 5)
        alpha = 0.1
    if N_choice == 2:
        Weight = (40, 120)
        Theta = (5, 1)
        alpha = 0.1
    if N_choice == 3:
        Weight = (80, 120)
        Theta = (5, 5)
        alpha = 0.1

    supervised_dataset = dataio.IntersectionHJI_Supervised(Hybrid_use, N_choice, seed=opt.seed)

    supervised_dataloader = DataLoader(supervised_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                                       num_workers=0)

    model = modules.SingleBVPNet(in_features=5, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    loss_fn_supervised = loss_functions.initialize_intersection_HJI_supervised(supervised_dataset, Weight, Theta, alpha)

    path = 'experiment_HJI_' + opt.model + '_supervised_' + str(policy[N_choice]) + '/'
    root_path = os.path.join(opt.logging_root, path)

    training_supervised.train(model=model, train_dataloader=supervised_dataloader, epochs=opt.num_epochs, lr=opt.lr,
                              steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                              model_dir=root_path, loss_fn=loss_fn_supervised, clip_grad=opt.clip_grad,
                              use_lbfgs=opt.use_lbfgs, validation_fn=None, start_epoch=opt.checkpoint_toload,
                              pretrain=False, pretrain_iters=opt.pretrain_iters)


if Hybrid == True:
    Hybrid_use = True
    if N_choice == 0:
        Weight1 = (40, 120)
        Weight2 = (40, 120, 15, 200)
        Theta = (1, 1)
        alpha = 1
    if N_choice == 1:
        Weight1 = (40, 120)
        Weight2 = (40, 120, 5, 200)
        Theta = (1, 5)
        alpha = 0.1
    if N_choice == 2:
        Weight1 = (40, 120)
        Weight2 = (40, 120, 5, 200)
        Theta = (5, 1)
        alpha = 0.1
    if N_choice == 3:
        Weight1 = (80, 120)
        Weight2 = (80, 120, 2.5, 200)
        Theta = (5, 5)
        alpha = 0.1

    supervised_dataset = dataio.IntersectionHJI_Supervised(Hybrid_use, N_choice, seed=opt.seed)
    supervised_dataloader = DataLoader(supervised_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                                       num_workers=0)

    hybrid_dataset = dataio.IntersectionHJI_Hybrid(N_choice, numpoints=30000, tMin=opt.tMin, tMax=opt.tMax,
                                                   counter_start=opt.counter_start, counter_end=opt.counter_end,
                                                   seed=opt.seed, num_src_samples=opt.num_src_samples)
    hybrid_dataloader = DataLoader(hybrid_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    model = modules.SingleBVPNet(in_features=5, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    loss_fn_supervised = loss_functions.initialize_intersection_HJI_supervised(hybrid_dataset, Weight1, Theta, alpha)
    loss_fn_hybrid = loss_functions.initialize_intersection_HJI_hyrid(hybrid_dataset, Weight2, Theta, alpha)

    path = 'experiment_HJI_' + opt.model + '_hybrid_' + str(policy[N_choice]) + '/'
    root_path = os.path.join(opt.logging_root, path)

    training_hybrid.train(model=model, train_dataloader=hybrid_dataloader, train_dataloader_supervised=supervised_dataloader,
                          epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                          epochs_til_checkpoint=opt.epochs_til_ckpt, model_dir=root_path, loss_fn=loss_fn_hybrid,
                          loss_fn_supervised=loss_fn_supervised, clip_grad=opt.clip_grad, use_lbfgs=opt.use_lbfgs,
                          validation_fn=None, start_epoch=opt.checkpoint_toload, pretrain=True, pretrain_iters=opt.pretrain_iters)


if Valuehardening == True:
    if N_choice == 0:
        Weight = (10, 200)
        Theta = (1, 1)
    if N_choice == 1:
        Weight = (5, 100)
        Theta = (1, 5)
    if N_choice == 2:
        Weight = (5, 100)
        Theta = (5, 1)
    if N_choice == 3:
        Weight = (2.5, 200)
        Theta = (5, 5)

    valuehardening_dataset = dataio.IntersectionHJI_PINN(numpoints=61000, pretrain=opt.pretrain, tMin=opt.tMin,
                                                         tMax=opt.tMax, counter_start=opt.counter_start,
                                                         counter_end=opt.counter_end, pretrain_iters=opt.pretrain_iters,
                                                         seed=opt.seed, num_src_samples=opt.num_src_samples)

    valuehardening_dataloader = DataLoader(valuehardening_dataset, shuffle=True, batch_size=opt.batch_size,
                                           pin_memory=True,  num_workers=0)

    model = modules.SingleBVPNet(in_features=5, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    gamma = np.linspace(0.001, 5, 50)

    for i in range(len(gamma)):
        if i == 0:
            load_dir = None
            start_epoch = 0
        else:
            path = 'experiment_HJI_' + opt.model + '_valuehardening_' + str(policy[N_choice]) + \
                   '/experiment_grow_gamma_tanh_' + str(i-1) + '/'
            load_dir = os.path.join(opt.logging_root, path)
            start_epoch = opt.num_epochs - 1

        path = 'experiment_HJI_' + opt.model + '_valuehardening_' + str(policy[N_choice]) + \
               '/experiment_grow_gamma_tanh_' + str(i) + '/'

        root_path = os.path.join(opt.logging_root, path)

        loss_fn_valuehardening = loss_functions.initialize_intersection_HJI_valuehardening(valuehardening_dataset, gamma[i], Weight, Theta)

        print(f'\nTraining with gamma: {gamma[i]}\n')

        training_valuehardening.train(model=model, train_dataloader=valuehardening_dataloader, epochs=opt.num_epochs, lr=opt.lr,
                                      steps_til_summary=opt.steps_til_summary,
                                      epochs_til_checkpoint=opt.epochs_til_ckpt,
                                      model_dir=root_path, loss_fn=loss_fn_valuehardening, clip_grad=opt.clip_grad,
                                      use_lbfgs=opt.use_lbfgs, validation_fn=None, load_dir=load_dir,
                                      start_epoch=start_epoch)