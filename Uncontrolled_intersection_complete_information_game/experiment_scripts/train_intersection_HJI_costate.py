# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio_el_costate, dataio_hl_costate, utils, loss_functions_el_c, loss_functions_hl_c, modules
import training_hybrid_c, training_epigraphical_c

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
2e-5 for hybrid, epigraphical learning
"""
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')

"""
training epoch --num_epochs:
200000 for hybrid 
20000 for epigraphical learning (with or without costate supervisory data)
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
100000 for epigraphical learning (with or without costate supervisory data)
"""
p.add_argument('--pretrain_iters', type=int, default=100000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculul training')

"""
training epoch --counter_end:
100000 for hybrid 
100000 for epigraphical learning (with or without costate supervisory data)
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

Epigraphical_with_costate = True
Epigraphical_without_costate = False
Hybrid_with_costate = False
Hybrid_without_costate = False

if Epigraphical_with_costate == True:
    Weight = (50, 10)
    Costate = True

    supervised_dataset = dataio_el_costate.IntersectionHJI_Supervised(seed=opt.seed)
    supervised_dataloader = DataLoader(supervised_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                                       num_workers=0)

    Epigraphical_dataset = dataio_el_costate.IntersectionHJI_EL(numpoints=30000, pretrain=opt.pretrain, tMin=opt.tMin,
                                                                tMax=opt.tMax, counter_start=opt.counter_start,
                                                                counter_end=opt.counter_end, pretrain_iters=opt.pretrain_iters,
                                                                seed=opt.seed, num_src_samples=opt.num_src_samples)
    Epigraphical_dataloader = DataLoader(Epigraphical_dataset, shuffle=True, batch_size=opt.batch_size,
                                         pin_memory=True, num_workers=0)

    model = modules.SingleBVPNet(in_features=6, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    loss_fn_supervised = loss_functions_el_c.initialize_intersection_HJI_supervised(Epigraphical_dataset, Weight, Costate)
    loss_fn_epigraphical = loss_functions_el_c.initialize_intersection_HJI_EL(Epigraphical_dataset, Weight, Costate)

    path = 'experiment_HJI_' + opt.model + '_el_a_a_w_costate/'
    root_path = os.path.join(opt.logging_root, path)

    training_epigraphical_c.train(model=model, train_dataloader=Epigraphical_dataloader, train_dataloader_supervised=supervised_dataloader,
                                  epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                                  epochs_til_checkpoint=opt.epochs_til_ckpt, model_dir=root_path, loss_fn=loss_fn_epigraphical,
                                  loss_fn_supervised=loss_fn_supervised, clip_grad=opt.clip_grad, use_lbfgs=opt.use_lbfgs,
                                  validation_fn=None, start_epoch=opt.checkpoint_toload, pretrain=True, pretrain_iters=opt.pretrain_iters)


if Epigraphical_without_costate == True:
    Weight = (50, 10)
    Costate = False

    supervised_dataset = dataio_el_costate.IntersectionHJI_Supervised(seed=opt.seed)
    supervised_dataloader = DataLoader(supervised_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                                       num_workers=0)

    Epigraphical_dataset = dataio_el_costate.IntersectionHJI_EL(numpoints=30000, pretrain=opt.pretrain, tMin=opt.tMin,
                                                                tMax=opt.tMax, counter_start=opt.counter_start,
                                                                counter_end=opt.counter_end,
                                                                pretrain_iters=opt.pretrain_iters,
                                                                seed=opt.seed, num_src_samples=opt.num_src_samples)
    Epigraphical_dataloader = DataLoader(Epigraphical_dataset, shuffle=True, batch_size=opt.batch_size,
                                         pin_memory=True, num_workers=0)

    model = modules.SingleBVPNet(in_features=6, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    loss_fn_supervised = loss_functions_el_c.initialize_intersection_HJI_supervised(Epigraphical_dataset, Weight, Costate)
    loss_fn_epigraphical = loss_functions_el_c.initialize_intersection_HJI_EL(Epigraphical_dataset, Weight, Costate)

    path = 'experiment_HJI_' + opt.model + '_el_a_a_wo_costate/'
    root_path = os.path.join(opt.logging_root, path)

    training_epigraphical_c.train(model=model, train_dataloader=Epigraphical_dataloader, train_dataloader_supervised=supervised_dataloader,
                                  epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                                  epochs_til_checkpoint=opt.epochs_til_ckpt, model_dir=root_path, loss_fn=loss_fn_epigraphical,
                                  loss_fn_supervised=loss_fn_supervised, clip_grad=opt.clip_grad, use_lbfgs=opt.use_lbfgs,
                                  validation_fn=None, start_epoch=opt.checkpoint_toload, pretrain=True, pretrain_iters=opt.pretrain_iters)


if Hybrid_with_costate == True:
    Weight1 = (40, 120)
    Weight2 = (40, 120, 15, 200)
    Costate = True

    supervised_dataset = dataio_hl_costate.IntersectionHJI_Supervised(seed=opt.seed)
    supervised_dataloader = DataLoader(supervised_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                                       num_workers=0)

    hybrid_dataset = dataio_hl_costate.IntersectionHJI_Hybrid(numpoints=30000, tMin=opt.tMin, tMax=opt.tMax,
                                                              counter_start=opt.counter_start, counter_end=opt.counter_end,
                                                              seed=opt.seed, num_src_samples=opt.num_src_samples)
    hybrid_dataloader = DataLoader(hybrid_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    model = modules.SingleBVPNet(in_features=5, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    loss_fn_supervised = loss_functions_hl_c.initialize_intersection_HJI_supervised(hybrid_dataset, Weight1, Costate)
    loss_fn_hybrid = loss_functions_hl_c.initialize_intersection_HJI_hyrid(hybrid_dataset, Weight2, Costate)

    path = 'experiment_HJI_' + opt.model + '_hl_a_a_w_costate/'
    root_path = os.path.join(opt.logging_root, path)

    training_hybrid_c.train(model=model, train_dataloader=hybrid_dataloader, train_dataloader_supervised=supervised_dataloader,
                            epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                            epochs_til_checkpoint=opt.epochs_til_ckpt, model_dir=root_path, loss_fn=loss_fn_hybrid,
                            loss_fn_supervised=loss_fn_supervised, clip_grad=opt.clip_grad, use_lbfgs=opt.use_lbfgs,
                            validation_fn=None, start_epoch=opt.checkpoint_toload, pretrain=True, pretrain_iters=opt.pretrain_iters)


if Hybrid_with_costate == True:
    Weight1 = (40, 120)
    Weight2 = (40, 120, 15, 200)
    Costate = False

    supervised_dataset = dataio_hl_costate.IntersectionHJI_Supervised(seed=opt.seed)
    supervised_dataloader = DataLoader(supervised_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True,
                                       num_workers=0)

    hybrid_dataset = dataio_hl_costate.IntersectionHJI_Hybrid(numpoints=30000, tMin=opt.tMin, tMax=opt.tMax,
                                                              counter_start=opt.counter_start, counter_end=opt.counter_end,
                                                              seed=opt.seed, num_src_samples=opt.num_src_samples)
    hybrid_dataloader = DataLoader(hybrid_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    model = modules.SingleBVPNet(in_features=5, out_features=1, type=opt.model, mode=opt.mode,
                                 final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(device)

    loss_fn_supervised = loss_functions_hl_c.initialize_intersection_HJI_supervised(hybrid_dataset, Weight1, Costate)
    loss_fn_hybrid = loss_functions_hl_c.initialize_intersection_HJI_hyrid(hybrid_dataset, Weight2, Costate)

    path = 'experiment_HJI_' + opt.model + '_hl_a_a_w_costate/'
    root_path = os.path.join(opt.logging_root, path)

    training_hybrid_c.train(model=model, train_dataloader=hybrid_dataloader, train_dataloader_supervised=supervised_dataloader,
                            epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                            epochs_til_checkpoint=opt.epochs_til_ckpt, model_dir=root_path, loss_fn=loss_fn_hybrid,
                            loss_fn_supervised=loss_fn_supervised, clip_grad=opt.clip_grad, use_lbfgs=opt.use_lbfgs,
                            validation_fn=None, start_epoch=opt.checkpoint_toload, pretrain=True, pretrain_iters=opt.pretrain_iters)


