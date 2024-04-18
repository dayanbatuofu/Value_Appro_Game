'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_dataloader, train_dataloader_supervised, epochs, lr, steps_til_summary,
          epochs_til_checkpoint, model_dir, loss_fn, loss_fn_supervised, summary_fn=None, val_dataloader=None,
          double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, validation_fn=None,
          start_epoch=0, pretrain=False, pretrain_iters=2000):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    # (a,na), (na, a), (na, na): 5000, (a,a): 4000
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.9, verbose=True, min_lr=1e-6,
                                                           patience=4000)

    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.9, verbose=True, min_lr=1e-6,
                                                               patience=5000)

    # Load the checkpoint if required
    if start_epoch > 0:
        # Load the model and start training from that point onwards
        model_path = os.path.join(model_dir, 'checkpoints', 'model_epoch_%04d.pth' % start_epoch)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.train()
        optim.load_state_dict(checkpoint['optimizer'])
        optim.param_groups[0]['lr'] = lr
        assert(start_epoch == checkpoint['epoch'])
    else:
        # Start training from scratch
        if os.path.exists(model_dir):
            val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
            if val == 'y':
                shutil.rmtree(model_dir)
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    pretrain_counter = 0
    with tqdm(total=len(train_dataloader) * epochs - start_epoch) as pbar:
        train_losses = []
        bcs_lossesA = []
        bcs_lossesB = []
        value_diffs = []
        constraint_diffs = []
        costate_diffs = []
        # action_diffs = []
        HJI_weight = []
        LR = []
        sampling_data = dict()
        sampling_data.update({'coords': 0,
                              'boundary_A': 0,
                              'boundary_B': 0})
        for epoch in range(start_epoch, epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                # Saving the optimizer state is important to produce consistent results
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict()}
                # torch.save(checkpoint,
                #            os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                if validation_fn is not None:
                    validation_fn(model, checkpoints_dir, epoch)

            if pretrain:
                # supervised learning
                for step, (model_input, gt) in enumerate(train_dataloader_supervised):
                    start_time = time.time()

                    model_input = {key: value.to(device) for key, value in model_input.items()}
                    gt = {key: value.to(device) for key, value in gt.items()}

                    if double_precision:
                        model_input = {key: value.double() for key, value in model_input.items()}
                        gt = {key: value.double() for key, value in gt.items()}

                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            model_output = model(model_input)
                            losses = loss_fn(model_output, gt)
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean()
                            train_loss.backward()
                            return train_loss
                        optim.step(closure)

                    model_output = model(model_input)

                    losses = loss_fn_supervised(model_output, gt)

                    # import ipdb; ipdb.set_trace()

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        if loss_name == 'weight':
                            if loss == 1:
                                hji_weight = 0
                            else:
                                hji_weight = loss
                            continue
                        if loss_name == 'dirichletA':
                            bcs_lossA = loss.mean()
                            single_loss = bcs_lossA
                        if loss_name == 'dirichletB':
                            bcs_lossB = loss.mean()
                            single_loss = bcs_lossB
                        if loss_name == 'values_difference':
                            value_diff = loss.mean()
                            single_loss = value_diff
                        if loss_name == 'constraint_difference':
                            constraint_diff = loss.mean()
                            single_loss = constraint_diff
                        if loss_name == 'costates_difference':
                            costate_diff = loss.mean()
                            single_loss = costate_diff
                        else:
                            single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                            single_loss *= loss_schedules[loss_name](total_steps)

                        writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    bcs_lossesA.append(bcs_lossA.item())
                    bcs_lossesB.append(bcs_lossB.item())
                    value_diffs.append(value_diff.item())
                    constraint_diffs.append(constraint_diff.item())
                    costate_diffs.append(costate_diff.item())
                    HJI_weight.append(hji_weight)
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        torch.save(model.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                        optim.step()

                    scheduler.step(train_loss)

                    lr_scheduler = optim.state_dict()['param_groups'][0]['lr']
                    LR.append(lr_scheduler)

            else:
                # epigraphical learning
                for step, (model_input, gt) in enumerate(train_dataloader):
                    start_time = time.time()

                    model_input = {key: value.to(device) for key, value in model_input.items()}
                    gt = {key: value.to(device) for key, value in gt.items()}

                    if double_precision:
                        model_input = {key: value.double() for key, value in model_input.items()}
                        gt = {key: value.double() for key, value in gt.items()}

                    model_output = model(model_input)

                    losses = loss_fn(model_output, gt)

                    # import ipdb; ipdb.set_trace()

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        if loss_name == 'weight':
                            if loss == 1:
                                hji_weight = 0
                            else:
                                hji_weight = loss
                            continue
                        if loss_name == 'dirichletA':
                            bcs_lossA = loss.mean()
                            single_loss = bcs_lossA
                        if loss_name == 'dirichletB':
                            bcs_lossB = loss.mean()
                            single_loss = bcs_lossB
                        if loss_name == 'values_difference':
                            value_diff = loss.mean()
                            single_loss = value_diff
                        if loss_name == 'constraint_difference':
                            constraint_diff = loss.mean()
                            single_loss = constraint_diff
                        if loss_name == 'costates_difference':
                            costate_diff = loss.mean()
                            single_loss = costate_diff
                        else:
                            single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps),
                                              total_steps)
                            single_loss *= loss_schedules[loss_name](total_steps)

                        writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    bcs_lossesA.append(bcs_lossA.item())
                    bcs_lossesB.append(bcs_lossB.item())
                    value_diffs.append(value_diff.item())
                    constraint_diffs.append(constraint_diff.item())
                    costate_diffs.append(costate_diff.item())
                    HJI_weight.append(hji_weight)
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        torch.save(model.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                        optim.step()

                        parm = {}
                        for name, parameters in model.named_parameters():
                            parm[name] = parameters

                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            model_output = model(model_input)
                            losses = loss_fn(model_output, gt)
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean()
                            train_loss.backward()
                            return train_loss

                        optim.step(closure)

            if pretrain:
                pretrain_counter += 1
            if pretrain and pretrain_counter == pretrain_iters:
                pretrain = False

            pbar.update(1)

            if not total_steps % steps_til_summary:
                tqdm.write("Epoch %d, Total loss %0.3f, bcs loss_A %0.2f, bcs loss_B %0.2f, value diff %0.2f, constraint diff %0.2f, "
                           "costate diff %0.2f, hji weight %0.2f, lr %0.6f"
                    % (epoch, train_loss, bcs_lossA, bcs_lossB, value_diff, constraint_diff, costate_diff, hji_weight, lr_scheduler))

                if val_dataloader is not None:
                    print("Running validation set...")
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for (model_input, gt) in val_dataloader:
                            model_output = model(model_input)
                            val_loss = loss_fn(model_output, gt)
                            val_losses.append(val_loss)

                        writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                    model.train()

            total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'bcs_lossesA_final.txt'),
                   np.array(bcs_lossesA))
        np.savetxt(os.path.join(checkpoints_dir, 'bcs_lossesB_final.txt'),
                   np.array(bcs_lossesB))
        np.savetxt(os.path.join(checkpoints_dir, 'value_diff_final.txt'),
                   np.array(value_diffs))
        np.savetxt(os.path.join(checkpoints_dir, 'constraint_diff_final.txt'),
                   np.array(constraint_diffs))
        np.savetxt(os.path.join(checkpoints_dir, 'costate_diff_final.txt'),
                   np.array(costate_diffs))
        np.savetxt(os.path.join(checkpoints_dir, 'hji_weight_final.txt'),
                   np.array(HJI_weight))
        np.savetxt(os.path.join(checkpoints_dir, 'learning rate.txt'),
                   np.array(LR))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
