"""
Modified Script - Original Source: https://github.com/M-Nauta/ProtoTree/tree/main

Description:
This script is a modified version of the original script by M. Nauta.
Modifications have been made to suit specific requirements or preferences.

"""

from tqdm import tqdm
import argparse

import torch
import torch.utils.data
import torch.utils.data
from torch.utils.data import DataLoader

from lvq.model import GrassmannLVQModel
# from lvq.prototypes import rotate_prototypes
from util.grassmann import orthogonalize_batch
from util.log import Log
# from util.glvq import make_soft_labels
from util.glvq import smooth_labels
from util.optimizer_grassmann import GrassmannOptimizer

def train_epoch(
        model: GrassmannLVQModel,
        trainloader: DataLoader,
        epoch: int,
        loss_fn,
        args: argparse.Namespace,
        optimizer_net: torch.optim.Optimizer,
        optimizer_protos: torch.optim.Optimizer,
        optimizer_rel: torch.optim.Optimizer,
        device,
        log: Log = None,
        log_prefix: str = 'log_train_epochs',
        progress_prefix: str = 'Train Epoch'
) -> dict:

    model = model.to(device)
    nclasses = args.nclasses

    # to store information about the procedure
    train_info = dict()
    total_loss = 0
    total_acc = 0

    # create a log
    log_loss = f"{log_prefix}_losses"

    # to show the progress-bar
    train_iter = tqdm(
        enumerate(trainloader),
        total=len(trainloader),
        desc=progress_prefix + ' %s' % epoch,
        ncols=0
    )

    acc_mean = 0

    # training process (one epoch)
    for i, (xtrain, ytrain) in enumerate(trainloader):
        soft_targets = smooth_labels(ytrain, nclasses, args.epsilon)
                
        # Reset gradients
        optimizer_rel.zero_grad()
        optimizer_net.zero_grad()
        if isinstance(optimizer_protos, torch.optim.Optimizer):
            optimizer_protos.zero_grad()
        

        xtrain, ytrain, soft_targets = xtrain.to(device), ytrain.to(device), soft_targets.to(device)
        # xtrain, ytrain = xtrain.to(device), ytrain.to(device)
        scores = model(xtrain)

        log_probs = scores
        if args.loss_fn == 'ce':
            cost = loss_fn(log_probs, ytrain)
        else:
            cost = loss_fn(log_probs, soft_targets)
        
        cost.backward()

        optimizer_net.step()
        optimizer_rel.step()

        # Enforce constraints on relevances
        with torch.no_grad():
            LOW_BOUND_LAMBDA = 1e-5  # lower bound to avoid zeros
            rel = model.prototype_layer.relevances

            # Clamp to ensure nonnegativity
            rel.clamp_(min=LOW_BOUND_LAMBDA)

            # Normalize to sum to 1
            rel /= rel.sum()


        # ==========================
        # 3️⃣ Prototypes
        # ==========================
        if isinstance(optimizer_protos, GrassmannOptimizer):
            with torch.no_grad():
                W = model.prototype_layer.xprotos
                G_euclid = W.grad
                W_new = optimizer_protos.step(W, G_euclid)
                model.prototype_layer.xprotos.copy_(W_new)
                W.grad.zero_()
        else:
            # If using standard Euclidean optimizer (SGD)
            optimizer_protos.step()
            # Optional: enforce orthonormality after SGD
            with torch.no_grad():
                model.prototype_layer.xprotos.copy_(
                    orthogonalize_batch(model.prototype_layer.xprotos)
                )  
        # optimizer_protos.step()
        # optimizer_rel.step()
        # optimizer_net.step()
        # with torch.no_grad():
        #     model.prototype_layer.xprotos.copy_(orthogonalize_batch(model.prototype_layer.xprotos))
            

        ##### second way: manual update ##############
        # with torch.no_grad():
        #     protos_updates = model.prototype_layer.xprotos - args.lr_protos * model.prototype_layer.xprotos.grad
        #     rel_updates = model.prototype_layer.relevances - args.lr_rel * model.prototype_layer.relevances.grad
        #     model.prototype_layer.xprotos.copy_(orthogonalize_batch(protos_updates))
        #     model.prototype_layer.relevances.copy_(rel_updates)         
        
        
        # with torch.no_grad():
        #     LOW_BOUND_LAMBDA = 0.0001
        #     model.prototype_layer.relevances[0, torch.argwhere(model.prototype_layer.relevances < LOW_BOUND_LAMBDA)[:, 1]] = LOW_BOUND_LAMBDA
        #     model.prototype_layer.relevances[:] = model.prototype_layer.relevances[
        #                                           :] / model.prototype_layer.relevances.sum()

        # compute the accuracy
        yspred = model.prototype_layer.yprotos[scores.argmax(axis=1)]
        acc = torch.sum(torch.eq(yspred, ytrain)).item() / float(len(xtrain))

        train_iter.set_postfix_str(
            f"Batch [{i + 1}/{len(trainloader)}, Loss: {cost.sum().item(): .3f}, Acc: {acc * 100: .2f}"
        )
        acc_mean += acc

        # update the total metrics
        total_acc += acc
        total_loss += torch.sum(cost).item()

        # write a log
        if log is not None:
            log.log_values(log_loss, epoch, i + 1, torch.sum(cost).item(), acc)

    print(model.prototype_layer.relevances)

    train_info['loss'] = total_loss / float(i + 1)
    train_info['train_accuracy'] = total_acc / float(i + 1)

    return train_info

