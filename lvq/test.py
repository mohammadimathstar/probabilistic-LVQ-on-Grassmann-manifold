"""
Modified Script - Original Source: https://github.com/M-Nauta/ProtoTree/tree/main

Description:
This script is a modified version of the original script by M. Nauta.
Modifications have been made to suit specific requirements or preferences.
"""

from tqdm import tqdm
import torch.utils.data
from torch.utils.data import DataLoader

from lvq.model import Model
from util.glvq import metrics
from util.log import Log
from util.utils import smooth_labels

import numpy as np
import argparse


@torch.no_grad()
def eval(model: Model,
         test_loader: DataLoader,
         epoch: int,
         loss,
         device,
         args: argparse.Namespace,
         log: Log = None,
         log_prefix: str = 'log_eval_epochs',
         progress_prefix: str = 'Eval Epoch'
) -> dict :

    model = model.to(device)
    nclasses = args.nclasses

    # to store information about the procedure
    test_info = dict()

    model.eval()

    # to show the progress-bar
    train_iter = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc=progress_prefix + ' %s' % epoch,
        ncols=0
    )

    conf_mat = np.zeros((nclasses, nclasses), dtype=int)

    # training process (one epoch)
    for i, (xs, ys) in train_iter:
        soft_targets = smooth_labels(ys, nclasses, args.epsilon)
        
        xs, ys, soft_targets = xs.to(device), ys.to(device), soft_targets.to(device)

        # forward pass
        scores = model(xs)

        # predict labels
        yspred = model.prototype_layer.yprotos[scores.argmax(axis=1)]
        acc = torch.sum(torch.eq(yspred, ys)).item() / float(len(xs))
        cost = loss(scores, soft_targets)

        # compute the confusion matrix
        acc, cmat = metrics(ys, yspred, nclasses=nclasses)
        conf_mat += cmat

        train_iter.set_postfix_str(
            f"Batch [{i + 1}/{len(test_loader)}, Loss: {cost.item(): .3f}, Acc: {acc: .3f}"
        )

    test_info['confusion_matrix'] = conf_mat
    test_info['test_accuracy'] = np.diag(conf_mat).sum() / conf_mat.sum()

    log.log_message("\nEpoch %i - Test accuracy: %s" %(epoch, test_info['test_accuracy'] * 100))

    return test_info
