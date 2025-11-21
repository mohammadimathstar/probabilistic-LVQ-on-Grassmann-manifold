import torch
import torch.nn as nn

import random, numpy as np

from util.log import Log
from util.args import get_args, save_args
from util.data import get_dataloaders
from util.net import get_network, freeze
from util.save import *
from util.analyse import analyse_output_shape
from util.optimizer_sgd import get_optimizer
from util.glvq import get_loss_fn
from util.load_model import load_grassmannlvq_model

from lvq.model import GrassmannLVQModel
from lvq.train import train_epoch
from lvq.test import eval
from lvq.measures.angle_measure import AngleMeasure


def check_device_consistency(model):
    devices = {p.device for p in model.parameters()}
    print("Model uses devices:", devices)


def main():
    # -------------------------------------------------------------------------
    # 1️⃣  Parse arguments & setup environment
    # -------------------------------------------------------------------------
    args = get_args()

    # Set random seeds for reproducibility
    # torch.manual_seed(args.seed)    
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    # Choose device
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')

    torch.backends.cudnn.benchmark = True  # ✅ faster training for fixed input size

    # -------------------------------------------------------------------------
    # 2️⃣  Logging setup
    # -------------------------------------------------------------------------
    log = Log(args.log_dir)
    print(f"Log dir: {args.log_dir}", flush=True)

    log.create_log('log_epoch_overview', 'epoch', 'test_acc', 'train_acc', 'train_loss_during_epoch')
    log.log_message(f"Device used: {device}")

    log_prefix = 'log_train_epochs'
    log_loss = f"{log_prefix}_losses"
    log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')

    
    # -------------------------------------------------------------------------
    # 3️⃣  Dataset & Dataloaders
    # -------------------------------------------------------------------------
    trainloader, testloader, classes, num_channels = get_dataloaders(args)
    args.num_channels = num_channels

    # -------------------------------------------------------------------------
    # 4️⃣  Network & Model Setup
    # -------------------------------------------------------------------------    
    model, _ = load_grassmannlvq_model(args=args, 
                                        device=device,                                        
                                        init_from_data=True if args.proto_init=="data" else False,
                                        trainloader=trainloader,)  # 👈 Use real data for prototype initialization

    print(f"Shape of prototypes: {model.prototype_layer.xprotos.shape}")
    print(f"prototype update method: {args.proto_opt}")

    # Loss function
    loss_fn = get_loss_fn(args)

    # -------------------------------------------------------------------------
    # 5️⃣  Optimizers
    # -------------------------------------------------------------------------
    optimizer_net, optimizer_proto, optimizer_rel, params_to_freeze, params_to_train = get_optimizer(model, args)

    # Save initial model state
    model.save_state(f"{log.checkpoint_dir}/model_init")

    # Save all arguments
    save_args(args, log.metadata_dir)

    analyse_output_shape(model, trainloader, log, device)

    # -------------------------------------------------------------------------
    # 6️⃣  Training & Evaluation
    # -------------------------------------------------------------------------
    best_train_acc = 0
    best_test_acc = 0

    check_device_consistency(model)


    for epoch in range(1, args.nepochs + 1):
        log.log_message(f"\nEpoch {epoch}")

        # Freeze / unfreeze if needed
        freeze(model, epoch, params_to_freeze, params_to_train, args, log)

        # ---- Training ----
        train_info = train_epoch(
            model=model,
            trainloader=trainloader,
            epoch=epoch,
            loss_fn=loss_fn,
            args=args,
            optimizer_net=optimizer_net,
            optimizer_protos=optimizer_proto,
            optimizer_rel=optimizer_rel,
            device=device,
            log=log,
            log_prefix=log_prefix,
            progress_prefix='Train Epoch',
        )


        # Save checkpoints
        save_model(model, epoch, log, args)
        best_train_acc = save_best_train_model(model, best_train_acc, train_info['train_accuracy'], log)

        # ---- Evaluation ----
        eval_info = eval(model, testloader, epoch, loss_fn, device, args, log)
        best_test_acc = save_best_test_model(model, best_test_acc, eval_info['test_accuracy'], log)

        # Log epoch results
        log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'],
                       train_info['train_accuracy'], train_info['loss'])

    # -------------------------------------------------------------------------
    # 7️⃣  Wrap Up
    # -------------------------------------------------------------------------
    log.log_message(
        f"\nTraining Finished.\nBest train acc: {best_train_acc:.4f}, best test acc: {best_test_acc:.4f}\n"
    )


if __name__ == "__main__":
    main()















# # This is a sample Python script.

# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import torch
# from torch.nn import KLDivLoss

# from util.log import Log
# from util.args import get_args, save_args
# from util.data import get_dataloaders
# from util.net import get_network, freeze
# from util.save import *
# from util.analyse import analyse_output_shape
# # from util.glvq import 
# from util.optimizer_sgd import get_optimizer

# from lvq.model import Model
# from lvq.train import train_epoch
# from lvq.test import eval
# from lvq.measures.angle_measure import AngleMeasure


# args = get_args()


# score_fn = AngleMeasure(beta=args.beta)
# loss = KLDivLoss(reduction="batchmean")

# # Create a logger
# log = Log(args.log_dir)
# print("Log dir: ", args.log_dir, flush=True)
# # Create a csv log for storing the test accuracy, mean train accuracy and mean loss for each epoch
# log.create_log('log_epoch_overview', 'epoch', 'test_acc', 'train_acc', 'train_loss_during_epoch')
# # Log the run arguments
# save_args(args, log.metadata_dir)

# if not args.disable_cuda and torch.cuda.is_available():
#     device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
# else:
#     device = torch.device('cpu')

# if torch.cuda.is_available():
#      device = torch.device('cuda:{}'.format(torch.cuda.current_device()))

# # Log which device was actually used
# log.log_message('Device used: '+str(device))

# # Create a log for logging the loss values
# log_prefix = 'log_train_epochs'
# log_loss = log_prefix+'_losses'
# log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')

# # Obtain the dataset and dataloaders
# # trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args)
# trainloader, testloader, classes, num_channels = get_dataloaders(args)


# # Create a convolutional network based on arguments and add 1x1 conv layer
# features_net, add_on_layers = get_network(num_channels, args)


# if args.num_features>0:
#     # if we want an add_on_layers
#     model = Model(
#         num_classes=len(classes),
#         feature_extractor=features_net,
#         score_fn=score_fn,
#         args=args,
#         add_on_layers=add_on_layers,
#         device=device,
#     )
# else:

#     # if we do not want an add_on_layers
#     model = Model(
#         num_classes=len(classes),
#         feature_extractor=features_net,
#         score_fn=score_fn,
#         args=args,
#         #add_on_layers=add_on_layers,
#         device=device,
#     )
# model = model.to(device=device)

# print(device)

# print(f"The shape of prototypes is: {model.prototype_layer.xprotos.shape}")

# # params_to_freeze, params_to_train = get_parameter_setting(model, args)
# optimizer_net, optimizer_proto, optimizer_rel, params_to_freeze, params_to_train = get_optimizer(model, args)
# # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_net, milestones=args.milestones, gamma=args.gamma)

# model.save(f"{log.checkpoint_dir}/model_init")

# analyse_output_shape(model, trainloader, log, device)


# # Initialize prototypes from one random image per class
# model.initialize_prototypes_from_data(trainloader, device)


# best_train_acc = 0
# best_test_acc = 0

# epoch = 1
# if epoch < args.nepochs + 1:
#     # Train the model
#     for epoch in range(epoch, args.nepochs + 1):
#         log.log_message("\nEpoch %i" % epoch)

#         # freeze part of network for some epochs if indicated in args
#         freeze(model, epoch, params_to_freeze, params_to_train, args, log)

#         # Train model
#         train_info = train_epoch(
#             model,
#             trainloader,
#             epoch,
#             loss,
#             args,
#             optimizer_net,
#             optimizer_proto,
#             optimizer_rel,
#             device,
#             log=log,
#             log_prefix=log_prefix,
#             progress_prefix='Train Epoch',
#         )

#         # save
#         save_model(model, epoch, log, args)

#         # complete the following
#         best_train_acc = save_best_train_model(model, best_train_acc, train_info['train_accuracy'], log)

#         eval_info = eval(model, testloader, epoch, loss, device, args, log)
#         original_test_acc = eval_info['test_accuracy']
#         best_test_acc = save_best_test_model(model,
#                                              best_test_acc,
#                                              eval_info['test_accuracy'],
#                                              log)
#         log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], train_info['train_accuracy'],
#                        train_info['loss'])


# else:
#     # model is loaded only for evaluation (not training)
#     eval_info = eval(model, testloader, epoch, device, log)
#     original_test_acc = eval_info['test_accuracy']
#     best_test_acc = save_best_test_model(model, best_test_acc, eval_info['test_accuracy'], log)
#     log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], "n.a.", "n.a.")


# log.log_message("Training Finished. Best training accuracy was %s, best test accuracy was %s\n"%(str(best_train_acc), str(best_test_acc)))


