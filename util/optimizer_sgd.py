import argparse

import torch.optim

import argparse
import torch
from util.optimizer_grassmann import GrassmannOptimizer  # The PyTorch version we built above


def get_optimizer(model, args: argparse.Namespace):
    """
    Constructs the optimizers for different parameter groups:
      - net_paramlist: feature extractor (Adam)
      - rel_paramlist: relevances (Adam)
      - proto_paramlist: prototypes (Grassmann or SGD)
    """

    # =====================
    # 1. Split model params
    # =====================
    params_to_freeze = []
    params_to_train = []

    if ('resnet50_inat' in args.net) or ('convnext_tiny_13' in args.net) or ('resnet50' in args.net):
        for name, param in model.feature_extractor.named_parameters():
            if 'layer4.2' not in name:
                params_to_freeze.append(param)
            else:
                params_to_train.append(param)

        # Group parameters for standard optimizers
        net_paramlist = [
            {
                "params": params_to_freeze,
                "lr": args.lr_net,
                "weight_decay": args.weight_decay,
            },
            {
                "params": params_to_train,
                "lr": args.lr_block,
                "weight_decay": args.weight_decay,
            },
        ]

        rel_paramlist = [
            {
                "params": model.prototype_layer.relevances,
                "lr": args.lr_rel,
                "weight_decay": args.weight_decay,
            }
        ]

        # Prototypes are handled separately (manifold optimizer)
        proto_params = model.prototype_layer.xprotos

    else:
        # fallback for other architectures
        all_params = list(model._net.parameters())
        net_paramlist = [{"params": all_params, "lr": args.lr_net, "weight_decay": args.weight_decay}]
        rel_paramlist = [
            {"params": model.prototype_layer.relevances, "lr": args.lr_rel, "weight_decay": args.weight_decay}
        ]
        proto_params = model.prototype_layer.xprotos

    # ====================================
    # 2. Construct standard torch optimizers
    # ====================================
    opt_net = torch.optim.Adam(net_paramlist)
    opt_rel = torch.optim.Adam(rel_paramlist)
    # opt_rel = torch.optim.SGD(rel_paramlist)

    # ====================================
    # 3. Construct Grassmann optimizer
    # ====================================
    # Choose between Grassmann or Euclidean prototype update
    if getattr(args, "proto_opt", "grassmann").lower() in ["grassmann", "exp", "qr"]:
        opt_proto = GrassmannOptimizer(
            lr=args.lr_protos,
            method=getattr(args, "proto_opt", "eucl"),  # exp or qr
        )
    else:
        # fallback to standard SGD if you want plain Euclidean update
        opt_proto = torch.optim.SGD([{"params": proto_params, "lr": args.lr_protos}])

    # Return all components
    return opt_net, opt_proto, opt_rel, params_to_freeze, params_to_train


# def get_optimizer(model, args: argparse.Namespace) -> tuple:
#     """
#     Construct the optimizer as dictated by the parsed arguments
#     :param model: The model that should be optimized
#     :param args: Parsed arguments containing hyperparameters. Optimizer specific arguments (such as
#     learning rate) can be passed this way as well
#     :return: the optimizer corresponding to the parsed arguments, parameter set that can be frozen,
#     and parameter set of the net that will be trained
#     """

#     #create parameter groups
#     params_to_freeze = []
#     params_to_train = []

#     # dist_params = []
#     # for name,param in model.named_parameters():
#     #     if 'dist_params' in name:
#     #         dist_params.append(param)

#     # set up optimizer
#     if ('resnet50_inat' in args.net) or ('convnext_tiny_13' in args.net) or ('resnet50' in args.net):  #to reproduce experimental results
#         # freeze resnet50 except last convolutional layer
#         for name, param in model.feature_extractor.named_parameters():
#             if 'layer4.2' not in name:
#                 params_to_freeze.append(param)
#             else:
#                 params_to_train.append(param)

#         net_paramlist = [
#             {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay, "momentum": args.momentum},
#             {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay,"momentum": args.momentum},
#             # {"params": model._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay,"momentum": args.momentum},
#         ]
#         proto_paramlist = [
#             {"params": model.prototype_layer.xprotos,
#              "lr": args.lr_protos,
#              "weight_decay_rate": 0,
#              "momentum": 0
#              },
#         ]
#         rel_paramlist = [
#             {"params": model.prototype_layer.relevances, "lr": args.lr_rel, "weight_decay_rate": args.weight_decay, "momentum": args.momentum}
#         ]

#     # else: #other network architectures
#     #     for name, param in model._net.named_parameters():
#     #         params_to_freeze.append(param)
#     #     paramlist = [
#     #         {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
#     #         # {"params": model._add_on.parameters(), "lr": args.lr_block}, #"weight_decay_rate": args.weight_decay},
#     #         {"params": model.prototype_layer.parameters(), "lr": args.lr_protos, "weight_decay_rate": 0},
#     #         {"params": model.relevances.parameters(), "lr": args.lr_rel, "weight_decay_rate": 0}
#     #     ]


#     return torch.optim.Adam(net_paramlist), torch.optim.SGD(proto_paramlist), torch.optim.Adam(
#         rel_paramlist), params_to_freeze, params_to_train
#     # return torch.optim.Adam(net_paramlist), torch.optim.SGD(proto_paramlist), torch.optim.SGD(
#     #     rel_paramlist), params_to_freeze, params_to_train