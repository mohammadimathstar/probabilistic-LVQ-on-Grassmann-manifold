import argparse

import torch.optim

# def get_optimizer(model, args: argparse.Namespace) -> torch.optim.Optimizer:
#
#     params_prototypes = []
#     params_relevances = []
#
#     for name, param in model.named_parameters():
#         if 'xprotos' in name:
#             params_prototypes.append(param)
#         elif 'rel' in name:
#             params_relevances.append(param)
#         else:
#             print(f"There are some parameter not being prototypes and relevances with name: {name}")
#
#     proto_param_list = [
#         {'params': params_prototypes, 'lr': args.lr_protos},
#     ]
#     rel_param_list = [
#         {'params': params_relevances, 'lr': args.lr_rel}
#     ]
#
#     return torch.optim.SGD(proto_param_list), torch.optim.SGD(rel_param_list), params_prototypes, params_relevances



def get_optimizer(model, args: argparse.Namespace) -> tuple:
    """
    Construct the optimizer as dictated by the parsed arguments
    :param model: The model that should be optimized
    :param args: Parsed arguments containing hyperparameters. Optimizer specific arguments (such as
    learning rate) can be passed this way as well
    :return: the optimizer corresponding to the parsed arguments, parameter set that can be frozen,
    and parameter set of the net that will be trained
    """

    #create parameter groups
    params_to_freeze = []
    params_to_train = []

    # dist_params = []
    # for name,param in model.named_parameters():
    #     if 'dist_params' in name:
    #         dist_params.append(param)

    # set up optimizer
    if ('resnet50_inat' in args.net) or ('convnext_tiny_13' in args.net) or ('resnet50' in args.net):  #to reproduce experimental results
        # freeze resnet50 except last convolutional layer
        for name, param in model.feature_extractor.named_parameters():
            if 'layer4.2' not in name:
                params_to_freeze.append(param)
            else:
                params_to_train.append(param)

        net_paramlist = [
            {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay, "momentum": args.momentum},
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay,"momentum": args.momentum},
            # {"params": model._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay,"momentum": args.momentum},
        ]
        proto_paramlist = [
            {"params": model.prototype_layer.xprotos,
             "lr": args.lr_protos,
             "weight_decay_rate": 0,
             "momentum": 0
             },
        ]
        rel_paramlist = [
            {"params": model.prototype_layer.relevances, "lr": args.lr_rel, "weight_decay_rate": args.weight_decay, "momentum": args.momentum}
        ]

    # else: #other network architectures
    #     for name, param in model._net.named_parameters():
    #         params_to_freeze.append(param)
    #     paramlist = [
    #         {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
    #         # {"params": model._add_on.parameters(), "lr": args.lr_block}, #"weight_decay_rate": args.weight_decay},
    #         {"params": model.prototype_layer.parameters(), "lr": args.lr_protos, "weight_decay_rate": 0},
    #         {"params": model.relevances.parameters(), "lr": args.lr_rel, "weight_decay_rate": 0}
    #     ]


    return torch.optim.Adam(net_paramlist), torch.optim.SGD(proto_paramlist), torch.optim.Adam(
        rel_paramlist), params_to_freeze, params_to_train
    # return torch.optim.Adam(net_paramlist), torch.optim.SGD(proto_paramlist), torch.optim.SGD(
    #     rel_paramlist), params_to_freeze, params_to_train