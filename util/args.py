"""
Modified Script - Original Source: https://github.com/M-Nauta/ProtoTree/tree/main

Description:
This script is a modified version of the original script by M. Nauta.
Modifications have been made to suit specific requirements or preferences.

"""

import os
import argparse
import pickle
import torch


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Train a PrototypeLayer.')
    parser.add_argument('--dataset',
                        type=str,
                        # default='ETH-80',
                        # default='CUB-200-2011',
                        # default='PETS',
                        # default='CARS',
                        # default='BRAIN',
                        # default='MURA',
                        default='SkinCancerISIC',
                        help='The name of dataset for training.')
    parser.add_argument('--nclasses',
                        type=int,
                        default=9, #37, #196, 9
                        help="The number of classes."
                        )
    parser.add_argument('--net',
                        type=str,
                        default='resnet50',
                        # default='resnet50_inat',
                        # default='convnext_tiny_13',
                        help='Base network used in the tree. Pretrained network on iNaturalist is only available for '
                             'resnet50_inat (default). Others are pretrained on ImageNet. Options are: resnet18, '
                             'resnet34, resnet50, resnet50_inat, resnet101, resnet152, densenet121, densenet169, '
                             'densenet201, densenet161, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn or '
                             'vgg19_bn')
    parser.add_argument('--loss_fn',
                        type=str,
                        default="ce",
                        help="The loss function to use. Options are 'ce', 'kl', 'reverse_kl', 'f_divergence', etc."
                        )
    parser.add_argument('--epsilon',
                        type=float,
                        default=0.001,
                        help="To create a smooth probability function (instead of one-hot encoding)."
                        )
    parser.add_argument('--beta',
                        type=float,
                        default=1.0,
                        help="The hyperparameter for the scoring function (related to temperature)."
                        )    
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="The random seed for initialization."
                        )
    # Training hyperparameters
    parser.add_argument('--batch_size_train',
                        type=int,
                        default=32,
                        help='Batch size of training data.')
    parser.add_argument('--batch_size_test',
                        type=int,
                        default=32,
                        help='Batch size for computing test error.')
    parser.add_argument('--nepochs',
                        type=int,
                        default=1000,
                        help='The number of epochs to train the prototypes.')

    # Prototype layer hyperparameters
    parser.add_argument('--num_features',
                        type=int,
                        default=512, # 512
                        help='Depth of the prototype and therefore also depth of convolutional output')
    
    parser.add_argument('--depth_of_net_last_layer',
                        type=int,
                        default=2048,
                        # default = 768,
                        help="Number of pixels in the last layer of the feature net."
                        )
    parser.add_argument('--W1',
                        type=int,
                        default=1,
                        help='Width of the prototype. Correct behaviour of the model with W1 != 1 is not guaranteed')
    parser.add_argument('--H1',
                        type=int,
                        default=1,
                        help='Height of the prototype. Correct behaviour of the model with H1 != 1 is not guaranteed')
    
    parser.add_argument('--proto_init',
                        type=str,
                        default="random",
                        help="The method for prototype initialization. Options are 'random', 'data', etc."
                        )
    parser.add_argument('--dim_of_subspace',
                        type=int,
                        default=10,
                        help="The dimensionality of subspaces 'd'."
                        )
    parser.add_argument('--coef_dim_of_subspace',
                        type=int,
                        default=1,
                        help="The number of times of d (for svd decomposition = coef x d)."
                        )
    # Optimization hyperparameters
    parser.add_argument('--proto_opt',
                        type=str,
                        default="exp",
                        help="The method for prototype updates on the Grassmann manifold. Options are 'exp', 'qr', or 'eucl'."
                        )
    parser.add_argument('--lr_protos',
                        type=float,
                        default=1e-2, #5e-2,
                        help='The learning rate for the training of the prototypes')
    parser.add_argument('--lr_rel',
                        type=float,
                        default=1e-5,#1e-6,
                        help='The learning rate for the training of the relevances.')
    parser.add_argument('--lr_block',
                        type=float,
                        default=1e-4, #1e-4
                        help='The optimizer learning rate for training the 1x1 conv layer and last conv layer of the underlying neural network (applicable to resnet50 and densenet121)')
    parser.add_argument('--freeze_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs where pretrained features_net will be frozen.'
                        )
    parser.add_argument('--lr_net',
                        type=float,
                        default=1e-5, #1e-5
                        help='The optimizer learning rate for the underlying neural network')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='The optimizer momentum parameter (only applicable to SGD)')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./run_prototypes',
                        help='The directory in which train progress should be logged')
    parser.add_argument('--disable_pretrained',
                        action='store_true',
                        help='When set, the backbone network is initialized with random weights instead of being '
                             'pretrained on another dataset). When not set, resnet50_inat is initalized with weights '
                             'from iNaturalist2017. Other networks are initialized with weights from ImageNet'
                        )

    args = parser.parse_args()
    
    return args



def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)                                                                               
    

def load_args(directory_path: str) -> argparse.Namespace:
    """
    Load the pickled arguments from the specified directory
    :param directory_path: The path to the directory from which the arguments should be loaded
    :return: the unpickled arguments
    """
    with open(directory_path + '/args.pickle', 'rb') as f:
        args = pickle.load(f)
    return args



