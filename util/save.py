import argparse
from lvq.model import GrassmannLVQModel
from util.log import Log


def save_model(
    model: GrassmannLVQModel,
    epoch: int,
    log: Log,
    args: argparse.Namespace
):
    """Save latest model and periodic checkpoints."""
    model.eval()

    # Always save latest
    model.save_state(f'{log.checkpoint_dir}/latest')

    # Save periodically or at the last epoch
    if epoch == args.nepochs or epoch % 10 == 0:
        model.save_state(f'{log.checkpoint_dir}/epoch_{epoch}')


def save_best_train_model(
    model: GrassmannLVQModel,
    best_train_acc: float,
    train_acc: float,
    log: Log
) -> float:
    """Save model when training accuracy improves."""
    model.eval()
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        model.save_state(f'{log.checkpoint_dir}/best_train_model')
    return best_train_acc


def save_best_test_model(
    model: GrassmannLVQModel,
    best_test_acc: float,
    test_acc: float,
    log: Log
) -> float:
    """Save model when test accuracy improves."""
    model.eval()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        model.save_state(f'{log.checkpoint_dir}/best_test_model')
    return best_test_acc


def save_model_description(
    model: GrassmannLVQModel,
    description: str,
    log: Log
):
    """Save model with a custom description."""
    model.eval()
    model.save_state(f'{log.checkpoint_dir}/{description}')



# """
# Modified Script - Original Source: https://github.com/M-Nauta/ProtoTree/tree/main

# Description:
# This script is a modified version of the original script by M. Nauta.
# Modifications have been made to suit specific requirements or preferences.

# """

# import argparse
# from lvq.model import GrassmannLVQModel
# from util.log import Log

# def save_model(
#         model: GrassmannLVQModel,
#         epoch: int,
#         log: Log,
#         args: argparse.Namespace
# ):
#     model.eval()
#     # Save latest model
#     model.save(f'{log.checkpoint_dir}/latest')
#     model.save_state(f'{log.checkpoint_dir}/latest')

#     # Save model every 10 epochs
#     if epoch == args.nepochs or epoch%10==0:
#         model.save(f'{log.checkpoint_dir}/epoch_{epoch}')
#         model.save_state(f'{log.checkpoint_dir}/epoch_{epoch}')

# def save_best_train_model(
#         model: GrassmannLVQModel,
#         best_train_acc: float,
#         train_acc: float,
#         log: Log
# ):
#     model.eval()
#     if train_acc > best_train_acc:
#         best_train_acc = train_acc
#         model.save(f'{log.checkpoint_dir}/best_train_model')
#         model.save_state(f'{log.checkpoint_dir}/best_train_model')

#     return best_train_acc

# def save_best_test_model(
#         model: GrassmannLVQModel,
#         best_test_acc: float,
#         test_acc: float,
#         log: Log
# ):
#     model.eval()
#     if test_acc > best_test_acc:
#         best_test_acc = test_acc
#         model.save(f'{log.checkpoint_dir}/best_test_model')
#         model.save_state(f'{log.checkpoint_dir}/best_test_model')
#     return best_test_acc

# def save_model_description(
#         model: GrassmannLVQModel,
#         description: str,
#         log: Log
# ):
#     model.eval()
#     # Save model with description
#     model.save(f'{log.checkpoint_dir}/'+description)
#     model.save_state(f'{log.checkpoint_dir}/'+description)
