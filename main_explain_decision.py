import argparse
import os
import logging
from logging.handlers import RotatingFileHandler

from PIL import Image

import torch
import torchvision.transforms as transforms

from lvq.model import GrassmannLVQModel
from explain.feature_importance import (
    compute_feature_importance_heatmap, plot_important_region_per_principal_direction)
from explain.args_explain import get_local_expl_args
from explain.data_utils import load_and_process_images

from util.glvq import FDivergence
from util.load_model import load_grassmannlvq_model


# ------------------------------
# Logging Configuration
# ------------------------------
logger = logging.getLogger("ExplainAPI")
logger.setLevel(logging.INFO)

fomatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(
    filename='./explanations.log', maxBytes=10000, backupCount=1
)
file_handler.setFormatter(fomatter)
logger.addHandler(file_handler)




def explain_decision(args: argparse.Namespace):
    """
    Explain model decisions by computing feature importance and visualizing regions of interest.
    """
    
    # Load the trained model
    model, model_args = load_grassmannlvq_model(
        args_path=args.model_path + 'metadata/args.pickle',
        checkpoint_path=args.model_path + 'checkpoints/best_test_model',
    )

    # model = GrassmannLVQModel.load(args.model)
    for k, v in model_args.__dict__.items():
        setattr(args, k, v)

    # Define loss function for explanation
    # loss_fn = FDivergence(reduction='batchmean', eps=args.epsilon)

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

    # Update results directory path to include dataset name
    args.results_dir = os.path.join(args.results_dir, args.dataset)
    # os.makedirs(args.results_dir, exist_ok=True)

    # Process images and obtain transformed data
    images_names, labels, transformed_images = load_and_process_images(args, logger)

    # Compute feature importance heatmap
    compute_feature_importance_heatmap(
        model, images_names, transformed_images, labels, logger, args)

    # Plot and save important regions per principal direction
    # plot_important_region_per_principal_direction(
    #     imgs, region_importance_per_principal_dir, images_names, args, 1)



if __name__ == '__main__':
    args = get_local_expl_args()
    explain_decision(args)