import argparse
import os
import logging
from logging.handlers import RotatingFileHandler

from PIL import Image

import torch
import torchvision.transforms as transforms

from lvq.model import GrassmannLVQModel
from explain.feature_importance import compute_feature_importance_heatmap
from explain.args_explain import get_local_expl_args
from explain.data_utils import load_and_process_images_generator

from util.glvq import FDivergence
from util.load_model import load_grassmannlvq_model


# ------------------------------
# Logging Configuration
# ------------------------------
logger = logging.getLogger("ExplainAPI")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(
    filename='./explanations.log', maxBytes=10000, backupCount=1
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Also add console handler for real-time feedback
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)




def explain_decision(args: argparse.Namespace):
    """
    Explain model decisions by computing feature importance and visualizing regions of interest.
    Processes images one-by-one instead of batching.
    """
    
    # Load the trained model
    model, model_args = load_grassmannlvq_model(
        args_path=args.model_path + 'metadata/args.pickle',
        checkpoint_path=args.model_path + 'checkpoints/best_test_model',
    )

    # Transfer model arguments to args
    for k, v in model_args.__dict__.items():
        setattr(args, k, v)
    args.image_size = 224 #model_args.image_size

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

    # Update results directory path to include dataset name
    args.results_dir = os.path.join(args.results_dir, args.dataset)
    
    logger.info(f"Starting explanation process for dataset: {args.dataset}")
    logger.info(f"Model loaded from: {args.model_path}")
    logger.info(f"Results will be saved to: {args.results_dir}")
    logger.info(f"Number of classes: {args.nclasses}")
    logger.info(f"Number of prototypes: {model.prototype_layer.xprotos.shape[0]}\n")

    # Create image generator for one-by-one processing
    image_generator = load_and_process_images_generator(args, logger)
    # print(image_generator)
    # filename, label_tensor, transformed_image, image = next(image_generator)
    # print(filename, label_tensor)
    # print(image)

    # Compute feature importance heatmap for each image
    compute_feature_importance_heatmap(
        model=model, 
        image_generator=image_generator, 
        logger=logger, 
        args=args
    )
    
    # logger.info("Explanation process completed successfully!")



if __name__ == '__main__':
    args = get_local_expl_args()
    explain_decision(args)