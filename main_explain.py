import os
import logging
from logging.handlers import RotatingFileHandler

import torch

from explain.config import get_unified_explain_args
from util.load_model import load_grassmannlvq_model


def setup_logger(log_file: str = './explanations.log') -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger("ExplainAPI")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = RotatingFileHandler(filename=log_file, maxBytes=100000, backupCount=1)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def main():
    """
    Main entry point for explaining model decisions.
    Supports both 'regions' and 'pixels' modes.
    """
    # 1. Load configuration
    args = get_unified_explain_args()
    logger = setup_logger()

    # 2. Load the trained model
    logger.info(f"Loading model from: {args.model_path}")
    model, model_args = load_grassmannlvq_model(
        args_path=os.path.join(args.model_path, 'metadata/args.pickle'),
        checkpoint_path=os.path.join(args.model_path, 'checkpoints/best_test_model'),
    )
    model.eval()

    # 3. Transfer model-specific arguments
    args.nclasses = model_args.nclasses
    # Ensure image size is consistent
    if not hasattr(args, 'image_size') or args.image_size is None:
        args.image_size = getattr(model_args, 'image_size', 224)

    # 4. Run appropriate explanation engine
    if args.mode == 'regions':
        from explain.regions.engine import run_explanation_engine
        logger.info(f"Starting 'regions' explanation process for dataset: {args.dataset}")
    elif args.mode == 'pixels':
        from explain.pixels.engine import run_explanation_engine
        logger.info(f"Starting 'pixels' explanation process for dataset: {args.dataset}")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    run_explanation_engine(model=model, args=args, logger=logger)


if __name__ == '__main__':
    main()
