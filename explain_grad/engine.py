"""
explain_grad/engine.py

High-level driver for generating gradient-based feature importance heatmaps.
"""

import os
import logging
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Any

from explain_grad.attribution import compute_saliency, smoothgrad, cap_saliency_map, save_saliency_heatmap
from explain_grad.utils import load_and_process_images_generator


def run_explanation_engine(model: torch.nn.Module, args: Any, logger: logging.Logger):
    """
    Main entry point to run the gradient-based explanation process.
    """
    image_generator = load_and_process_images_generator(args, logger)
    
    processed_count = 0
    for img_name, label, img_transformed, original_image in image_generator:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Image {processed_count + 1}: {img_name}")
        logger.info(f"{'='*60}")

        explain_single_image(
            model=model,
            img_name=img_name,
            img_transformed=img_transformed,
            original_image=original_image,
            args=args,
            logger=logger
        )
        
        processed_count += 1
        if args.num_images and processed_count >= args.num_images:
            break
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Completed processing {processed_count} images.")
    logger.info(f"{'='*60}\n")


def explain_single_image(model: torch.nn.Module, 
                         img_name: str, 
                         img_transformed: torch.Tensor, 
                         original_image: Any, 
                         args: Any, 
                         logger: logging.Logger):
    """
    Generate and save heatmaps for a single image using gradient-based methods.
    """
    fname = os.path.splitext(img_name)[0]
    out_dir = os.path.join(args.results_dir, args.dataset, fname)
    os.makedirs(out_dir, exist_ok=True)
    
    # Prepare original image for overlay
    image_resized = original_image.resize((args.image_size, args.image_size))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    
    # Detach image for gradient computation
    image_tensor = img_transformed.detach()
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        target_class = output.argmax(dim=1).item()

    # Compute Saliency
    if args.method == 'smoothgrad':
        logger.info("Using SmoothGrad for feature importance.")
        saliency = smoothgrad(
            model, image_tensor, target_class, 
            n_samples=args.smoothgrad_samples, 
            noise_std=args.smoothgrad_noise_std
        )
        saliency = saliency * img_transformed  # Grad * Input
    elif args.method == 'raw_grad':
        logger.info("Using raw gradient for feature importance.")
        saliency = compute_saliency(model, image_tensor, target_class)
    elif args.method == 'grad_times_input':
        logger.info("Using gradient times input for feature importance.")
        grad = compute_saliency(model, image_tensor, target_class)
        saliency = grad * img_transformed
    else:
        raise ValueError(f"Unknown method '{args.method}'")

    # Process Saliency Map
    # Take absolute value and collapse color channels (max across channels)
    saliency_map, _ = torch.max(saliency.abs(), dim=0)
    saliency_map_np = saliency_map.detach().cpu().numpy()

    # Cap outliers
    saliency_map_np = cap_saliency_map(saliency_map_np, percentile=args.cap_percentile)

    # Log scaling if requested
    if args.add_log_scaling:
        saliency_map_np = np.log1p(saliency_map_np * args.add_log_scaling) / np.log1p(args.add_log_scaling)

    # Save Heatmap
    grayscale_path = os.path.join(out_dir, 'heatmap_grayscale.png')
    heatmap_norm = save_saliency_heatmap(saliency_map_np, grayscale_path)
    
    # Overlay
    overlay = 0.4 * image_np + 0.5 * heatmap_norm
    overlay_path = os.path.join(out_dir, 'heatmap_original_image.png')
    plt.imsave(overlay_path, overlay)
    
    logger.info(f"Heatmap saved in {out_dir}")
