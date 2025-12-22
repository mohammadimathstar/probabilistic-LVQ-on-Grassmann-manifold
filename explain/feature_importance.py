"""
explain/feature_importance.py

High-level driver to compute per-image feature importance heatmaps.
Supports multiple explanation methods.
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional



from explain.importance_scores import compute_feature_importance, save_feature_importance_heatmap
import torch

from typing import List

from lvq.model import GrassmannLVQModel
from explain.visualization import plot_important_region_per_principal_direction
import torchvision.transforms.functional as F


# helper for saving
def save_map_numpy(map_hw: torch.Tensor, path: str):
    arr = map_hw.detach().cpu().numpy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-12)
    heatmap = cv2.applyColorMap(np.uint8(255 * arr), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    plt.imsave(path, heatmap, vmin=0, vmax=1)
    return heatmap


def compute_feature_importance_heatmap(
    model: GrassmannLVQModel,
    image_generator,
    logger,
    args,
):
    """
    Compute feature importance heatmaps for all images using a generator.
    
    Args:
        model: GrassmannLVQ model
        image_generator: Generator yielding (filename, label, transformed_image, original_image)
        logger: Logger instance
        args: Arguments namespace
    """
    os.makedirs(args.results_dir, exist_ok=True)
    
    processed_count = 0
    for img_name, label, img_transformed, original_image in image_generator:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing image {processed_count + 1}: {img_name}")
        logger.info(f"{'='*60}")

        region_importance_per_principal_dir = compute_single_image_heatmap(
            model=model,
            img_name=img_name,
            img_transformed=img_transformed,
            label=label,
            logger=logger,
            args=args,
            original_image=original_image
        )
        
        
        processed_count += 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Completed processing {processed_count} images")
    logger.info(f"{'='*60}\n")


def compute_single_image_heatmap(
    model,
    img_name: str,
    img_transformed: torch.Tensor,
    label: torch.Tensor,
    logger,
    args,
    original_image=None,
):
    """
    Compute and save heatmap for a single image.
    
    Args:
        model: GrassmannLVQ model
        img_name: Image filename
        img_transformed: Transformed image tensor (C, H, W)
        label: Ground truth label (scalar tensor)
        logger: Logger instance
        args: Arguments namespace
        original_image: Original PIL image for overlay (optional)
    """
    fname = os.path.splitext(img_name)[0]
    out_dir = os.path.join(args.results_dir, fname)
    os.makedirs(out_dir, exist_ok=True)
    
    
    image_resized = F.resize(original_image, (args.image_size, args.image_size)) 
    image_resized_np = np.array(image_resized).astype(np.float32) / 255.0
    
    if args.which_direction is not None:
        with torch.no_grad():
            model.prototype_layer.relevances[0, :] = 0 # only focus on first principal direction
            model.prototype_layer.relevances[0, args.which_direction] = 1.0 # only focus on first principal direction


    # Add batch dimension for model forward pass
    sample = img_transformed.unsqueeze(0)  # (1, C, H, W)

    # Get model outputs
    with torch.no_grad():
        feature, subspace, Rt, S, output = model.forward_partial(sample)

        region_heatmap, rotated_prototype_pos = compute_feature_importance(
            feature, label, Rt, S, output,
            model.prototype_layer.xprotos,
            model.prototype_layer.relevances,
            k_negatives=args.k_negatives,
            args=args,   
        )

    HEATMAP_PATH = os.path.join(out_dir, 'heatmap.png')
    save_feature_importance_heatmap(region_heatmap, output_path=HEATMAP_PATH)
    logger.info(f"The importance of regions (of '{img_name}') has been completed!")
    logger.info(f"Its heatmap has been saved in '{HEATMAP_PATH}'.")

    # Resize to image size and save the (upsampled) heatmap
    heatmap_upsampled = cv2.resize(
        region_heatmap.numpy(),
        dsize=(args.image_size, args.image_size), #(sample_array.shape[1], sample_array.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    UPSAMPLED_HEATMAP_PATH = os.path.join(out_dir, 'heatmap_upsampled.png')
    heatmap_upsampled_normalized = save_feature_importance_heatmap(heatmap_upsampled, UPSAMPLED_HEATMAP_PATH)

    # Use 0.6/0.4 ratio for better visibility
    overlay = 0.6 * image_resized_np  + 0.4 * heatmap_upsampled_normalized
    OVERLAY_PATH = os.path.join(out_dir, 'heatmap_original_image.png')

    plt.imsave(fname=OVERLAY_PATH, arr=overlay, vmin=0.0, vmax=1.0)


    logger.info(f"The image with its heatmap has been saved in '{OVERLAY_PATH}'.\n")

    plot_important_region_per_principal_direction(
            image_resized_np, feature[0], rotated_prototype_pos, img_name, args)








