"""
explain/engine.py

High-level driver for generating feature importance heatmaps for multiple images.
"""

import os
import logging
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Any, Tuple, List

from explain.regions.attribution import compute_feature_importance, save_importance_heatmap
from explain.regions.visualization import (
    plot_important_region_per_principal_direction, 
    visualize_regions,
    visualize_regions_with_patch_matching
)
from explain.regions.patch_finder import find_closest_patches_from_dataset, extract_and_save_patches
from explain.common_utils import load_and_process_images_generator
from util.data import get_dataloaders, get_data


def run_explanation_engine(model: torch.nn.Module, args: Any, logger: logging.Logger):
    """
    Main entry point to run the explanation process for a set of images.
    """
    # Get classes for visualization
    _, _, _, classes, _ = get_data(args)
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
            label=label,
            original_image=original_image,
            classes=classes,
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
                         label: torch.Tensor, 
                         original_image: Any, 
                         classes: List[str],
                         args: Any, 
                         logger: logging.Logger):
    """
    Generate and save all heatmaps (total and per-direction) for a single image.
    """
    fname = os.path.splitext(img_name)[0]
    out_dir = os.path.join(args.results_dir, args.dataset, fname)
    os.makedirs(out_dir, exist_ok=True)
    
    # Prepare original image for overlay
    image_resized = original_image.resize((args.image_size, args.image_size))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    
    # Forward pass
    sample = img_transformed.unsqueeze(0)
    with torch.no_grad():
        feature, _, Rt, S, output = model.forward_partial(sample)

    overlay_path = os.path.join(out_dir, 'original_image.png')
    plt.imsave(overlay_path, image_np)

    # 1. Total Heatmap
    region_heatmap, rotated_prototype_pos, total_max = _generate_total_heatmap(
        model, feature, label, Rt, S, output, out_dir, img_name, image_np, args, logger
    )

    # 2. Per-Direction Heatmaps
    dir_max_values = _generate_per_direction_heatmaps(
        model, feature, label, Rt, S, output, out_dir, img_name, image_np, args, logger
    )

    # 3. Additional Visualizations
    patch_base_dir = os.path.join(args.results_dir, args.dataset, 'patchs')
    class_name = classes[label.item()]
    plot_important_region_per_principal_direction(
        image_np, feature[0], rotated_prototype_pos, img_name, args,
        patch_base_dir=patch_base_dir,
        class_name=class_name
    )

    # 4. Summary Visualization
    # Get relevances (shared across prototypes in this model version)
    with torch.no_grad():
        relevances = model.prototype_layer.relevances[0].cpu().numpy()

    visualize_regions(out_dir, relevances=relevances)
    
    # 5. Summary Visualization with Patch Matching
    visualize_regions_with_patch_matching(
        out_dir, 
        patch_base_dir, 
        class_name, 
        relevances=relevances,
        dir_max_values=dir_max_values,
        total_max=total_max,
        grid_size=(7, 7)
    )
    
    logger.info(f"The image with its heatmap has been saved in '{out_dir}'.\n")


def _generate_total_heatmap(model, feature, label, Rt, S, output, out_dir, img_name, image_np, args, logger):
    """Internal helper to generate the total importance heatmap."""
    region_heatmap, rotated_prototype_pos = compute_feature_importance(
        feature, label, Rt, S, output,
        model.prototype_layer.xprotos,
        model.prototype_layer.relevances,
        k_negatives=args.k_negatives,
        args=args,
        print_info=True
    )

    heatmap_path = os.path.join(out_dir, 'heatmap.png')
    save_importance_heatmap(region_heatmap, heatmap_path)
    
    # Upsample and Overlay
    heatmap_upsampled = cv2.resize(
        region_heatmap.detach().cpu().numpy(),
        dsize=(args.image_size, args.image_size),
        interpolation=cv2.INTER_CUBIC
    )
    
    upsampled_path = os.path.join(out_dir, 'heatmap_upsampled.png')
    heatmap_norm = save_importance_heatmap(heatmap_upsampled, upsampled_path)
    
    overlay = 0.6 * image_np + 0.4 * heatmap_norm
    overlay_path = os.path.join(out_dir, 'heatmap_original_image.png')
    plt.imsave(overlay_path, overlay)
    
    logger.info(f"Total importance heatmap saved in {out_dir}")
    total_max = region_heatmap.max().item()
    return region_heatmap, rotated_prototype_pos, total_max


def _generate_per_direction_heatmaps(model, feature, label, Rt, S, output, out_dir, img_name, image_np, args, logger):
    """Internal helper to generate heatmaps for each principal direction."""
    subspace_dim = model.prototype_layer.relevances.shape[1]
    directions_dir = os.path.join(out_dir, 'directions')
    os.makedirs(directions_dir, exist_ok=True)

    logger.info(f"Generating heatmaps for {subspace_dim} principal directions...")
    dir_max_values = []
    
    for d in range(subspace_dim):
        # Create temporary relevances with only direction d active
        temp_relevances = torch.zeros_like(model.prototype_layer.relevances)
        temp_relevances[0, d] = 1.0
        
        region_heatmap_d, _ = compute_feature_importance(
            feature, label, Rt, S, output,
            model.prototype_layer.xprotos,
            temp_relevances,
            k_negatives=args.k_negatives,
            args=args,
            print_info=False
        )
        
        # Save heatmap
        path_d = os.path.join(directions_dir, f'heatmap_dir_{d}.png')
        save_importance_heatmap(region_heatmap_d, path_d)
        
        # Upsample and Overlay
        heatmap_upsampled_d = cv2.resize(
            region_heatmap_d.detach().cpu().numpy(),
            dsize=(args.image_size, args.image_size),
            interpolation=cv2.INTER_CUBIC
        )
        
        upsampled_path_d = os.path.join(directions_dir, f'heatmap_upsampled_dir_{d}.png')
        heatmap_norm_d = save_importance_heatmap(heatmap_upsampled_d, upsampled_path_d)
        
        overlay_d = 0.6 * image_np + 0.4 * heatmap_norm_d
        overlay_path_d = os.path.join(directions_dir, f'heatmap_overlay_dir_{d}.png')
        plt.imsave(overlay_path_d, overlay_d)
        
        dir_max_values.append(region_heatmap_d.max().item())

    logger.info(f"Per-direction heatmaps saved in {directions_dir}")
    return dir_max_values


def run_patch_finding_engine(model: torch.nn.Module, args: Any, logger: logging.Logger):
    """
    Entry point for finding the closest patches in the training set to prototype directions.
    """
    logger.info("Initializing patch finding engine...")
    
    # Get projectset (training set without augmentation)
    _, projectset, _, classes, _ = get_data(args)
    
    if hasattr(projectset, 'root'):
        logger.info(f"Using data from folder: {projectset.root}")
    
    if args.num_images:
        # Create a balanced subset to ensure all classes are represented
        num_classes = len(classes)
        images_per_class = max(1, args.num_images // num_classes)
        
        # Group indices by class
        class_indices = [[] for _ in range(num_classes)]
        # projectset is an ImageFolder
        for i, (_, label) in enumerate(projectset.samples):
            class_indices[label].append(i)
            
        indices = []
        for c_idx in range(num_classes):
            c_indices = class_indices[c_idx]
            if len(c_indices) > images_per_class:
                sampled = random.sample(c_indices, images_per_class)
            else:
                sampled = c_indices
            indices.extend(sampled)
            
        # Sort indices to maintain some order
        indices.sort()
        projectset = torch.utils.data.Subset(projectset, indices)
        logger.info(f"Created a balanced subset of {len(indices)} images across {num_classes} classes.")

    cuda = not args.disable_cuda and torch.cuda.is_available()
    projectloader = torch.utils.data.DataLoader(
        projectset,
        batch_size=getattr(args, 'batch_size_train', 32),
        shuffle=False,
        pin_memory=cuda
    )
    
    # Find closest patches
    best_patches = find_closest_patches_from_dataset(
        model=model,
        dataloader=projectloader,
        args=args,
        logger=logger
    )
    
    # Extract and save patches
    extract_and_save_patches(
        best_patches=best_patches,
        classes=classes,
        args=args,
        logger=logger
    )
