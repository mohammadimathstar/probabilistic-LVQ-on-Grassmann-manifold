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
from explain.importance_scores import compute_feature_importance, save_feature_importance_heatmap
from lvq.model import GrassmannLVQModel


# helper for saving
def save_map_numpy(map_hw: torch.Tensor, path: str):
    arr = map_hw.detach().cpu().numpy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-12)
    heatmap = cv2.applyColorMap(np.uint8(255 * arr), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    plt.imsave(path, heatmap, vmin=0, vmax=1)
    return heatmap


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
    
    # Add batch dimension for model forward pass
    sample = img_transformed.unsqueeze(0)  # (1, C, H, W)
    
    # Get model outputs
    with torch.no_grad():
        feature, subspace, Rt, S, output = model.forward_partial(sample)
    
    # Compute feature importance gradient
    # Since nprotos = nclasses, we can use label as prototype index
    sim_grad = compute_feature_importance(
        feature, 
        Rt, 
        S, 
        output,
        model.prototype_layer.xprotos,
        model.prototype_layer.relevances,
        target_class=int(label.item())  # Get gradient for the true class
    )
    
    # sim_grad is now (C, H, W) for the target class
    # Aggregate over channels to get spatial importance map
    importance_map = torch.norm(sim_grad, dim=0)  # (H, W)
    
    # Upsample to original image size
    importance_map_upsampled = torch.nn.functional.interpolate(
        importance_map.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        size=(args.image_size, args.image_size),
        mode='bilinear',
        align_corners=False
    ).squeeze()  # (image_size, image_size)
    
    # Save heatmap
    heatmap_path = os.path.join(out_dir, 'heatmap.png')
    heatmap_np = save_feature_importance_heatmap(
        importance_map_upsampled, 
        heatmap_path,
        save_raw=True
    )
    logger.info(f"Saved heatmap to {heatmap_path}")
    
    # Create overlay if original image is available
    if original_image is not None:
        import numpy as np
        # Convert PIL image to numpy array
        img_np = np.array(original_image.resize((args.image_size, args.image_size)))
        if img_np.ndim == 2:  # Grayscale
            img_np = np.stack([img_np] * 3, axis=-1)
        img_rgb = img_np.astype(np.float32) / 255.0
        
        # Create overlay
        overlay = 0.5 * img_rgb + 0.5 * heatmap_np
        overlay_path = os.path.join(out_dir, 'overlay.png')
        plt.imsave(overlay_path, overlay, vmin=0, vmax=1)
        logger.info(f"Saved overlay to {overlay_path}")
    
    # Also compute and save gradients for all prototypes (optional)
    if args.__dict__.get('save_all_prototypes', False):
        all_grads = compute_feature_importance(
            feature, 
            Rt, 
            S, 
            output,
            model.prototype_layer.xprotos,
            model.prototype_layer.relevances,
            target_class=None  # Get all prototypes
        )
        # all_grads is (nprotos, C, H, W)
        for proto_idx in range(all_grads.shape[0]):
            proto_importance = torch.norm(all_grads[proto_idx], dim=0)  # (H, W)
            proto_importance_up = torch.nn.functional.interpolate(
                proto_importance.unsqueeze(0).unsqueeze(0),
                size=(args.image_size, args.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            proto_heatmap_path = os.path.join(out_dir, f'heatmap_proto_{proto_idx}.png')
            save_feature_importance_heatmap(proto_importance_up, proto_heatmap_path)
        
        logger.info(f"Saved heatmaps for all {all_grads.shape[0]} prototypes")


def compute_feature_importance_heatmap(
    model,
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
        
        compute_single_image_heatmap(
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


