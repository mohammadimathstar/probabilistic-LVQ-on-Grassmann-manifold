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

from typing import Union
from torch import Tensor

from scipy.ndimage import gaussian_filter


# from explain.importance_scores import compute_feature_importance, save_feature_importance_heatmap
import torch

from typing import List

from lvq.model import GrassmannLVQModel
import torchvision.transforms.functional as F

# from explain.explain_utils import produce_map_from_grad_or_compute

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
    model,
    image_generator,
    logger,
    args,
    # method: str = 'raw_grad',
    # grad_provided: Optional[List[torch.Tensor]] = None,
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
            logger=logger,
            args=args,
            original_image=original_image
        )
        
        
        processed_count += 1
        # break
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Completed processing {processed_count} images")
    logger.info(f"{'='*60}\n")


def compute_single_image_heatmap(
    model,
    img_name: str,
    img_transformed: torch.Tensor,
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

    ##################
    # Make sure original image does NOT keep old gradients
    image = img_transformed.detach()

    # Get prediction
    output = model(image.unsqueeze(0))
    target_class = output.argmax(dim=1).item()

    if args.method == 'smoothgrad':
        logger.info(f"Using SmoothGrad for feature importance.")

        # Compute SmoothGrad saliency
        saliency = smoothgrad(
            model,
            image,
            target_class,
            n_samples=args.smoothgrad_samples,
            noise_std=args.smoothgrad_noise_std,
        )
        
        saliency = saliency * img_transformed
    elif args.method == 'raw_grad':
        logger.info(f"Using raw gradient for feature importance.")

        saliency = compute_saliency(model, image, target_class)
        
    elif args.method == 'grad_times_input':
        logger.info(f"Using gradient times input for feature importance.")

        grad = compute_saliency(model, image, target_class)
        saliency = grad * img_transformed
    else:
        raise ValueError(f"Unknown method '{args.method}' for feature importance.")

    
    # Take absolute value and collapse color channels
    saliency_map, _ = torch.max(saliency.abs(), dim=0) # common
    # saliency_map, _ = saliency.abs().median(dim=0)  # alternative
    # saliency_map = saliency.abs().sum(dim=0)
    # saliency_map = saliency.abs().mean(dim=0)


    # Collapse channels if needed
    saliency_map = saliency_map.cpu().numpy()    

    # Cap outliers
    saliency_map = cap_saliency_map(saliency_map, percentile=args.cap_percentile)

    if args.add_log_scaling is not None and args.add_log_scaling:
        saliency_map = np.log1p(saliency_map * args.add_log_scaling) / np.log1p(args.add_log_scaling)

    # sigma = 0.2  # controls the amount of smoothing
    # saliency_map = gaussian_filter(saliency_map, sigma=sigma)

    UPSAMPLED_HEATMAP_PATH = os.path.join(out_dir, 'heatmap_grayscale.png')
    heatmap_upsampled_normalized = save_feature_importance_heatmap(saliency_map, UPSAMPLED_HEATMAP_PATH)

    # Use 0.6/0.4 ratio for better visibility
    overlay = 0.4 * image_resized_np  + 0.5 * heatmap_upsampled_normalized
    OVERLAY_PATH = os.path.join(out_dir, 'heatmap_original_image.png')

    plt.imsave(fname=OVERLAY_PATH, arr=overlay, vmin=0.0, vmax=1.0)


    logger.info(f"The image with its heatmap has been saved in '{OVERLAY_PATH}'.\n")



def compute_saliency(model, image, target_class=None):
    """
    image: tensor of shape (C, H, W), must be a leaf tensor
    """
    model.zero_grad()

    if image.requires_grad is False:
        image.requires_grad_(True)

    output = model(image.unsqueeze(0))  # add batch dim

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    score = output[0, target_class]
    score.backward()

    assert image.grad is not None, "Gradient is None — image is not a leaf tensor"

    return image.grad.detach()




def smoothgrad(model, image, target_class, n_samples=50, noise_std=0.1):
    """
    image: tensor of shape (C, H, W)
    """
    model.eval()
    smooth_saliency = torch.zeros_like(image)

    for _ in range(n_samples):
        noise = torch.randn_like(image) * noise_std

        # Create leaf tensor
        noisy_image = (image + noise).detach().clone()
        noisy_image.requires_grad_(True)

        saliency = compute_saliency(model, noisy_image, target_class)
        smooth_saliency += saliency

    smooth_saliency /= n_samples
    return smooth_saliency


# import numpy as np

def cap_saliency_map(saliency_map: np.ndarray, percentile: float = 99.0):
    """
    Caps extreme values of saliency map at the given percentile.
    
    Args:
        saliency_map: 2D numpy array (H, W)
        percentile: high percentile value to cap
    
    Returns:
        saliency_capped: capped map (H, W)
    """
    # Compute the percentile value
    cap_value = np.percentile(saliency_map, percentile)
    
    # Cap values
    saliency_capped = np.minimum(saliency_map, cap_value)
    
    # Optional: normalize to [0,1]
    saliency_capped = saliency_capped - np.min(saliency_capped)
    saliency_capped /= (np.max(saliency_capped) + 1e-8)
    
    return saliency_capped



def save_feature_importance_heatmap(
        effect_map: Union[Tensor, np.array],
        output_path: str,
) -> np.ndarray:
    """Generate and save heatmap from effect map."""

    if isinstance(effect_map, Tensor):
        effect_map_np = effect_map.detach().cpu().numpy()
    else:
        effect_map_np = effect_map

    print(effect_map_np.shape)
    rescaled_map = (effect_map_np - np.amin(effect_map_np))
    rescaled_map = rescaled_map / (np.amax(rescaled_map) + 1e-8)

    gray_map_uint8 = np.uint8(255 * rescaled_map)
    cv2.imwrite(output_path, gray_map_uint8)

    # Apply colormap to create heatmap
    heatmap = cv2.applyColorMap(gray_map_uint8, cv2.COLORMAP_JET)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]

    # plt.imsave(fname=output_path, arr=heatmap, vmin=0.0, vmax=1.0)
    return heatmap
