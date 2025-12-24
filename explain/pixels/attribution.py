"""
explain_grad/attribution.py

Core logic for gradient-based attribution methods (Saliency, SmoothGrad).
"""

import torch
import numpy as np
import cv2
from typing import Optional, Any


def compute_saliency(model: torch.nn.Module, image: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
    """
    Compute the raw gradient of the target class score with respect to the input image.
    
    Args:
        model: The model to explain
        image: Input image tensor (C, H, W)
        target_class: Index of the target class (default: predicted class)
        
    Returns:
        Gradient tensor of shape (C, H, W)
    """
    model.zero_grad()

    if not image.requires_grad:
        image.requires_grad_(True)

    output = model(image.unsqueeze(0))  # Add batch dimension

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    score = output[0, target_class]
    score.backward()

    if image.grad is None:
        raise RuntimeError("Gradient is None. Ensure the image is a leaf tensor and requires_grad=True.")

    return image.grad.detach()


def smoothgrad(model: torch.nn.Module, 
               image: torch.Tensor, 
               target_class: int, 
               n_samples: int = 50, 
               noise_std: float = 0.1) -> torch.Tensor:
    """
    Compute SmoothGrad saliency by averaging gradients of noisy versions of the image.
    """
    model.eval()
    smooth_saliency = torch.zeros_like(image)

    for _ in range(n_samples):
        noise = torch.randn_like(image) * noise_std
        noisy_image = (image + noise).detach().clone()
        noisy_image.requires_grad_(True)

        saliency = compute_saliency(model, noisy_image, target_class)
        smooth_saliency += saliency

    smooth_saliency /= n_samples
    return smooth_saliency


def cap_saliency_map(saliency_map: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Cap extreme values of the saliency map at the given percentile and normalize to [0, 1].
    """
    cap_value = np.percentile(saliency_map, percentile)
    saliency_capped = np.minimum(saliency_map, cap_value)
    
    # Normalize to [0, 1]
    saliency_capped = saliency_capped - np.min(saliency_capped)
    saliency_capped /= (np.max(saliency_capped) + 1e-8)
    
    return saliency_capped


def save_saliency_heatmap(saliency_map: np.ndarray, output_path: str) -> np.ndarray:
    """
    Save a saliency map as a grayscale image and return a JET colormap version.
    """
    # Normalize to [0, 1]
    rescaled = (saliency_map - np.amin(saliency_map))
    rescaled = rescaled / (np.amax(rescaled) + 1e-8)

    # Save grayscale
    gray_uint8 = np.uint8(255 * rescaled)
    cv2.imwrite(output_path, gray_uint8)

    # Create JET heatmap
    heatmap = cv2.applyColorMap(gray_uint8, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # BGR to RGB
    
    return heatmap
