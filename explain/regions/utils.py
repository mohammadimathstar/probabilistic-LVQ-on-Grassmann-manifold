"""
explain/regions/utils.py

Method-specific utility functions for region-based explanations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def upto_image_size(x_hw: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
    """
    Upsample a (H, W) or (1, 1, H, W) tensor to the target image size using bilinear interpolation.
    """
    if x_hw.dim() == 2:
        x = x_hw.unsqueeze(0).unsqueeze(0)
    elif x_hw.dim() == 3 and x_hw.shape[0] == 1:
        x = x_hw.unsqueeze(0)
    else:
        x = x_hw.unsqueeze(0) if x_hw.dim() == 3 else x_hw
    
    x_up = F.interpolate(x, size=img_size, mode='bilinear', align_corners=False)
    return x_up.squeeze(0).squeeze(0)


def raw_grad_map(grad_feat: torch.Tensor) -> torch.Tensor:
    """
    Compute importance map by summing gradients over channels.
    grad_feat: (C, H, W) -> (H, W)
    """
    return grad_feat.sum(dim=0)


def input_x_grad_map(feature_map: torch.Tensor, grad_feat: torch.Tensor) -> torch.Tensor:
    """
    Compute importance map as the absolute sum of (feature_map * grad_feat) over channels.
    """
    prod = feature_map * grad_feat
    return torch.abs(prod.sum(dim=0))


def gradcam_map(feature_map: torch.Tensor, grad_feat: torch.Tensor, relu: bool = True) -> torch.Tensor:
    """
    Compute Grad-CAM map.
    """
    C, H, W = feature_map.shape
    alpha = grad_feat.view(C, -1).mean(dim=1)  # Global average pooling of gradients
    cam = (alpha.view(C, 1, 1) * feature_map).sum(dim=0)
    if relu:
        cam = F.relu(cam)
    return cam
