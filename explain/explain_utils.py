"""
explain/explain_utils.py

Utility functions for producing region- and pixel-level importance maps
from gradients and convolutional feature maps.

Supported methods:
 - raw_grad        : per-spatial-location L2 norm over channel gradients
 - input_x_grad    : elementwise product (feature_map * gradient), then channel-norm
 - gradcam         : Grad-CAM style: alpha_c = global-avg-pool(grad), map = ReLU(sum_c alpha_c * feat_c)
 - smoothgrad      : average raw_grad over noisy inputs
 - integrated_gradients : Integrated Gradients on input images (end-to-end)

Notes:
 - This module assumes feature maps are of shape (C, H, W).
 - Upsampling to image size uses bilinear interpolation.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def upto_image_size(x_hw: torch.Tensor, img_size: Tuple[int,int]) -> torch.Tensor:
    """Upsample (H, W) or (1,1,H,W) tensor to image size (H_img, W_img)."""
    if x_hw.dim() == 2:
        x = x_hw.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    elif x_hw.dim() == 3 and x_hw.shape[0] == 1:
        x = x_hw.unsqueeze(0)  # 1x1xHxW
    else:
        # assume shape 1x1xHxW or Bx1xHxW
        x = x_hw.unsqueeze(0) if x_hw.dim() == 3 else x_hw
    x_up = F.interpolate(x, size=img_size, mode='bilinear', align_corners=False)
    return x_up.squeeze(0).squeeze(0)


def raw_grad_map(grad_feat: torch.Tensor) -> torch.Tensor:
    """
    grad_feat: (C, H, W)
    return: (H, W) with L2 norm over channels
    """
    return grad_feat.sum(dim=0) # torch.norm(grad_feat, dim=0)
    

def input_x_grad_map(feature_map: torch.Tensor, grad_feat: torch.Tensor) -> torch.Tensor:
    """
    Compute elementwise product then aggregate channels by L2 norm.
    feature_map, grad_feat: (C, H, W)
    """
    # prod = torch.abs(feature_map) * grad_feat
    prod = feature_map * grad_feat
    #########CHECK *****************
    # We sum over channels to preserve sign (unlike norm which is always positive)
    # return prod.sum(dim=0)
    # sal_map, _ = torch.max(prod.abs(), dim=0)
    # return sal_map

    return torch.abs(prod.sum(dim=0))
    # return torch.norm(prod, dim=0)


def gradcam_map(feature_map: torch.Tensor, grad_feat: torch.Tensor, relu: bool = True) -> torch.Tensor:
    """
    Standard Grad-CAM: alpha_c = spatial average of grad over HxW; map = sum_c alpha_c * feat_c
    feature_map, grad_feat: (C, H, W)
    """
    C, H, W = feature_map.shape
    alpha = grad_feat.view(C, -1).mean(dim=1)  # C
    # weighted sum
    cam = (alpha.view(C,1,1) * feature_map).sum(dim=0)  # HxW
    if relu:
        cam = F.relu(cam)

    return cam


def smoothgrad_feature_map(
    feature_map: torch.Tensor,
    compute_grad_fn,
    n_samples: int = 20,
    stdev_spread: float = 0.15,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    SmoothGrad at the feature-map level:
    - compute_grad_fn: function that given a noisy image or feature map returns gradient wrt feature map (C,H,W)
      We keep this generic: the caller can supply a function that returns grad for a noisy input.
    - stdev_spread: fraction of the (max-min) range to use as noise std.
    """
    dev = device or feature_map.device
    fm = feature_map.detach()
    fm_min, fm_max = fm.min(), fm.max()
    stdev = float(stdev_spread * (fm_max - fm_min).item())

    acc = torch.zeros_like(fm, device=dev)
    for i in range(n_samples):
        noise = torch.randn_like(fm, device=dev) * stdev
        noisy = fm + noise
        grad_noisy = compute_grad_fn(noisy)  # should return (C,H,W)
        acc += grad_noisy
    avg_grad = acc / float(n_samples)
    return torch.norm(avg_grad, dim=0)


def integrated_gradients_image(
    model_forward_fn,
    input_image: torch.Tensor,
    target_index: int,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 50,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute Integrated Gradients with respect to the input image (pixel-level).
    - model_forward_fn: a function that given input image tensor (1,C,H,W) returns scalar score (logit or similarity)
    - input_image: (C,H,W) or (1,C,H,W) tensor
    - target_index: the class/prototype index whose score to attribute (if forward returns vector of logits)
    - baseline: (C,H,W) or (1,C,H,W) baseline image; if None uses zeros
    - steps: number of steps for approximation
    Returns:
      attributions (H_img, W_img) (pixel-level map)
    """
    dev = device or input_image.device
    if input_image.dim() == 3:
        inp = input_image.unsqueeze(0).to(dev)
    else:
        inp = input_image.to(dev)
    if baseline is None:
        baseline = torch.zeros_like(inp).to(dev)
    else:
        if baseline.dim() == 3:
            baseline = baseline.unsqueeze(0).to(dev)

    # scale inputs and accumulate gradients
    total_grad = torch.zeros_like(inp, device=dev)
    for step in range(1, steps+1):
        alpha = step / float(steps)
        interp = baseline + alpha * (inp - baseline)
        interp.requires_grad_(True)
        out = model_forward_fn(interp)  # assume returns scalar or vector
        if out.dim() > 0:
            score = out[0, target_index]
        else:
            score = out
        score.backward(retain_graph=True)
        if interp.grad is None:
            grad = torch.zeros_like(interp)
        else:
            grad = interp.grad.detach()
        total_grad += grad
        interp.grad.zero_()
    avg_grad = total_grad / float(steps)  # shape 1xCxHxW
    attributions = (inp - baseline) * avg_grad  # elementwise
    # collapse channels to make single map
    map_hw = torch.norm(attributions.squeeze(0), dim=0)
    return map_hw.detach()


# Convenience wrapper that takes either a precomputed grad or computes it via function
def produce_map_from_grad_or_compute(
    feature_map: torch.Tensor,
    grad_feat: Optional[torch.Tensor],
    compute_grad_fn,
    method: str,
    img_size: Tuple[int,int] = (224,224),
    **kwargs
) -> torch.Tensor:
    """
    feature_map: (C,H,W)
    grad_feat: (C,H,W) or None
    compute_grad_fn: callable that given a feature_map (C,H,W) or input returns gradient (C,H,W)
    method: one of 'raw_grad','input_x_grad','gradcam','smoothgrad','integrated_gradients'
    returns: upsampled (H_img, W_img) importance map (torch.Tensor)
    """
    if method == 'smoothgrad':
        # compute smoothgrad obtains avg grad -> returns map (H,W)
        map_hw = smoothgrad_feature_map(feature_map, compute_grad_fn, **kwargs)
        return upto_image_size(map_hw, img_size)

    if method == 'integrated_gradients':
        # compute integrated gradients w.r.t. image using compute_grad_fn as a model_forward wrapper
        # compute_grad_fn in this case should be model_forward_fn(image) -> vector/logit
        map_hw = integrated_gradients_image(compute_grad_fn, kwargs['input_image'], kwargs['target_index'],
                                            baseline=kwargs.get('baseline', None), steps=kwargs.get('steps',50),
                                            device=feature_map.device)
        return upto_image_size(map_hw, img_size)

    # other methods require grad_feat; if not provided, compute via compute_grad_fn(feature_map)
    if grad_feat is None:
        grad_feat = compute_grad_fn(feature_map)
    if method == 'raw_grad':
        map_hw = raw_grad_map(grad_feat)
    elif method == 'input_x_grad':
        map_hw = input_x_grad_map(feature_map, grad_feat)
    elif method == 'gradcam':
        map_hw = gradcam_map(feature_map, grad_feat)
    else:
        raise ValueError(f"Unknown method {method}")
    return upto_image_size(map_hw, img_size)
