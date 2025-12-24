"""
explain/attribution.py

Core logic for computing feature importance and gradient similarity in the Grassmann LVQ model.
"""

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Union, Dict, Any, Tuple, List

from explain.regions.utils import raw_grad_map, input_x_grad_map, gradcam_map


def get_relevance_matrix(relevances: torch.Tensor) -> torch.Tensor:
    """
    Convert a relevance vector to a diagonal matrix.
    """
    return torch.diag(relevances[0])


def compute_subspace_contribution(Rt: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Compute the contribution of the reconstructed subspace.
    Returns a (d x n) tensor representing coordinates on the reconstructed subspace.
    """
    Sinv = torch.diag(1 / s[0])
    return torch.matmul(Sinv, Rt)


def compute_principal_direction_contribution(subspace_contribution: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Compute the importance of each region on the principal directions.
    """
    return Q.T @ subspace_contribution


def compute_grad_sim(feature_map: torch.Tensor, 
                     reconstructed_subspace: torch.Tensor, 
                     prototype_info: Dict[str, Any], 
                     prototype_features: torch.Tensor, 
                     relevances: torch.Tensor, 
                     args: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient similarity for a given prototype and relevances.
    
    Args:
        feature_map: (1, C, H, W) tensor
        reconstructed_subspace: Contribution from the subspace reconstruction
        prototype_info: Dictionary containing 'Q', 'Qw', and 'index' for the prototype
        prototype_features: All prototype feature vectors
        relevances: Relevance vector for principal directions
        args: Configuration arguments
        
    Returns:
        Tuple of (regional_effects, rotated_prototype)
    """
    _, num_channels, height, width = feature_map.shape
    
    # Contribution of regions to principal directions
    principal_direction_contribution = compute_principal_direction_contribution(
        reconstructed_subspace, prototype_info['Q']
    )
    
    # Rotate prototype into the subspace
    rotated_prototype = prototype_features[prototype_info['index']] @ prototype_info['Qw']
    
    # Gradient similarity: how much each region aligns with the prototype in the weighted subspace
    grad_sim = rotated_prototype @ get_relevance_matrix(relevances) @ principal_direction_contribution
    
    # Map to spatial dimensions using the selected method
    method = args.explain_method
    if method == 'raw_grad':        
        regional_effects = raw_grad_map(grad_sim.view(num_channels, height, width))
    elif method == 'input_x_grad':
        regional_effects = input_x_grad_map(
            feature_map.squeeze(0), 
            grad_sim.view(num_channels, height, width),   
        )
    elif method == 'gradcam':
        regional_effects = gradcam_map(
            feature_map.squeeze(0), 
            grad_sim.view(num_channels, height, width),   
            relu=True,
        )
    else:
        raise ValueError(f"Unknown explanation method: {method}")
    
    return regional_effects, rotated_prototype


def compute_feature_importance(feature_map: torch.Tensor, 
                               label: torch.Tensor,
                               Rt: torch.Tensor, 
                               S: torch.Tensor, 
                               output_dict: Dict[str, Any],
                               prototype_features: torch.Tensor, 
                               relevances: torch.Tensor, 
                               k_negatives: int, 
                               args: Any,
                               print_info: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the importance of each spatial region affecting the model's decision.
    Calculates the difference between positive prototype effect and mean negative prototype effects.
    """
    assert feature_map.shape[0] == 1, "Batch size must be 1"
    
    # Ensure relevances is 2D (1, D)
    if relevances.dim() == 1:
        relevances = relevances.unsqueeze(0)

    Rt = Rt.squeeze(0)
    scores = output_dict['score'].squeeze(0)
    Q = output_dict['Q'].squeeze(0)
    Qw = output_dict['Qw'].squeeze(0)

    # Identify positive and top negative prototypes
    positive_idx = label.item()
    winner_idx = torch.argmax(scores)
    
    _, top_indices = torch.topk(scores, k=min(k_negatives + 1, scores.shape[0]))
    negative_indices = [idx.item() for idx in top_indices if idx.item() != positive_idx][:k_negatives]
    
    if print_info:
        print(f"\t\t\tPositive Index: {positive_idx}, Winner Index: {winner_idx}")
        if negative_indices:
            print(f"\t\t\tNegative Indices (top {len(negative_indices)}): {negative_indices}")

    # Prepare prototype data
    positive_proto_info = {'index': positive_idx, 'Q': Q[positive_idx], 'Qw': Qw[positive_idx]}
    negative_protos_info = [{'index': idx, 'Q': Q[idx], 'Qw': Qw[idx]} for idx in negative_indices]

    # Compute subspace contribution once
    reconstructed_subspace = compute_subspace_contribution(Rt, S)
    
    # 1. Positive Prototype Attribution
    regional_effects_pos, rotated_prototype_pos = compute_grad_sim(
        feature_map, reconstructed_subspace, positive_proto_info, 
        prototype_features, relevances, args
    )
    
    # 2. Negative Prototypes Attribution
    negative_effects = []
    for neg_info in negative_protos_info:
        effects, _ = compute_grad_sim(
            feature_map, reconstructed_subspace, neg_info, 
            prototype_features, relevances, args
        )
        negative_effects.append(effects)
        
    # Final Decision Attribution: Positive - Mean(Negatives)
    if negative_effects:        
        mean_negative_effect = torch.stack(negative_effects).mean(dim=0)
        decision_importance = (regional_effects_pos - mean_negative_effect).squeeze()
    else:
        decision_importance = regional_effects_pos.squeeze()

    return decision_importance, rotated_prototype_pos


def save_importance_heatmap(effect_map: Union[Tensor, np.ndarray], output_path: str) -> np.ndarray:
    """
    Normalize an importance map and save it as a JET heatmap image.
    """
    if isinstance(effect_map, Tensor):
        effect_map_np = effect_map.detach().cpu().numpy()
    else:
        effect_map_np = effect_map
        
    # Min-max normalization
    rescaled = (effect_map_np - np.amin(effect_map_np))
    rescaled = rescaled / (np.amax(rescaled) + 1e-8)

    # Apply JET colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # BGR to RGB

    plt.imsave(fname=output_path, arr=heatmap, vmin=0.0, vmax=1.0)
    return heatmap
