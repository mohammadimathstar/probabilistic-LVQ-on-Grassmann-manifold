import torch
from torch import Tensor
from torch.nn import ReLU
import numpy as np

import matplotlib.pyplot as plt
import cv2
from typing import Union

from explain.explain_utils import input_x_grad_map, gradcam_map, raw_grad_map



def rel_matrix(rel: torch.Tensor):
    return torch.diag(rel[0])

def compute_subspace_contribution(Rt, s):
    """
    returns a (d x n) tensor:
    It also shows the coordinate of each region on the reconstructed subspace
    """
    
    Sinv = torch.diag(1 / s[0])
    Sinv_Rt = torch.matmul(Sinv, Rt)    
    return Sinv_Rt


def compute_principal_direction_contribution(Sinv_Rt, Q):
    """
    returns a (d x n) matrix: each row captures the importance of a region on principal directions
    It also shows the coordinate of each region on the rotated subspace
    """
    return Q.T @ Sinv_Rt



# def compute_region_influence(reconstructed_subspace, Q, 
#                              prototype, Qw, relevances, 
#                              args):
#     principal_direction_contribution = compute_principal_direction_contribution(reconstructed_subspace, Q)
#     rotated_prototype = prototype @ Qw
#     grad_sim = rotated_prototype @ rel_matrix(relevances) @ principal_direction_contribution
    
#     if args.explain_method == 'raw_grad':        
#         regional_effects_pos = raw_grad_map(grad_sim.view(num_channels, height, width))
#     elif args.explain_method == 'input_x_grad':
#         regional_effects_pos = input_x_grad_map(
#             feature_map, 
#             grad_sim.view(num_channels, height, width),   
#         )
#     elif args.explain_method == 'gradcam':
#         regional_effects_pos = gradcam_map(
#             feature_map, 
#             grad_sim.view(num_channels, height, width),   
#             relu=True,
#         )
#     else:
#         raise ValueError(f"Unknown method: {args.method}")


def compute_feature_importance(feature_map, label,
                              Rt, S, output_dict,
                              prototype_features, 
                              relevances, k_negatives, args):
    """
    Compute the importance of each spatial region in the last layer of a feature extractor.
    For example, in ResNet50, it returns a (batch_size x 7 x 7) tensor, where 7x7 represents
    the spatial dimensions of the final convolutional layer.

    Args:
        feature_map (torch.Tensor): Feature map from the model's last convolutional layer.
        label (torch.Tensor): Ground truth label for the input sample.
        vh_matrix (torch.Tensor): Matrix representing directions of the principal subspace.
        s_matrix (torch.Tensor): Singular value matrix.
        output_dict (dict): Dictionary containing model outputs like distances and projection values.
        prototype_features (torch.Tensor): Prototypical feature vectors.
        relevances (torch.Tensor): Relevance of different principal direction in the feature space.
        k_negatives (int): Number of negative prototypes to consider

    Returns:
        torch.Tensor: Importance of each region affecting the model's decision.
        torch.Tensor: Principal direction effect per region.
    """

    # Ensure batch size is 1, as we squeeze dim 0
    assert feature_map.shape[0] == 1, "Batch size must be 1 for this function"
    _, num_channels, height , width = feature_map.shape

    # Ensure relevances is 2D (1, D) if it's 1D (D)
    if relevances.dim() == 1:
        relevances = relevances.unsqueeze(0)

    # Retrieve distance and initialize loss
    feature_map = feature_map.squeeze(0)
    Rt = Rt.squeeze(0)
    
    
    scores = output_dict['score'].squeeze(0)
    Q = output_dict['Q'].squeeze(0)
    Qw = output_dict['Qw'].squeeze(0)

    # Get top k+1 to ensure we have enough negatives even if positive is in top k
    values, indices = torch.topk(scores, k=k_negatives + 1)
    
    positive_idx = label.item()
    winner_idx = torch.argmax(scores)

    # Filter out positive index to get negative indices
    negative_indices = [idx.item() for idx in indices if idx.item() != positive_idx]
    # Take top k_neg
    negative_indices = negative_indices[:k_negatives]
    
    # Classification status check
    print(f"\t\t\tpositive_idx: {positive_idx}, \
              \t\t\t\twinner_idx: {winner_idx}")
    print(f"\t\t\tNegative indices (top {len(negative_indices)}): {negative_indices}")

    
    # Store winning prototype data
    positive_prototype = {'index': positive_idx,
                          'Q': Q[positive_idx],
                          'Qw': Qw[positive_idx]}
    
    negative_prototypes = []
    for neg_idx in negative_indices:
        negative_prototypes.append({
            'index': neg_idx,
            'Q': Q[neg_idx],
            'Qw': Qw[neg_idx]
        })

    # Process positive prototype
    effect_dict = {'positive': None, 'negative': [], 'negative_3d': []}
    
    # 1. Positive Prototype
    reconstructed_subspace = compute_subspace_contribution(Rt, S)
    principal_direction_contribution = compute_principal_direction_contribution(reconstructed_subspace,
                                                                                positive_prototype['Q'])
    rotated_prototype_pos = prototype_features[positive_prototype['index']] @ positive_prototype['Qw']
    grad_sim = rotated_prototype_pos @ rel_matrix(relevances) @ principal_direction_contribution
    
    if args.explain_method == 'raw_grad':        
        regional_effects_pos = raw_grad_map(grad_sim.view(num_channels, height, width))
    elif args.explain_method == 'input_x_grad':
        regional_effects_pos = input_x_grad_map(
            feature_map, 
            grad_sim.view(num_channels, height, width),   
        )
    elif args.explain_method == 'gradcam':
        regional_effects_pos = gradcam_map(
            feature_map, 
            grad_sim.view(num_channels, height, width),   
            relu=True,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    effect_dict['positive'] = regional_effects_pos
    
    # 2. Negative Prototypes
    for neg_proto in negative_prototypes:
        principal_direction_contribution = compute_principal_direction_contribution(reconstructed_subspace,
                                                                                    neg_proto['Q'])
        rotated_prototype = prototype_features[neg_proto['index']] @ neg_proto['Qw']
        grad_sim = rotated_prototype @ rel_matrix(relevances) @ principal_direction_contribution
        
        if args.explain_method == 'raw_grad':
            regional_effects_neg = raw_grad_map(grad_sim.view(num_channels, height, width))
        elif args.explain_method == 'input_x_grad':
            regional_effects_neg = input_x_grad_map(
                feature_map, 
                grad_sim.view(num_channels, height, width),   
            )
        elif args.explain_method == 'gradcam':
            regional_effects_neg = gradcam_map(
                feature_map, 
                grad_sim.view(num_channels, height, width),   
                relu=True,
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")        

        effect_dict['negative'].append(regional_effects_neg)
        

    # Calculate difference: Positive - Mean(Negatives)
    if len(effect_dict['negative']) > 0:        
        negative_effect = torch.stack(effect_dict['negative']).sum(dim=0)
        region_effect_on_decision = (effect_dict['positive'] - negative_effect).squeeze()
    else:
        region_effect_on_decision = effect_dict['positive'].squeeze()

    return region_effect_on_decision, rotated_prototype_pos 



def save_feature_importance_heatmap(
        effect_map: Union[Tensor, np.array],
        output_path: str,
) -> np.ndarray:
    """Generate and save heatmap from effect map."""

    if isinstance(effect_map, Tensor):
        effect_map_np = effect_map.cpu().numpy()
    else:
        effect_map_np = effect_map
    rescaled_map = (effect_map_np - np.amin(effect_map_np))
    rescaled_map = rescaled_map / (np.amax(rescaled_map) + 1e-8)

    # Apply colormap to create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]

    plt.imsave(fname=output_path, arr=heatmap, vmin=0.0, vmax=1.0)
    return heatmap


