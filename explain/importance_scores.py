import torch
from torch import Tensor
import numpy as np

import matplotlib.pyplot as plt
import cv2
from typing import Union

from util.glvq import smooth_labels



def compute_subspace_contribution(vh, s):
    """
    returns a (nbatch x n x d) tensor:
    It also shows the coordinate of each region on the reconstructed subspace
    """
    RS_1 = torch.bmm(vh.transpose(-1, -2), torch.diag_embed(1 / s))
    return RS_1


def compute_principal_direction_contribution(RS_1, Q):
    """
    returns a (nbatch x n x d) matrix: each row captures the importance of a region on principal directions
    It also shows the coordinate of each region on the rotated subspace
    """
    return torch.bmm(RS_1, Q)


def project_features_on_prototype(X, rotated_proto):
    """it return W : (nbatch x n x d)"""
    W = torch.bmm(X.transpose(-1, -2), rotated_proto)
    return W


def compute_regionwise_effect(M, W, rel: torch.Tensor):
    """
    X (Dxn) = U(Dxd) S(dxd) Vh(d x n)
    M (nxd) = R(nxd) S^-1(dxd) Q(dxd)
    W (nxd) = X.T(nxD) V (Dxd)
    Result = (nxd) matrix capturing
    """
    nxd_mat = (M * W) @ torch.tile(torch.diag(rel[0]).unsqueeze(0), dims=(M.shape[0], 1, 1))

    # return 1 - nxd_mat.sum(axis=-1) #???????????????????????
    return nxd_mat #, nxd_mat.sum(axis=-1),
    # eps = 0.8
    # log_nxd_mat = torch.log10((-nxd_mat + 2)/(1-nxd_mat+eps))
    # return log_nxd_mat.sum(axis=-1)


def compute_feature_importance(feature_map, 
                              Rt_matrix, 
                              s_matrix, 
                              output_dict,
                              prototypes, 
                              relevances,
                              target_class=None):
    """
    Compute the gradient of the similarity measure w.r.t. input features.
    
    Mathematical formulation:
    -------------------------
    For similarity s_j = exp(β * Λ^T Σ_j), the gradient is:
    
        ∂s_j/∂X = (V_j Λ Q_j^T) (S^{-1} R_j^T)
    
    where:
        - V_j = w_j Q_{w,j} is the rotated prototype
        - Λ is the diagonal relevance matrix
        - Q_j, R_j are rotation matrices from SVD
        - S is the singular value matrix
    
    Args:
        feature_map (torch.Tensor): Feature map from CNN, shape (1, C, H, W)
        Rt_matrix (torch.Tensor): Right singular vectors, shape (1, d, n)
        s_matrix (torch.Tensor): Singular values, shape (1, d)
        output_dict (dict): Contains 'Q', 'Qw' rotation matrices
        prototypes (torch.Tensor): Prototype subspaces, shape (nprotos, D, d)
        relevances (torch.Tensor): Relevance weights, shape (1, d)
        target_class (int, optional): If provided, only compute gradient for this class
    
    Returns:
        torch.Tensor: Gradient importance map, shape (C, H, W) or (nprotos, C, H, W)
    """
    # Validate input - should be batch_size=1
    assert feature_map.shape[0] == 1, f"Expected batch_size=1, got {feature_map.shape[0]}"
    
    batch_size, num_channels, width, height = feature_map.shape
    nprotos = prototypes.shape[0]
    
    # Compute inverse singular values
    S_inv = torch.diag_embed(1 / s_matrix)  # (1, d, d)
    Lamda = torch.diag(relevances[0])  # (d, d)
    H_right = torch.bmm(S_inv, Rt_matrix)  # (1, d, n)
    
    # Since batch_size=1, we can squeeze and work with single sample
    Qt = torch.transpose(output_dict['Q'][0], -1, -2)  # (nprotos, d, n)
    Qw = output_dict['Qw'][0]  # (nprotos, n, d)
    
    # Compute gradient for each prototype
    V = torch.bmm(prototypes, Qw)  # (nprotos, D, d)
    H_left = torch.bmm(torch.matmul(V, Lamda), Qt)  # (nprotos, D, n)
    sim_grad = torch.matmul(H_left, H_right[0])  # (nprotos, D, n)
    
    # Reshape to spatial dimensions
    sim_grad = sim_grad.view(nprotos, num_channels, width, height)  # (nprotos, C, H, W)
    
    # If target_class is specified, return only that class's gradient
    if target_class is not None:
        return sim_grad[target_class]  # (C, H, W)
    
    # Otherwise return all prototypes' gradients
    return sim_grad  # (nprotos, C, H, W)



def save_feature_importance_heatmap(
        effect_map: Union[Tensor, np.array],
        output_path: str,
        colormap: int = cv2.COLORMAP_JET,
        save_raw: bool = False,
) -> np.ndarray:
    """Generate and save heatmap from effect map."""
    
    if isinstance(effect_map, Tensor):
        effect_map_np = effect_map.cpu().detach().numpy()
    else:
        effect_map_np = effect_map
    
    # Normalize to [0, 1]
    rescaled_map = (effect_map_np - np.amin(effect_map_np))
    rescaled_map = rescaled_map / (np.amax(rescaled_map) + 1e-8)
    
    # Save raw values if requested
    if save_raw:
        raw_path = output_path.replace('.png', '_raw.npy')
        np.save(raw_path, rescaled_map)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_map), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    plt.imsave(fname=output_path, arr=heatmap, vmin=0.0, vmax=1.0)
    return heatmap