import torch
from torch import Tensor
from torch.nn import ReLU
import numpy as np

import matplotlib.pyplot as plt
import cv2
from typing import Union

# from util.glvq import IdentityLoss
relu = ReLU()


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


def project_features_on_prototype(X, rotated_proto):
    """it return W : (nbatch x n x d)"""
    W = torch.matmul(X.transpose(-1, -2), rotated_proto)
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


def compute_feature_importance(feature_map, label,
                              Rt, S, output_dict,
                              prototype_features, 
                            #   proto_labels_matrix, proto_complementary_labels_matrix,
                              relevances,
                              return_full_output=False):
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
        return_full_output (bool): If True, return additional intermediate results.

    Returns:
        torch.Tensor: Importance of each region affecting the model's decision.
        torch.Tensor: Principal direction effect per region.
    """
    _, num_channels, height , width = feature_map.shape

    # Retrieve distance and initialize loss
    feature_map = feature_map.squeeze(0)
    Rt = Rt.squeeze(0)
    # s_matrix = s_matrix.squeeze(0)
    scores = output_dict['score'].squeeze(0)
    Q = output_dict['Q'].squeeze(0)
    Qw = output_dict['Qw'].squeeze(0)

    winner_idx = torch.argmax(scores)
    positive_idx = label.item()
    values, indices = torch.topk(scores, k=2)
    negative_idx = indices[1]

    reshaped_features = feature_map.view(num_channels, width * height)

    # Store winning prototype data
    positive_prototype = {'index': positive_idx,
                          'Q': Q[positive_idx],
                          'Qw': Qw[positive_idx]}
    negative_prototype = {'index': negative_idx,
                          'Q': Q[negative_idx],
                          'Qw': Qw[negative_idx]}

    # Classification status check
    classification_status = 'misclassified' if winner_idx != positive_idx else 'correct'
    print(f"\t\t\tpositive_idx: {positive_idx}, \
              \t\t\t\twinner_idx: {winner_idx}")

    # Process both positive and negative prototypes
    prototype_types = ['positive', 'negative']
    prototypes = [positive_prototype, negative_prototype]

    rotated_prototypes = {}
    effect_dict = {}

    from explain_old.explain_utils import raw_grad_map, input_x_grad_map, gradcam_map

    for proto_type, prototype in zip(prototype_types, prototypes):
        # Compute contributions to subspaces
        reconstructed_subspace = compute_subspace_contribution(Rt, S)
        principal_direction_contribution = compute_principal_direction_contribution(reconstructed_subspace,
                                                                                    prototype['Q'])

        # Rotate prototypes
        rotated_prototype = prototype_features[prototype['index']] @ prototype['Qw']

        
        grad_sim = rotated_prototype @ rel_matrix(relevances) @ principal_direction_contribution

        # Project image features onto prototype subspace
        # reshaped_features = feature_map.view(num_channels, width * height)
        # projection = project_features_on_prototype(reshaped_features, rotated_prototype)
        

        # Calculate effect of hidden regions on prototypes
        # regional_effects = compute_regionwise_effect(principal_direction_contribution,
        #                                             projection, relevances)
        # print("rotated_prototype", rotated_prototype.shape, rel_matrix(relevances).shape)
        # print("grad", grad_sim.shape, reshaped_features.shape)

        # only grad
        # regional_effects = grad_sim.view(num_channels, height, width).sum(dim=0)

        # normalized grad
        # regional_effects = raw_grad_map(
        #     grad_sim.view(num_channels, height, width)
        # )

        # input x grad
        regional_effects = input_x_grad_map(
            feature_map, 
            grad_sim.view(num_channels, height, width),   
            # torch.permute(grad_sim.view(num_channels, height, width), dims=(0, 2, 1))         
        )
        

        # gradcam
        # regional_effects = gradcam_map(
        #     feature_map, 
        #     grad_sim.view(num_channels, height, width),   
        #     relu = True,         
        # )

        # Reshape back to spatial dimensions
        # regional_effects = torch.reshape(
        #     torch.permute(regional_effects, dims=(0, 2, 1)),##### CHECK
        #     (-1, width, height)
        # )

        # Store results
        effect_dict[proto_type] = {
            'regional_effects_on_principal_directions': regional_effects
        }
        rotated_prototypes[proto_type] = rotated_prototype

    # Calculate difference in regional effects between positive and negative prototypes
    region_effect_per_principal_direction = (
            effect_dict['positive']['regional_effects_on_principal_directions'] - effect_dict['negative']['regional_effects_on_principal_directions']
    ).squeeze()

    
    print(region_effect_per_principal_direction.shape, feature_map.shape)
    region_effect_on_decision = region_effect_per_principal_direction #, feature_map[0])

    return region_effect_on_decision, region_effect_per_principal_direction



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
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]

    plt.imsave(fname=output_path, arr=heatmap, vmin=0.0, vmax=1.0)
    return heatmap