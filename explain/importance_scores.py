import torch
from torch import Tensor
import numpy as np

import matplotlib.pyplot as plt
import cv2
from typing import Union

from util.glvq import IdentityLoss


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


def compute_feature_importance(feature_map, label,
                              vh_matrix, s_matrix, output_dict,
                              prototype_features, proto_labels_matrix, proto_complementary_labels_matrix,
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
        proto_labels_matrix (torch.Tensor): Prototype labels in a matrx of size (nclass x nprotos).
        proto_complementary_labels_matrix (torch.Tensor): Complementary prototype labels in a matrix of size (nclass x nprotos)
        relevances (torch.Tensor): Relevance of different principal direction in the feature space.
        return_full_output (bool): If True, return additional intermediate results.

    Returns:
        torch.Tensor: Importance of each region affecting the model's decision.
        torch.Tensor: Principal direction effect per region.
    """
    batch_size, num_channels, width, height = feature_map.shape

    # Retrieve distance and initialize loss
    distance_matrix = output_dict['distance']
    loss = IdentityLoss()

    # Compute prototype match and mismatch indices
    _, positive_idx, negative_idx = loss(label,
                                         proto_labels_matrix,
                                         proto_complementary_labels_matrix,
                                         distance_matrix)

    # Extract distances of positive and negative prototypes
    positive_distance, negative_distance = distance_matrix[0, positive_idx], distance_matrix[0, negative_idx]

    # Store winning prototype data
    positive_prototype = {'index': positive_idx,
                          'Q': output_dict['Q'][0, positive_idx],
                          'Qw': output_dict['Qw'][0, positive_idx]}
    negative_prototype = {'index': negative_idx,
                          'Q': output_dict['Q'][0, negative_idx],#########
                          'Qw': output_dict['Qw'][0, negative_idx]}

    # Classification status check
    classification_status = 'misclassified' if positive_distance > negative_distance else 'correct'
    print(f"\t\t\t{classification_status}: \t  {(positive_distance - negative_distance).item()}, \
              normalized:    {((positive_distance - negative_distance) / (positive_distance + negative_distance)).item()}")
    print(f"\t\t\tpositive_idx: {positive_idx.item()}, \
              \t\t\t\tnegative_idx: {negative_idx.item()}")

    # Process both positive and negative prototypes
    prototype_types = ['positive', 'negative']
    prototypes = [positive_prototype, negative_prototype]

    rotated_prototypes = {}
    effect_dict = {}

    for proto_type, prototype in zip(prototype_types, prototypes):
        # Compute contributions to subspaces
        reconstructed_subspace = compute_subspace_contribution(vh_matrix, s_matrix)
        principal_direction_contribution = compute_principal_direction_contribution(reconstructed_subspace,
                                                                                    prototype['Q'])

        # Rotate prototypes
        rotated_prototype = prototype_features[prototype['index']] @ prototype['Qw']

        # Project image features onto prototype subspace
        reshaped_features = feature_map.view(batch_size, num_channels, width * height)
        projection = project_features_on_prototype(reshaped_features, rotated_prototype)

        # Calculate effect of hidden regions on prototypes
        regional_effects = compute_regionwise_effect(principal_direction_contribution,
                                                    projection, relevances)

        # Reshape back to spatial dimensions
        regional_effects = torch.reshape(
            torch.permute(regional_effects, dims=(0, 2, 1)),
            (batch_size, -1, width, height)
        )

        # Store results
        effect_dict[proto_type] = {
            'regional_effects_on_principal_directions': regional_effects
        }
        rotated_prototypes[proto_type] = rotated_prototype

    # Calculate difference in regional effects between positive and negative prototypes
    region_effect_per_principal_direction = (
            effect_dict['positive']['regional_effects_on_principal_directions'] -
            effect_dict['negative']['regional_effects_on_principal_directions']
    ).squeeze()

    region_effect_on_decision = region_effect_per_principal_direction.sum(axis=-3)

    # Return results
    if return_full_output:
        return (region_effect_on_decision, region_effect_per_principal_direction,
                effect_dict['positive']['regional_effects_on_principal_directions'],
                effect_dict['negative']['regional_effects_on_principal_directions'])
    else:
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

