import torch
from torch import Tensor


def smooth_labels(labels: Tensor, num_classes: int, epsilon: float = 0.1) -> Tensor:
    """
    Convert integer labels to smoothed probability distributions.

    Args:
        labels (torch.Tensor): Tensor of shape (batch_size,) containing class indices.
        num_classes (int): Total number of classes.
        epsilon (float): Smoothing factor (0 <= ε < 1).

    Returns:
        torch.Tensor: Smoothed label probabilities of shape (batch_size, num_classes).
    """
    assert 0 <= epsilon < 1, "epsilon should be between 0 and 1"

    batch_size = labels.size(0)
    
    # Start with all values = epsilon / (num_classes - 1)
    smooth = torch.full((batch_size, num_classes), epsilon / (num_classes - 1), device=labels.device)
    
    # Assign 1 - epsilon to the correct class
    smooth.scatter_(1, labels.unsqueeze(1).long(), 1.0 - epsilon)
    
    return smooth



def project_to_tangent_single(W: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Project Euclidean gradient G onto tangent space at single W (n_features, k)."""
    return (torch.eye(W.shape[0], device=W.device, dtype=W.dtype) - W @ W.T) @ G


def project_to_tangent(prototypes: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    Vectorized projection for many prototypes.
    prototypes: (n_protos, n_features, k)
    G:          (n_protos, n_features, k)
    returns: projected gradients same shape as G
    """
    # Compute (I - W W^T) G efficiently for each batch
    n_protos, n_features, _ = prototypes.shape
    I = torch.eye(n_features, device=prototypes.device, dtype=prototypes.dtype).expand(n_protos, -1, -1)
    WWt = torch.bmm(prototypes, prototypes.transpose(1, 2))
    proj = torch.bmm(I - WWt, G)
    return proj


def qr_retraction(prototypes_tentative: torch.Tensor) -> torch.Tensor:
    """
    Orthonormalize each prototype (used after W - lr * G_riem).
    prototypes_tentative: (n_protos, n_features, k)
    returns (n_protos, n_features, k)
    """
    Q, _ = torch.linalg.qr(prototypes_tentative)
    return Q


def exp_map_up(W: torch.Tensor, G_riem: torch.Tensor, lr: float) -> torch.Tensor:
    """Exact exponential map on Grassmann manifold for a single prototype."""
    U, S, Vh = torch.linalg.svd(G_riem, full_matrices=False)
    if S.numel() == 0:
        return W

    cos_term = torch.diag_embed(torch.cos(lr * S))
    sin_term = torch.diag_embed(torch.sin(lr * S))
    V = Vh.transpose(-1, -2)

    W_new = W @ (V @ cos_term @ V.transpose(-1, -2)) + U @ sin_term @ V.transpose(-1, -2)
    return W_new


def exp_map_update(W: torch.Tensor, G_riem: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Batched exponential map update.
    W, G_riem: (n_protos, n_features, k)
    """
    n_protos = W.shape[0]
    W_new = []

    for i in range(n_protos):
        W_new.append(exp_map_up(W[i], G_riem[i], lr))

    return torch.stack(W_new, dim=0)
