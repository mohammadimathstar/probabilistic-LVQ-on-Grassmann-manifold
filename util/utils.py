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