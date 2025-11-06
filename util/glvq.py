import torch
from torch import nn, Tensor
import argparse
from typing import Callable
from sklearn.metrics import confusion_matrix, accuracy_score  # f1_score


def metrics(y_true: Tensor, y_pred: Tensor, nclasses):

    assert y_true.shape == y_pred.shape, f'their shape is labels: {y_true.shape}, pred:{y_pred.shape}'

    acc = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    c = confusion_matrix(
        y_true.cpu().numpy(), y_pred.cpu().numpy(),
        labels=range(nclasses),
        # normalize='true'
    )
    return 100 * acc, c


def compute_classification_metrics(y_true: Tensor, y_pred: Tensor, nclasses: int) -> tuple:
    """
    Compute classification accuracy and confusion matrix.

    Parameters:
    -----------
    y_true : Tensor
        True class labels, a tensor of shape (num_samples,).
    y_pred : Tensor
        Predicted class labels, a tensor of shape (num_samples,).
    nclasses : int
        Number of classes.

    Returns:
    --------
    tuple
        A tuple containing:
        - accuracy (float): Classification accuracy in percentage.
        - confusion_matrix (ndarray): Confusion matrix of shape (nclasses, nclasses).

    Raises:
    -------
    AssertionError
        If the shapes of `y_true` and `y_pred` do not match.
    """
    assert y_true.shape == y_pred.shape, f'Shape mismatch: labels: {y_true.shape}, predictions: {y_pred.shape}'

    # Convert tensors to numpy arrays
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # Compute accuracy
    accuracy = accuracy_score(y_true_np, y_pred_np)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_np, y_pred_np, labels=range(nclasses))

    return 100 * accuracy, conf_matrix



def winner_prototype_indices(
        ydata: Tensor,
        yprotos_mat: Tensor,
        distances: Tensor
) -> Tensor:
    """
    Find the closest prototypes to a batch of features.

    Parameters:
    -----------
    ydata : Tensor
        Labels of input images, shape (batch_size,).
    yprotos_mat : Tensor
        Labels of prototypes, shape (nclass, number_of_prototypes).
        This can be used for prototypes with the same or different labels (W^+ and W^-).
    distances : Tensor
        Distances between images and prototypes, shape (batch_size, number_of_prototypes).

    Returns:
    --------
    Tensor
        A tensor containing the indices of the winner prototypes for each image in the batch.
    """
    assert distances.ndim == 2, (f"Distances should be a matrix of shape (batch_size, number_of_prototypes), "
                                 f"but got {distances.shape}")

    # Generate a mask for the prototypes corresponding to each image's label
    mask = yprotos_mat[ydata]
    # Y = yprotos_mat[ydata.tolist()]

    # Apply the mask to distances
    distances_sparse = distances * mask
    # distances_sparse = distances * Y

    # Find the index of the closest prototype for each image
    winner_indices = torch.stack(
        [
            torch.argwhere(w).T[0,
            torch.argmin(
                w[torch.argwhere(w).T],
            )
            ] for w in torch.unbind(distances_sparse)
        ], dim=0
    ).T

    return winner_indices

