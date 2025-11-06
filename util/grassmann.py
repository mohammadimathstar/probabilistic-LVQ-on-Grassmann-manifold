import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union


def orthogonalize_batch(x_batch: Tensor) -> Tensor:
    """
    Orthogonalize each matrix in a batch of matrices using QR decomposition.

    Parameters:
    -----------
    x_batch : Tensor
        Input tensor of shape (batch_size, m, n), where each (m, n) matrix will be orthogonalized.

    Returns:
    --------
    Tensor
        A tensor of orthogonal matrices of shape (batch_size, m, min(m, n)), where each matrix in the batch is orthogonalized.

    """
    Q, _ = torch.linalg.qr(x_batch, mode='reduced')
    return Q


def grassmann_repr(batch_imgs: Tensor, dim_of_subspace: int) -> Tensor:
    """
    Generate Grassmann representations from a batch of images.

    Parameters:
    -----------
    batch_imgs : Tensor
        A batch of features of size (batch_size, num_of_channels, W, H).
    dim_of_subspace : int
        The dimensionality of the extracted subspace.

    Returns:
    --------
    Tensor
        An orthonormal matrix of size (batch_size, W*H, dim_of_subspace) representing the Grassmann subspaces.

    Raises:
    -------
    AssertionError
        If the input tensor `batch_imgs` does not have 4 dimensions.
    """
    assert batch_imgs.ndim == 4, f"batch_imgs should be of shape (batch_size, num_of_channels, W, H), but it is {batch_imgs.shape}"

    bsize, nchannel, w, h = batch_imgs.shape
    # Reshape the batch images to shape (batch_size, num_of_channels, W*H)
    xs = batch_imgs.view(bsize, nchannel, w * h)

    try:
        U, S, Vh = torch.linalg.svd(xs, full_matrices=False)
    except:
        eps = 1e-2
        batch_eye = torch.eye(nchannel, w * h).unsqueeze(0).repeat(bsize, 1, 1).to(xs.device)
        U, S, Vh = torch.linalg.svd(xs + batch_eye * eps, full_matrices=False)

    # Select the appropriate orthonormal matrix based on dimensions
    if U.shape[-2] > U.shape[-1]:
        return U[:, :, :dim_of_subspace]  # Shape: (batch_size, num_of_channels, dim_of_subspace)
    else:
        return Vh.transpose(-1, -2)[:, :, :dim_of_subspace]  # Shape: (batch_size, W*H, dim_of_subspace)


def grassmann_repr_full(batch_imgs: Tensor, dim_of_subspace: int) -> Tensor:
    """
    Generate Grassmann representations from a batch of images.
    It returns both singular values and left/right principal directions.

    Parameters:
    -----------
    batch_imgs : Tensor
        A batch of features of size (batch_size, num_of_channels, W, H).
    dim_of_subspace : int
        The dimensionality of the extracted subspace.

    Returns:
    --------
    Tensor
        An diagonal matrix of size (batch_size, dim_of_subspace, dim_of_subspace) containing singular values.
        Two orthonormal matrices of size (batch_size, W*H, dim_of_subspace) representing left/right principal directions.

    Raises:
    -------
    AssertionError
        If the input tensor `batch_imgs` does not have 4 dimensions.
    """
    assert batch_imgs.ndim == 4, f"batch_imgs should be of shape (batch_size, num_of_channels, W, H), but it is {batch_imgs.shape}"

    bsize, nchannel, w, h = batch_imgs.shape
    # Reshape the batch images to shape (batch_size, num_of_channels, W*H)
    xs = batch_imgs.view(bsize, nchannel, w * h)

    # SVD: generate principal directions
    U, S, Vh = torch.linalg.svd(xs, full_matrices=False)

    assert U.shape[-2] > U.shape[-1], f"The matrix size is {U.shape[1:]}."

    # Shape: (batch_size, num_of_channels, dim_of_subspace)
    return U[:, :, :dim_of_subspace], Vh[:, :dim_of_subspace,:], S[:, :dim_of_subspace]


def init_randn(
        dim_of_data: int,
        dim_of_subspace: int,
        labels: Tensor = None,
        num_of_protos: [int, Tensor] = 1,
        num_of_classes: int = None,
        device='cpu'
) -> tuple:
    """
    Initialize prototypes randomly using a Gaussian distribution.

    Parameters:
    -----------
    dim_of_data : int
        Dimensionality of the data space.
    dim_of_subspace : int
        Dimensionality of the subspace.
    labels : Tensor, optional
        Tensor containing class labels. If None, `num_of_classes` must be provided.
    num_of_protos : int or Tensor, optional
        Number of prototypes per class if an integer, or a tensor specifying the number of prototypes for each class.
        Default is 1.
    num_of_classes : int, optional
        Number of classes. Required if `labels` is None.
    device : str, optional
        Device on which to place the tensors. Default is 'cpu'.

    Returns:
    --------
    tuple
        A tuple containing:
        - xprotos (Tensor): Initialized prototypes of shape (total_num_of_protos, dim_of_data, dim_of_subspace).
        - yprotos (Tensor): Labels of the prototypes of shape (total_num_of_protos,).
        - yprotos_mat (Tensor): One-hot encoded labels of the prototypes of shape (nclass, total_num_of_protos).
        - yprotos_mat_comp (Tensor): Complementary one-hot encoded labels of the prototypes of shape (nclass, total_num_of_protos).
    """
    if labels is None:
        assert num_of_classes is not None, "num_of_classes must be provided if labels are not given."
        classes = torch.arange(num_of_classes)
    else:
        classes = torch.unique(labels)

    if isinstance(num_of_protos, int):
        total_num_of_protos = len(classes) * num_of_protos
    else:
        total_num_of_protos = torch.sum(num_of_protos).item()

    nclass = len(classes)
    prototype_shape = (total_num_of_protos, dim_of_data, dim_of_subspace)

    # Initialize prototypes using QR decomposition for orthogonalization
    Q, _ = torch.linalg.qr(
        # 0.5 + 0.1 * torch.randn(prototype_shape, device=device),
        0.1 * torch.randn(prototype_shape, device=device),
        mode='reduced')
    xprotos = nn.Parameter(Q)

    # Set prototypes' labels
    yprotos = torch.repeat_interleave(classes, num_of_protos).to(torch.int32)

    yprotos_mat = torch.zeros((nclass, total_num_of_protos), dtype=torch.int32)
    yprotos_mat_comp = torch.ones((nclass, total_num_of_protos), dtype=torch.int32, device=device)

    # Setting prototypes' labels
    for i, class_label in enumerate(yprotos):
        yprotos_mat[class_label, i] = 1
        yprotos_mat_comp[class_label, i] = 0

    return xprotos, yprotos.to(device), yprotos_mat.to(device), yprotos_mat_comp.to(device)



def compute_measurement_on_grassmann_mdf(
        xdata: Tensor,
        xprotos: Tensor,
        relevance: Tensor,
) -> dict:
    """
    Compute the (geodesic or chordal) distances between an input subspace and all prototypes.

    Parameters:
    -----------
    xdata : Tensor
        Input tensor representing the subspaces, expected shape (batch_size, W*H, dim_of_subspace).
    xprotos : Tensor
        Prototype tensor representing the prototype subspaces, expected shape (num_of_prototypes, W*H, dim_of_subspace).
    relevance : Tensor
        Tensor representing the relevance of each dimension in the subspace, expected shape (1, dim_of_subspace).

    Returns:
    --------
    dict
        A dictionary containing:
        - 'Q' : Left singular vectors (U), shape (batch_size, num_of_prototypes, WxH, dim_of_subspaces).
        - 'Qw' : Right singular vectors (Vh), shape (batch_size, num_of_prototypes, WxH, dim_of_subspaces).
        - 'canonicalcorrelation' : Singular values (S), shape (batch_size, num_of_prototypes, dim_of_subspaces).
        - 'measure' : Computed distances/similarity, shape (batch_size, num_of_prototypes).

    Raises:
    -------
    Exception
        If any NaN values are encountered in the distance computation.
    """
    assert xdata.ndim == 3, f"xdata should be of shape (batch_size, W*H, dim_of_subspace), but it is {xdata.shape}"

    xdata = xdata.unsqueeze(dim=1)  # Shape: (batch_size, 1, W*H, dim_of_subspace)

    # Compute the singular value decomposition of the product of the transposed xdata and xprotos
    U, S, Vh = torch.linalg.svd(
        torch.transpose(xdata, 2, 3) @ xprotos.to(xdata.dtype),
        full_matrices=False,
    )

    measure = torch.transpose(
        relevance @ torch.transpose(S, 1, 2).to(relevance.dtype),
        1, 2
    )
    
    if torch.isnan(measure).any():
        raise Exception('Error: NaN values! Using the --log_probabilities flag might fix this issue')

    output = {
        'Q': U,  # Shape: (batch_size, num_of_prototypes, WxH, dim_of_subspaces)
        'Qw': torch.transpose(Vh, 2, 3),  # Shape: (batch_size, num_of_prototypes, WxH, dim_of_subspaces)
        'canonicalcorrelation': S,  # Shape: (batch_size, num_of_prototypes, dim_of_subspaces)
        'measure': torch.squeeze(measure, -1),  # Shape: (batch_size, num_of_prototypes)
    }

    return output


def prediction(
        data: Tensor,
        xprotos: Tensor,
        yprotos: Tensor,
        lamda: Tensor,
) -> Tensor:
    """
    Predict the class labels for input data based on distances to prototype subspaces.

    Parameters:
    -----------
    data : Tensor
        Input tensor representing the subspaces, expected shape (batch_size, W*H, dim_of_subspace).
    xprotos : Tensor
        Prototype tensor representing the prototype subspaces, expected shape (num_of_prototypes, W*H, dim_of_subspace).
    yprotos : Tensor
        Tensor containing the class labels for each prototype, expected shape (num_of_prototypes,).
    lamda : Tensor
        Tensor representing the relevance of each dimension in the subspace, expected shape (1, dim_of_subspace).
    metric_type : Union['geodesic', 'chordal']
        Type of distance metric to use, either 'chordal' or 'geodesic'.

    Returns:
    --------
    Tensor
        A tensor containing the predicted class labels for the input data, shape (batch_size,).

    Raises:
    -------
    Exception
        If any NaN values are encountered in the distance computation.
    """
    # Compute distances between data and prototypes using the specified metric type and relevance
    results = compute_measurement_on_grassmann_mdf(
        data, xprotos,
        relevance=lamda
    )

    # Select the prototype with the minimum distance for each input in the batch
    predictions = yprotos[results['measure'].argmax(axis=1)]

    return predictions



