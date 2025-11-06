import torch
from torch import Tensor
from torch.autograd import Function

from lvq.measures.base_measure import GrassmannMeasureBase


def rotate_data(xs: Tensor, 
                rotation_matrix: Tensor, 
                winner_ids: Tensor = None, 
                return_rotation_matrix: bool = False):
    """
    Rotate the input data based on the rotation matrices.
    Args:
        - xs (torch.Tensor): Input data of shape (batch_size, dim_of_data, dim_of_subspace).
        - rotation_matrix (torch.Tensor): Rotation matrices of shape (batch_size, num_of_prototypes, dim_of_subspace, dim_of_subspace).
        - winner_ids (torch.Tensor): Indices of winner prototypes for each data point, of shape (batch_size, 2).
    Returns: 
        - torch.Tensor or tuple: Rotated data 
    """
    nbatch, D, d = xs.shape

    if winner_ids is None:
        # rotate wrt all prototypes
        return torch.matmul(xs.unsqueeze(1), rotation_matrix)  # (B, P, D, d)

    else:
        # rotate wrt winner prototypes
        Qwinners = rotation_matrix[torch.arange(nbatch).unsqueeze(-1), winner_ids]
        Qwinners1, Qwinners2 = Qwinners[:, 0], Qwinners[:, 1]
        rotated_xs1 = torch.bmm(xs, Qwinners1)
        rotated_xs2 = torch.bmm(xs, Qwinners2)
        if return_rotation_matrix:
            return rotated_xs1, rotated_xs2, Qwinners1, Qwinners2
        return rotated_xs1, rotated_xs2



class AnglePrototypeLayer(Function):
    
    @staticmethod
    def forward(ctx, 
                xs_subspace: Tensor, 
                xprotos: Tensor, 
                relevances: Tensor,
                score_fn: GrassmannMeasureBase):
        """
        Forward pass of the AnglePrototypeLayer.
        Args:
            ctx: Context object to save tensors for backward computation.
            xs_subspace (torch.Tensor): Input subspaces.
            xprotos (torch.Tensor): Prototypes.
            relevances (torch.Tensor): Relevance parameters.
            score_fn: scoring (similaritly/dissimilarity) function on Grassmann manifold
        Returns:
            score (torch.Tensor): The similarity/dissimilarity between the input subspaces and the prototypes
        """
        output = score_fn(xs_subspace, xprotos, relevances)
        rotation_matrices = torch.matmul(output['Q'], output['Qw']) # # Qw has been already transposed (Vh)

        xs_prototype_frame = rotate_data(xs_subspace, rotation_matrices)

        ctx.score_fn = score_fn  # store the measure object
        ctx.save_for_backward(
            xs_prototype_frame, relevances, output['canonical_correlation'])
        
        return output['score'] 
    

    @staticmethod
    def backward(ctx, scores_grad):
        """
        Backward pass of the ChordalPrototypeLayer.
        Args:
            ctx: Context object containing saved tensors from forward pass.
            scores_grad (torch.Tensor): Gradient of the loss wrt the scores (dL/ds or dL/d_d).
        Returns:
            tuple: Gradient wrt prototypes/relevances.
        """
        x_prototype_frame, relevances, canonical_correlation = ctx.saved_tensors

        # Get the analytical gradient from your measure class
        measure = ctx.score_fn        
        ds_dw, ds_drelevances = measure.euclidean_gradient(x_prototype_frame, canonical_correlation, relevances)  # (nbatch, nprotos, D, d), (nbatch, nprotos, d)

        grad_xs, grad_protos, grad_relevances, grad_score_fn = None, None, None, None
        
        grad_protos = scores_grad.unsqueeze(-1).unsqueeze(-1) * ds_dw # (nbatch, embedding_dim, subspace_dim)
        grad_relevances = scores_grad.unsqueeze(-1) * ds_drelevances
        
        return grad_xs, grad_protos, grad_relevances, grad_score_fn

        