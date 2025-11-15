from lvq.measures.base_measure import GrassmannMeasureBase
from torch import Tensor
import torch


class AngleMeasure(GrassmannMeasureBase):
    def __init__(self, beta: float = 1.0):
        self._beta = torch.tensor(beta)
    
    def compute(self, singular_values: Tensor, relevances: Tensor) -> Tensor:
        scores = torch.transpose(
            relevances @ torch.transpose(singular_values, 1, 2).to(relevances.dtype),
            1, 2
        )
        return (torch.exp( self._beta * scores))
        # return scores

    def euclidean_gradient(self, rotated_data: Tensor, prototype_rotated: Tensor, canonical_corr: Tensor, relevance: Tensor) -> Tensor:
        """Compute (Euclidean) gradient of score (similarity/distance measure) wrt prototypes."""
        ds_dw = relevance.unsqueeze(0) * rotated_data # (nbatch, embedding_dim, subspace_dim) 
        ds_dx = relevance.unsqueeze(0) * prototype_rotated # (nbatch, embedding_dim, subspace_dim)
        ds_drel = canonical_corr # (nbatch, nprotos, subspace_dim)
        return ds_dw, ds_dx, ds_drel