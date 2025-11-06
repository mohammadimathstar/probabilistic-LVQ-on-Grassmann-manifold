import torch
from torch import Tensor
from abc import ABC, abstractmethod


class GrassmannMeasureBase(ABC):
    """
    Base class for computing measures (e.g., chordal, geodesic) between 
    subspaces and prototype subspaces on the Grassmann manifold.
    """

    def _validate_inputs(self, subspaces: Tensor):
        assert subspaces.ndim == 3, (
            f"Expected shape (batch_size, dim_of_embedding, dim_of_subspace), got {subspaces.shape}"
        )

    def _svd(self, subspaces: Tensor, prototypes: Tensor):
        """Compute SVD of the pairwise cross-covariance matrices."""
        subspaces = subspaces.unsqueeze(dim=1)
        U, S, Vh = torch.linalg.svd(
            torch.transpose(subspaces, 2, 3) @ prototypes.to(subspaces.dtype),
            full_matrices=False,
        )
        return U, S, Vh

    @abstractmethod
    def compute(self, singular_values: Tensor, relevances: Tensor) -> Tensor:
        """Define how to compute the measure from singular values S."""
        pass

    def __call__(self, subspaces: Tensor, prototypes: Tensor, relevances: Tensor):
        """Compute measure and return structured output."""
        self._validate_inputs(subspaces)
        U, S, Vh = self._svd(subspaces, prototypes)

        measure = self.compute(S, relevances)

        if torch.isnan(measure).any():
            raise Exception('NaN values detected! Try using a more stable configuration.')

        return {
            # 'Q': U,
            # 'Qw': torch.transpose(Vh, 2, 3),
            # 'canonical_correlation': S,
            # 'measure': torch.squeeze(measure, -1),
            'Q': U, # SHAPE: (batch_size, num_of_prototypes, WxH, dim_of_subspaces)
            'Qw': torch.transpose(Vh, 2, 3), # SHAPE: (batch_size, num_of_prototypes, WxH, dim_of_subspaces)
            'canonical_correlation': S, # SHAPE: (batch_size, num_of_prototypes, dim_of_subspaces)
            'score': torch.squeeze(measure, -1), # SHAPE: (batch_size, num_of_prototypes)
        }
    
    @abstractmethod
    def euclidean_gradient(self, subspaces: Tensor, prototypes: Tensor, relevance: Tensor) -> Tensor:
        """Compute Euclidean gradient wrt prototypes."""
        pass

        
