import torch
import torch.nn as nn
from lvq.prototypes_gradients import AnglePrototypeLayer
from util.grassmann import init_randn
from lvq.measures.base_measure import GrassmannMeasureBase
import argparse


class PrototypeLayer(nn.Module):
    """
    Prototype layer that stores class prototypes as subspaces and computes
    (dis)similarities using a Grassmann-based metric.
    """
    def __init__(
        self,
        num_classes: int,
        score_fn: GrassmannMeasureBase,
        args: argparse.Namespace,
        dtype=torch.float32,
        device='cpu',
        init_from_data: bool = False,
    ):
        super().__init__()
        self.num_prototypes_per_class = 1 #args.num_of_protos
        self.embedding_dim = args.num_features
        self.subspace_dim = args.dim_of_subspace
        self.score_fn = score_fn

        
        # Initialize prototypes
        if not init_from_data:
            # Initialize prototypes randomly
            self.xprotos, self.yprotos, _, _ = init_randn(
                self.embedding_dim,
                self.subspace_dim,
                num_of_protos=self.num_prototypes_per_class,
                num_of_classes=num_classes,
                device=device,
            )
        else:
            # Just create empty placeholder (will be overwritten later)
            self.xprotos = nn.Parameter(torch.empty(args.nclasses * self.num_prototypes_per_class, self.embedding_dim, self.subspace_dim))

        # self.num_prototypes = self.yprotos.shape[0]
        self.num_prototypes = args.nclasses

        # Relevance weights
        self.relevances = nn.Parameter(
            torch.ones((1, self.xprotos.shape[-1]), dtype=dtype, device=device)
            / self.xprotos.shape[-1]
        )

    def forward(self, xs_subspace: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — compute dissimilarities between input subspaces and prototypes.
        """
        return AnglePrototypeLayer.apply(
            xs_subspace,
            self.xprotos,
            self.relevances,
            self.score_fn,
        )





# import torch.nn as nn
# from lvq.prototypes_gradients import AnglePrototypeLayer
# from util.grassmann import init_randn

# from lvq.measures.base_measure import GrassmannMeasureBase

# import torch
# import torch.nn.functional as F
# import argparse


# class PrototypeLayer(nn.Module):
#     def __init__(self,
#                  num_classes,                 
#                  score_fn: GrassmannMeasureBase,
#                  args: argparse.Namespace,
#                  dtype=torch.float32,
#                  device='cpu'):
#         """
#         Initialize the PrototypeLayer.
#         Args:
#             num_classes (int): Number of classes.
#             score_fn: The scoring function, computing (dis)similarity between inputs and prototypes.
#             args: the arguments passed in the main functions.
#             dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.float32.
#             device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
#         """
#         super().__init__()
#         self._nprototypes_per_class = args.num_of_protos
#         self._embedding_dim = args.num_features
#         self._subspace_dim = args.dim_of_subspace
#         # self._coef_subspace_dim = args.coef_dim_of_subspace # * d: the dimensionality of extracted subspace

#         self.score_fn = score_fn

#         # Initialize prototypes
#         self.xprotos, self.yprotos, self.yprotos_mat, self.yprotos_comp_mat = init_randn(
#             self._embedding_dim,
#             self._subspace_dim,
#             num_of_protos=args.num_of_protos,
#             num_of_classes=num_classes,
#             device=device,
#         )

#         self._number_of_prototypes = self.yprotos.shape[0]

#         # Initialize relevance parameters
#         self.relevances = nn.Parameter(
#             torch.ones((1, self.xprotos.shape[-1]), dtype=dtype, device=device) / self.xprotos.shape[-1]
#         )

#     def forward(self, xs_subspace: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the PrototypeLayer.

#         Args:
#             xs_subspace (torch.Tensor): Input subspaces.

#         Returns:
#             torch.Tensor: Output from the PrototypeLayer.
#         """
#         return AnglePrototypeLayer.apply(xs_subspace, 
#                                          self.xprotos, 
#                                          self.relevances, 
#                                          self.score_fn)
        

