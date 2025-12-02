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


