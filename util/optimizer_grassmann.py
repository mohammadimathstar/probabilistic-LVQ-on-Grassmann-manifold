import torch
from util.utils import project_to_tangent, qr_retraction, exp_map_update  


class GrassmannOptimizer:
    def __init__(self, lr: float = 0.01, method: str = "exp"):
        """
        PyTorch Grassmann optimizer.
        Args:
            lr: Learning rate for manifold updates
            method: 'exp', 'qr', or 'eucl'
        """
        self.lr = lr
        self.method = method.lower()

    @torch.no_grad()
    def step(self, prototypes: torch.Tensor, G_euclid: torch.Tensor) -> torch.Tensor:
        """
        Perform one Grassmann optimization step.
        Args:
            prototypes: (n_protos, n_features, k)
            G_euclid:   (n_protos, n_features, k)
        Returns:
            Updated prototypes with orthonormal columns.
        """
        if self.method in ("eucl", "euclidean"):
            # Euclidean update followed by retraction
            tentative = prototypes - self.lr * G_euclid
            return qr_retraction(tentative)

        # Project Euclidean gradient onto tangent space
        G_riem = project_to_tangent(prototypes, G_euclid)

        if self.method == "exp":
            # print("Using exponential map for Grassmann update.")
            return exp_map_update(prototypes, -G_riem, self.lr)
        else:
            # Default to QR retraction
            tentative = prototypes - self.lr * G_riem
            return qr_retraction(tentative)

