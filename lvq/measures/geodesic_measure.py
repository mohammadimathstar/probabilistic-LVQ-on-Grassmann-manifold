from measures.base_measure import GrassmannMeasureBase
from torch import Tensor
import torch


class GeodesicMeasure(GrassmannMeasureBase):
    def compute(self, singular_values: Tensor, relevances: Tensor) -> Tensor:
        return torch.transpose(
            relevances @ torch.transpose(
                torch.acos(singular_values) ** 2,
                1, 2
            ).to(relevances.dtype),
            1, 2
        )
