from measures.base_measure import GrassmannMeasureBase
from torch import Tensor
import torch


class ChordalMeasure(GrassmannMeasureBase):
    def compute(self, singular_values: Tensor, relevances: Tensor) -> Tensor:
        return 1 - torch.transpose(
            relevances @ torch.transpose(singular_values, 1, 2).to(relevances.dtype),
            1, 2
        )
