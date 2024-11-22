from torch import Tensor
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

class STOI:
    def __init__(self):
        self.stoi = ShortTimeObjectiveIntelligibility(8000)

    def __call__(self, preds: Tensor, target: Tensor) -> float:
        stoi_result = self.stoi(preds, target)
        return stoi_result.item()
