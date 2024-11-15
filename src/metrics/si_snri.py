from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SiSNRi:
    def __init__(self):
        self.si_snri = ScaleInvariantSignalNoiseRatio()

    def __call__(self, preds: Tensor, target: Tensor) -> float:
        si_snri_result = self.si_snri(preds, target)
        return si_snri_result.item()
