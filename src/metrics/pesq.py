from torch import Tensor
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

class PESQ:
    def __init__(self):
        self.pesq = PerceptualEvaluationSpeechQuality(8000, 'nb')

    def __call__(self, preds: Tensor, target: Tensor) -> float:
        pesq_result = self.pesq(preds, target)
        return pesq_result.item()