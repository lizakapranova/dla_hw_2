import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from src.metrics.base_metric import BaseMetric


class SiSNRi(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.si_snr = ScaleInvariantSignalNoiseRatio().to(device)

    def __call__(self, predicted_audio: Tensor, speaker_audio: Tensor, mix_audio: Tensor, **kwargs) -> float:
        return self.si_snr(predicted_audio, speaker_audio) - self.si_snr(mix_audio, speaker_audio)
