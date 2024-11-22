import torch
from torch import Tensor
from torchmetrics.audio import SignalDistortionRatio
from src.metrics.base_metric import BaseMetric


class SDRi(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sdr = SignalDistortionRatio().to(device)

    def __call__(self, predicted_audio: Tensor, speaker_audio: Tensor, mix_audio: Tensor, **kwargs) -> float:
        return self.sdr(predicted_audio, speaker_audio) - self.sdr(mix_audio, speaker_audio)
