import torch
from torch import nn

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


class SISNRLoss(nn.Module):
    """
    SI-SNR loss
    """

    def __init__(self):
        super().__init__()
        self.loss = ScaleInvariantSignalNoiseRatio()

    def forward(self, predicted_audio: torch.Tensor, speaker_audio: torch.Tensor, **batch):
        return {"loss": -self.loss(predicted_audio, speaker_audio)}
