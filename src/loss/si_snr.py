import torch
from torch import nn

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SISNRLoss(nn.Module):
    """
    SI-SNR loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **batch):
        return {"loss": ScaleInvariantSignalNoiseRatio(prediction, target)}
