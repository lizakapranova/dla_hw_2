import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.model.convolutions import ConvNorm


class TF_AR(nn.Module):
    """
    Temporal-Frequency Attention Reconstruction
    """

    def __init__(self, channels, is2d=True):
        """
        Args:
            channels - number of input and output channels
        """
        super().__init__()

        self.W1 = ConvNorm(4, 1, "same", channels, channels, is2d=is2d) # warning ignore?
        self.W2 = ConvNorm(4, 1, "same", channels, channels, is2d=is2d)
        self.W3 = ConvNorm(4, 1, "same", channels, channels, is2d=is2d)

        self.is2d = is2d

    def forward(self, m, n): # shape of n < shape of m
        size = m.shape[-2:] if self.is2d else m.shape[-1:]     
        x1 = nn.functional.interpolate(nn.functional.sigmoid(self.W1(n)), size=size, mode="nearest")
        x2 = self.W2(m)
        x3 = nn.functional.interpolate(self.W3(n), size=size, mode="nearest")

        return x1 * x2 + x3