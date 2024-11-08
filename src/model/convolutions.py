import torch
from torch import nn
import torch.nn.functional as F


class ConvNorm(nn.Module):
    """
    Depth-wise convolution with group layer norm
    """

    def __init__(self, kernel, stride, padding, in_channels, out_channels, is2d=True):
        """
        Args:
            args for Conv2d
        """
        super().__init__()
        if is2d:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                groups=in_channels
            )
        else:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                groups=in_channels
            )
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        return self.norm(self.conv(x))