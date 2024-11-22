from torch import nn
import torch.nn.functional as F


class ConvNorm(nn.Module):
    """
    Depth-wise (by default) convolution with group layer norm
    """

    def __init__(self, kernel, stride, padding, in_channels, out_channels, groups=None, is2d=True):
        """
        Args:
            args for Conv2d
        """
        super().__init__()
        if groups is None: # depth-wise
            groups = in_channels
        if is2d:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                groups=groups
            )
        else:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                groups=groups
            )
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        return self.norm(self.conv(x))
    

class FeedForwardNetwork(nn.Module):
    """
    feed-forward network for attention of video-processing
    """

    def __init__(self, kernel, in_channels, hidden_channels, dropout=0.1):
        """
        Args:
            kernel - kernel size for 2nd convolution
            in_channels - number of input channels
            hidden_channels - number of hidden channels
            dropout - dropout rate
        """
        super().__init__()

        self.conv1 = ConvNorm(1, 1, 0, in_channels, hidden_channels, groups=1, is2d=False)
        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels, kernel, 1, 2, groups=hidden_channels
        ) # depth-wise
        self.conv3 = ConvNorm(1, 1, 0, hidden_channels, in_channels, groups=1, is2d=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        FFN forward method.

        Args:
            x (Tensor): input tensor of shape (B, in_channels, T)
        Returns:
            tensor of same shape as input
        """ 
        x = self.conv1(x) 
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)

        return x
