import torch
from torch import nn


class S3Block(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 1):
        super(S3Block, self).__init__()

        self.prelu = nn.PReLU()
        self.conv = nn.Conv2d(channels, channels, kernel_size)
        self.relu = nn.ReLU()

        self.channels = channels

    def forward(self, a_0, a_r):
        mask = self.prelu(a_r)
        mask = self.conv(mask)
        mask = self.relu(mask)

        m_r, m_i = mask.split(self.channels // 2, dim=1)
        E_r, E_i = a_0.split(self.channels // 2, dim=1)

        z_r = m_r * E_r - m_i * E_i
        z_i = m_r * E_i + m_i * E_r

        return torch.concat((z_r, z_i), dim=1)
