import torch
import torch.nn as nn
import torch.nn.functional as functional


class TransposedLayerNorm(nn.Module):
    """
    Layer norm along 1st and 3rd dimension
    """

    def __init__(self, input_shape):
        super(TransposedLayerNorm, self).__init__()

        self.norm = nn.LayerNorm(input_shape)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """
    Time-Frequency - domain self-attention
    """

    def __init__(self, in_channels, hidden_channels, n_freqs, n_head = 4):
        """
        Args:
            in_channels - number of input channels
            hidden_channels - number of hidden channels
            n_freqs - number of frequencies
            n_head - number of heads
        """
        super(MultiHeadSelfAttention, self).__init__()

        self.in_channels = in_channels
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hidden_channels = hidden_channels

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()

        for _ in range(self.n_head):
            self.Queries.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        hidden_channels,
                        (1, 1),
                        stride=(1, 1)
                    ),
                    nn.PReLU(),
                    TransposedLayerNorm([hidden_channels, n_freqs])
                )
            )
            self.Keys.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        hidden_channels,
                        (1, 1),
                        stride=(1, 1)
                    ),
                    nn.PReLU(),
                    TransposedLayerNorm([hidden_channels, n_freqs])
                )
            )
            self.Values.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels // n_head,
                        (1, 1),
                        stride=(1, 1)
                    ),
                    nn.PReLU(),
                    TransposedLayerNorm([in_channels // n_head, n_freqs])
                )
            )

        self.after_attention = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                (1, 1),
                stride=(1, 1)
            ),
            nn.PReLU(),
            TransposedLayerNorm([in_channels, n_freqs])
        )

    def forward(self, x):
        """
        Attention forward method.

        Args:
            x (Tensor): input tensor.
        Returns:
            tensor of same shape as input
        """   
        batch_size, _, Td, Fd = x.size()
        residual = x

        head_results = []
        for i in range(self.n_head):
            Q = self.Queries[i](x).transpose(1, 2).reshape(batch_size, Td, Fd * self.hidden_channels)
            K = self.Keys[i](x).transpose(1, 2).reshape(batch_size, Td, Fd * self.hidden_channels)
            V = self.Values[i](x).transpose(1, 2).reshape(batch_size, Td, Fd * (self.in_channels // self.n_head))

            attention = functional.softmax(Q @ K.transpose(1, 2) / (self.n_freqs * self.hidden_channels)**0.5) @ V

            head_results.append(attention)

        result = torch.cat(head_results, dim=-1).reshape(batch_size, Td, self.in_channels, Fd).transpose(1, 2)
        return self.after_attention(result) + residual
    