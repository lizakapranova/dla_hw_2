import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.model.convolutions import FeedForwardNetwork


class TransposedLayerNorm(nn.Module):
    """
    Layer norm along 1st and 3rd dimension
    """

    def __init__(self, input_shape):
        super(TransposedLayerNorm, self).__init__()

        self.norm = nn.LayerNorm(input_shape)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)
    

class PositionalEncoding(nn.Module):
    """
    Positional encoding for multihead attention
    """

    def __init__(self, in_channels, max_length):
        super().__init__()

        pos = torch.zeros(max_length, in_channels)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_channels, 2) * -(torch.log(torch.tensor(max_length)) / in_channels))
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        pos = pos.unsqueeze(0)

        self.pos = pos

    def forward(self, x):
        """
        (batch_size, X, channels) shape expected
        """
        if x.device != self.pos.device:
            self.pos = self.pos.to(x.device)
        x = x + self.pos[:, :x.size(1)]
        return x  

    def to(self, device):
        print('HA?')
        self.pos.to(device)


class Attention(nn.Module):
    """
    Multihead attention with FFN for video processing block
    """

    def __init__(self, in_channels, n_head=8, dropout=0.1, max_length=10000):
        """
        Args:
            in_channels - number of input channels
            n_head - number of heads for attention
            dropout - dropout rate
            max_length - length for positional encoding
        """
        super(Attention, self).__init__()

        self.positional_encoding = PositionalEncoding(in_channels, max_length)
        self.norm1 = nn.LayerNorm(in_channels)
        self.attention = nn.MultiheadAttention(in_channels, n_head, dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(in_channels)

        self.ffn = FeedForwardNetwork(5, in_channels, in_channels * 2)

    def forward(self, x):
        """
        Attention forward method.

        Args:
            x (Tensor): input tensor of shape (B, D, T)
        Returns:
            tensor of same shape as input
        """
        x = x.transpose(1, 2) # needs for norms, positional encoding and attention
        residual = x

        x = self.positional_encoding(self.norm1(x))  
        x = self.attention(x, x, x)[0]
        x = self.norm2(x + self.dropout(x))

        x = x + residual
        x = x.transpose(1, 2)

        x = x + self.ffn(x)

        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Time-Frequency - domain self-attention (for RTFS-block)
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

        self.queries = nn.ModuleList()
        self.keys = nn.ModuleList()
        self.values = nn.ModuleList()

        for _ in range(self.n_head):
            self.queries.append(
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
            self.keys.append(
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
            self.values.append(
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
            Q = self.queries[i](x).transpose(1, 2).reshape(batch_size, Td, Fd * self.hidden_channels)
            K = self.keys[i](x).transpose(1, 2).reshape(batch_size, Td, Fd * self.hidden_channels)
            V = self.values[i](x).transpose(1, 2).reshape(batch_size, Td, Fd * (self.in_channels // self.n_head))
            attention = functional.softmax(Q @ K.transpose(1, 2) / (self.n_freqs * self.hidden_channels)**0.5, dim=-1) @ V

            head_results.append(attention)

        result = torch.cat(head_results, dim=-1).reshape(batch_size, Td, self.in_channels, Fd).transpose(1, 2)
        return self.after_attention(result) + residual
   