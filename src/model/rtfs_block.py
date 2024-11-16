from torch import nn
import torch.nn.functional as F

from src.model.attention import MultiHeadSelfAttention
from src.model.fusion import TF_AR


class RTFSBlock(nn.Module):
    """
    RTFS block
    """

    def __init__(self, q=2, Ca=256, D=64, hidden_size=32, rnn_layers=4, n_freqs=64, n_head=4, layers=1):
        """
        Args:
            q - number of compression steps
            Ca - number of input channels
            D - compressed number of channels
            hidden_size - size of hidden vector in RNN
            rnn_layers - number of layers in RNN
            n_freqs - number of frequencies
            n_head - number of heads for self-attention
            layers - number of iterations to process with RTFS block
        """
        super().__init__()

        self.q = q
        self.hidden_size = hidden_size
        self.layers = layers

        self.channel_down = nn.Conv2d(Ca, D, (1, 1))
        self.channel_up = nn.Conv2d(D, Ca, (1, 1))
        self.convs = nn.ModuleList() # convolutions for compression
        self.reconstruction1 = nn.ModuleList() # convolutions for reconstructions (1st phase)
        self.reconstruction2 = nn.ModuleList() # convolutions for reconstructions (2nd phase)

        for _ in range(q):
            self.convs.append(nn.Conv2d(D, D, (4, 4), stride=2, groups=D))
            self.reconstruction1.append(TF_AR(D))
            if (_ < q - 1):
                self.reconstruction2.append(TF_AR(D))

        self.frequency_unfold = nn.Unfold((8, 1), stride=(1, 1))
        self.time_unfold = nn.Unfold((8, 1), stride=(1, 1))

        self.frequency_rnn = nn.LSTM(
            input_size=D * 8,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )

        self.time_rnn = nn.LSTM(
            input_size=D * 8,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )

        self.frequency_conv_t = nn.ConvTranspose2d(2 * hidden_size, D, kernel_size=(1, 8), stride=(1, 1))
        self.time_conv_t = nn.ConvTranspose2d(2 * hidden_size, D, kernel_size=(8, 1), stride=(1, 1))

        self.attention = MultiHeadSelfAttention(D, n_head, n_freqs // 2**q)


    def forward(self, x, residual=None, **batch):
        """
        Block forward method.

        Args:
            x (Tensor): input tensor of shape (B, Ca, T_dim, F_dim)
        Returns:
            tensor of same shape as input
        """      
        for _ in range(self.layers):
            if residual is not None:
                x = x + residual
            x = self.process_one_iteration(x)

        return x

    def process_one_iteration(self, input_data):
        out = self.channel_down(input_data) # channel downsampling
        
        # Compression

        output_size=(out.shape[-2] // 2**self.q, out.shape[-1] // 2**self.q) # padding?

        A = [out]
        A_G = F.adaptive_avg_pool2d(out, output_size=output_size)

        for conv in self.convs:
            out = conv(out)
            A.append(F.adaptive_avg_pool2d(out, output_size=output_size))
            A_G += F.adaptive_avg_pool2d(out, output_size=output_size)

        # Dual-Path architecture

        # Frequency dimension processing
        batch_size, channels, Td, Fd = A_G.shape
        R_f = A_G.permute(0, 2, 1, 3).reshape(batch_size * Td, channels, Fd, 1) # needs for unfold
        R_f = self.frequency_unfold(R_f)
        R_f = R_f.permute(0, 2, 1)
        R_ff = self.frequency_rnn(R_f)[0].view(batch_size, Td, -1, 2 * self.hidden_size).permute(0, 3, 1, 2) # RNN processing
        R_fff = self.frequency_conv_t(R_ff) + A_G

        # Time dimension processing
        R_t = R_fff.permute(0, 3, 1, 2).reshape(batch_size * Fd, channels, Td, 1) # needs for unfold
        R_t = self.time_unfold(R_t)
        R_t = R_t.permute(0, 2, 1)
        R_tt = self.time_rnn(R_t)[0].view(batch_size, Fd, -1, 2 * self.hidden_size).permute(0, 3, 2, 1) # RNN processing
        R_ttt = self.time_conv_t(R_tt) + R_fff

        # Time-Frequency self-attention

        A_Gs = self.attention(R_ttt) + R_ttt

        # Reconstruction

        A_s = []
        for i in range(self.q): # 1st phase of reconstruction
            A_s.append(self.reconstruction1[i](A[i], A_Gs))

        A_ss = A_s[-1]
        for i in range(self.q - 2, -1, -1): # 2nd phase of reconstruction
            A_ss = self.reconstruction2[i](A_s[i], A_ss) + A[i] # typo in paper?..

        A_0 = self.channel_up(A_ss) # channel upsampling

        return A_0
