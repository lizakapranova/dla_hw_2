import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion import TF_AR


class VP(nn.Module):
    """
    Video processing block
    """
    def __init__(self, q=2, Ca=256, D=64):
        """
        Args:
            q - number of compression steps
            Ca - number of input channels
            D - compressed number of channels
        """
        super(VP, self).__init__()

        self.q = q
        self.channel_down = nn.Conv1d(Ca, D, 1)
        self.channel_up = nn.Conv1d(D, Ca, 1)

        self.convs = nn.ModuleList()
        self.reconstruction1 = nn.ModuleList()
        self.reconstruction2 = nn.ModuleList()
        for _ in range(q):
            self.convs.append(nn.Conv1d(D, D, 4, stride=2, groups=D))
            self.reconstruction1.append(TF_AR(D, is2d=False))
            if (_ < q - 1):
                self.reconstruction2.append(TF_AR(D, is2d=False))

    def forward(self, x):
        """
        Block forward method.

        Args:
            x (Tensor): input tensor of shape (B, Ca, T_dim)
        Returns:
            tensor of same shape as input
        """
        out = self.channel_down(x) # channel downsampling
        
        # Compression

        output_size=(out.shape[-1] // 2**self.q) # padding?

        V = [out]
        V_G = F.adaptive_avg_pool1d(out, output_size=output_size)

        for conv in self.convs:
            out = conv(out)
            V.append(F.adaptive_avg_pool1d(out, output_size=output_size))
            V_G += F.adaptive_avg_pool1d(out, output_size=output_size)

        # attention (not yet)
        V_Gs = V_G

        V_s = []
        for i in range(self.q): # 1st phase of reconstruction
            V_s.append(self.reconstruction1[i](V[i], V_Gs))

        V_ss = V_s[-1]
        for i in range(self.q - 2, -1, -1): # 2nd phase of reconstruction
            V_ss = self.reconstruction2[i](V_s[i], V_ss) + V[i] # typo in paper?..

        V_0 = self.channel_up(V_ss) # channel upsampling

        return V_0
    