import torch
import torch.nn as nn
import torch.fft

class AudioEncoder(nn.Module):
    def __init__(self, output_channels=64, kernel_size=3, hop_length=128, win_length=256, hann=False, **kwargs):
        super(AudioEncoder, self).__init__()

        self.hop_length = hop_length
        self.win_length = win_length
        self.hann = hann
        
        self.conv2d = nn.Conv2d(in_channels=2,
                                out_channels=output_channels, 
                                kernel_size=kernel_size)
        
    def forward(self, wav):
        stft_result = torch.stft(wav,
                                n_fft=1024,
                                hop_length=self.hop_length,
                                win_length=self.win_length,
                                window=torch.hann_window(self.win_length) if self.hann else torch.ones(self.win_length),
                                onesided=True,
                                center=False,
                                return_complex=True)  
        
        real_part = stft_result.real
        imag_part = stft_result.imag
            
        alpha = torch.stack((real_part, imag_part), dim=1) 
        
        encoded_audio = self.conv2d(alpha)
        return encoded_audio
    