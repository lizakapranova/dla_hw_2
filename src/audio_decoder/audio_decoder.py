import torch
import torch.nn as nn
import torch.fft


class AudioDecoder(nn.Module):
    def __init__(self, output_channels=64, kernel_size=3, hop_length=128, win_length=256, hann=False, **kwargs):
        super(AudioDecoder, self).__init__()

        self.hop_length = hop_length
        self.win_length = win_length
        self.hann = hann
        
        self.deconv2d = nn.ConvTranspose2d(in_channels=output_channels,
                                            out_channels=2,
                                            kernel_size=kernel_size)
        
    def forward(self, encoded_audio, length):
        alpha = self.deconv2d(encoded_audio)
        
        real_part = alpha[:, 0, :, :]
        imag_part = alpha[:, 1, :, :]
        
        complex_stft = torch.complex(real_part, imag_part)

        final_wav = torch.istft(complex_stft, 
                          n_fft=1024, 
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=torch.hann_window(self.win_length) if self.hann else torch.ones(self.win_length),
                          onesided=True, 
                          center=False,
                          length=length)
        return final_wav
