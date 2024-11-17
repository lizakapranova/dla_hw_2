import torch
import torch.nn as nn
import torch.fft

class AudioDecoder(nn.Module):
    def __init__(self, output_channels=64, kernel_size=3, **kwargs):
        super(AudioDecoder, self).__init__()
        
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
                          onesided=True, 
                          center=False,
                          length=length)
        
        return final_wav
