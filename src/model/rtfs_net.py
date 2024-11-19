from torch import nn
import torch

from src.video_encoder import VideoEncoder
from src.audio_encoder.audio_encoder import AudioEncoder
from src.model.rtfs_block import RTFSBlock
from src.model.video_processing import VP
from src.model.caf import CAFBlock, ConvParameters
from src.model.s3 import S3Block
from src.audio_decoder import AudioDecoder
import time

class RTFSNet(nn.Module):
    """
    RTFS-net
    """
    def __init__(self, n_freqs, Ca=256, Cv=256, caf_heads=4, D=64, rtfs_layers=6): # add more paras and config
        super().__init__()

        self.Ca = Ca
        self.Cv = Cv

        self.audio_encoder = AudioEncoder(output_channels=Ca)
        self.video_encoder = VideoEncoder() # shapes ok?

        self.audio_processing = RTFSBlock(Ca=Ca, D=D, n_freqs=n_freqs, layers=1)
        # self.video_processing = VP(Cv=Cv, D=D)
        '''
        for param in self.video_processing.parameters():
            param.requires_grad = False'''

        # self.caf = CAFBlock(ConvParameters(Cv, Ca, 3), ConvParameters(Cv, Cv, 1), caf_heads) # params ok?

        self.rtfs_block = RTFSBlock(Ca=Ca, D=D, n_freqs=n_freqs, layers=rtfs_layers)

        self.s3 = S3Block(Ca)

        self.audio_decoder = AudioDecoder(output_channels=Ca)

    def forward(self, mix_audio, video, **batch):
        """
        RTFS-net forward method.

        Args:
            audio_input (Tensor): input tensor of shape (B, 1, T_a)
            video_input (Tensor): input tensor of shape (B, T_v, H, W)
        Returns:
            tensor of shape (B, T_a) - predicted audio
        """
        audio_input = mix_audio
        batch_size = audio_input.size(0)

        audio_encoded = self.audio_encoder(audio_input).transpose(-1, -2)
        # video = torch.zeros(batch_size, self.Cv, 100) # fix this

        # audio = self.audio_processing(audio_encoded)

        # video = self.video_processing(video_encoded)
        # x = self.caf(audio, video.cuda())
        x = audio_encoded

        x = self.rtfs_block(x, audio_encoded)

        x = self.s3(audio_encoded, x)

        x = self.audio_decoder(x.transpose(-1, -2), audio_input.size(1))

        return {"predicted_audio": x} # fix tests
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = ""
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        for block in ["audio_encoder", "video_encoder", "audio_processing", "video_processing", "caf", "rtfs_block", "s3", "audio_decoder"]:           
            trainable_parameters = sum(
                [p.numel() for p in self.__getattr__(block).parameters() if p.requires_grad]
            )
            result_info = result_info + f"\nTrainable parameters of {block}: {trainable_parameters}"

        return result_info
