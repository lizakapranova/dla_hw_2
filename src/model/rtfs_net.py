from torch import nn

from src.video_encoder import VideoEncoder
from src.audio_encoder.audio_encoder import AudioEncoder
from src.model.rtfs_block import RTFSBlock
from src.model.video_processing import VP
from src.model.caf import CAFBlock, ConvParameters
from src.model.s3 import S3Block
from src.audio_decoder import AudioDecoder


class RTFSNet(nn.Module):
    """
    RTFS-net
    """
    def __init__(self,              
                n_fft=1024,
                Ca=256, 
                Cv=512, 
                caf_heads=4, 
                D=64, 
                rtfs_layers=6, 
                hop_length=256,
                win_length=1024,
                hann=False,
                device="auto"):
        super().__init__()

        self.Ca = Ca
        self.Cv = Cv

        self.audio_encoder = AudioEncoder(output_channels=Ca, n_fft=n_fft, hop_length=hop_length, win_length=win_length, hann=hann)
        self.video_encoder = VideoEncoder(device=device)

        self.audio_processing = RTFSBlock(Ca=Ca, D=D, n_freqs=n_fft // 2 - 1, layers=1)
        self.video_processing = VP(Cv=Cv, D=D)
        
        for param in self.video_encoder.parameters():
            param.requires_grad = False

        self.caf = CAFBlock(ConvParameters(Ca, Ca, 1), ConvParameters(Cv, Cv, 1), caf_heads)

        self.rtfs_block = RTFSBlock(Ca=Ca, D=D, n_freqs=n_fft // 2 - 1, layers=rtfs_layers)

        self.s3 = S3Block(Ca)

        self.audio_decoder = AudioDecoder(output_channels=Ca, n_fft=n_fft, hop_length=hop_length, win_length=win_length, hann=hann)

    def forward(self, mix_audio, video, **batch):
        """
        RTFS-net forward method.

        Args:
            audio_input (Tensor): input tensor of shape (B, T_a)
            video_input (Tensor): input tensor of shape (B, T_v, H, W)
        Returns:
            tensor of shape (B, T_a) - predicted audio
        """
        audio_input = mix_audio

        audio_encoded = self.audio_encoder(audio_input).transpose(-1, -2)
        video_encoded = self.video_encoder(video).transpose(-1, -2)

        audio = self.audio_processing(audio_encoded) + audio_encoded
        video = self.video_processing(video_encoded)

        x = self.caf(audio, video)

        x = self.rtfs_block(x, audio_encoded)

        x = self.s3(audio_encoded, x)

        x = self.audio_decoder(x.transpose(-1, -2), audio_input.size(1))

        return {"predicted_audio": x}
    
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
