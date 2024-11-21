import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from src.video_encoder.lipreading.model import Lipreading


class VideoEncoder(nn.Module):
    def __init__(self, output_channels=512, device='auto'):
        super(VideoEncoder, self).__init__()
        self.model = Lipreading(modality='video',
                                num_classes=500,
                                use_boundary=False,
                                backbone_type="resnet",
                                relu_type="swish",
                                width_mult=1.0,
                                extract_feats=True,
                                densetcn_options={
            "block_config": [
                3,
                3,
                3,
                3
            ],
            "growth_rate_set": [
                384,
                384,
                384,
                384
            ],
            "kernel_size_set": [
                3,
                5,
                7
            ],
            "dilation_size_set": [
                1,
                2,
                5
            ],
            "reduced_size": output_channels,
            "squeeze_excitation": True,
            "dropout": 0.2,
        })
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load('src/video_encoder/lrw_resnet18_dctcn_video.pth', map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        
    def forward(self, data):
        self.model.eval()
        embedding_array = []
        for batch_idx in range(data.shape[0]):
            frame = data[batch_idx, :, :, :]
            embedding = self.model(frame[None, None, :, :, :], lengths=[frame.shape[0]])
            embedding_array.append(embedding[0, :, :])
        embeddings = torch.stack(embedding_array)
        return embeddings
