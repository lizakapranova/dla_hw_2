import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

def load_video_npz(file_path):
    data = np.load(file_path)
    frames = data['data']

    frames = torch.tensor(frames, dtype=torch.float32)

    if frames.ndim == 3:
        frames = frames.unsqueeze(1)
        frames = frames.expand(-1, 3, -1, -1)

    frames = frames / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    return frames


class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        
        resnet = models.resnet18(pretrained=True)
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        num_frames, channels, height, width = x.shape

        frame_features = []
        for t in range(num_frames):
            frame = x[t].unsqueeze(0) 
            features = self.feature_extractor(frame)  # (1, 512, 1, 1)
            features = features.view(-1)
            frame_features.append(features)
        
        frame_features = torch.stack(frame_features) 
        
        video_embedding = frame_features.mean(dim=0)
        
        return video_embedding # (512,)