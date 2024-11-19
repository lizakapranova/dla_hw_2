import torch

from src.video_encoder.video_encoder import VideoEncoder

def test_video_encoder_shape():
    T, H, W = 50, 96, 96
    batch_size = 3

    v_encoder = VideoEncoder()

    frame = torch.randn(batch_size, T, H, W)

    output = v_encoder(frame)

    print("Output shape:", output.shape)

    expected_shape = (batch_size, T, 512)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Video encoder shape test passed!")
