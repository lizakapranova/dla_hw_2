import torch

from src.video_encoder.video_encoder import VideoEncoder

def test_video_encoder_init():
    v_encoder = VideoEncoder()

    assert v_encoder
    print("Video encoder init test passed!")
