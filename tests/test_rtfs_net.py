import torch

from src.model.rtfs_net import RTFSNet

def test_rtfs_net_forward():
    batch_size = 3
    Ta = 32000
    Tv = 300
    H = 96
    W = 96
    F_dim=511

    rtfs_net = RTFSNet(n_freqs=F_dim)

    audio_input = torch.randn(batch_size, Ta)
    video_input = torch.randn(batch_size, Tv, H, W)

    output = rtfs_net(audio_input, video_input)["predicted_audio"]

    print("Output shape:", output.shape)

    expected_shape = (batch_size, Ta)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Forward method test passed!")
