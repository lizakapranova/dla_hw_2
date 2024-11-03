import torch

from src.model.caf import CAFBlock, ConvParameters


def test_caf_forward():
    video_params = ConvParameters(in_channels=3, out_channels=4, kernel_size=3)
    audio_params = ConvParameters(in_channels=3, out_channels=4, kernel_size=1)
    heads = 2
    caf_block = CAFBlock(video_params, audio_params, heads)

    batch_size = 8
    Ca, Cv, Ta, Tv, F = 3, 3, 16, 32, 64
    audio_features = torch.randn(batch_size, Ca, Ta, F)
    video_features = torch.randn(batch_size, Cv, Tv)

    output = caf_block(audio_features, video_features)

    print("Output shape:", output.shape)

    expected_shape = (batch_size, audio_params.out_channels, Ta, F)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Forward method test passed!")
