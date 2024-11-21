import torch

from src.model.caf import CAFBlock, ConvParameters


def test_caf_forward():
    video_params = ConvParameters(in_channels=8, out_channels=8, kernel_size=1)
    audio_params = ConvParameters(in_channels=4, out_channels=4, kernel_size=1)
    heads = 2
    caf_block = CAFBlock(audio_params, video_params, heads)

    batch_size = 8
    Ca, Cv, Ta, Tv, F = 4, 8, 16, 32, 64
    audio_features = torch.randn(batch_size, Ca, Ta, F)
    video_features = torch.randn(batch_size, Cv, Tv)

    output = caf_block(audio_features, video_features)

    print("Output shape:", output.shape)

    expected_shape = (batch_size, audio_params.out_channels, Ta, F)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Forward method test passed!")
