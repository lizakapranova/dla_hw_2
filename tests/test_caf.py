import pytest
from src.model.caf import CAFBlock, ConvParameters
import torch

def test_caf_forward():
    video_params = ConvParameters(in_channels=3, out_channels=4, kernel_size=3)
    audio_params = ConvParameters(in_channels=3, out_channels=4, kernel_size=1)
    heads = 2  # Number of attention heads

    # Initialize CAFBlock
    caf_block = CAFBlock(video_params, audio_params, heads)

    # Create random input tensors for audio and video features
    batch_size = 8
    Ca, Cv, Ta, Tv, F = 3, 3, 16, 32, 64  # Define dimensions
    audio_features = torch.randn(batch_size, Ca, Ta, F)  # (batch_size, Ca, Ta, F)
    video_features = torch.randn(batch_size, Cv, Tv)  # (batch_size, Cv, Tv)

    # Forward pass
    output = caf_block(audio_features, video_features)

    # Print output shape to verify
    print("Output shape:", output.shape)

    # Check if output shape matches expected shape
    expected_shape = (batch_size, audio_params.out_channels, Ta, F)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Forward method test passed!")