import torch

from src.model.s3 import S3Block

def test_s3_block():
    channels = 10
    s3_block = S3Block(channels)

    batch_size = 8
    Ca, Ta, F = 10, 16, 64
    a_0 = torch.randn(batch_size, Ca, Ta, F)
    a_r = torch.randn(batch_size, Ca, Ta, F)

    output = s3_block(a_0, a_r)

    print("Output shape:", output.shape)

    expected_shape = (batch_size, channels, Ta, F)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Forward method test passed!")
