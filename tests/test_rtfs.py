import torch

from src.model.rtfs_block import RTFSBlock

def test_rtfs_block_forward():
    batch_size = 3
    Ca = 256
    Ta = 125
    F_dim = 1030
    rtfs_block = RTFSBlock(n_freqs=F_dim)

    x = torch.randn(batch_size, Ca, Ta, F_dim)

    output = rtfs_block(x)

    print("Output shape:", output.shape)

    expected_shape = (batch_size, Ca, Ta, F_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Forward method test passed!")
