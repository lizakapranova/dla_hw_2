import torch

from src.model.video_processing import VP

def test_vp_block_forward():
    Cv, T_dim = 256, 500
    vp_block = VP()

    batch_size = 8
    x = torch.randn(batch_size, Cv, T_dim)

    output = vp_block(x)

    print("Output shape:", output.shape)

    expected_shape = (batch_size, Cv, T_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Forward method test passed!")
