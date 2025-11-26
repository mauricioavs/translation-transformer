import torch

from decoder_layer import DecoderLayer
from encoder_layer import EncoderLayer


def test_encoder_layer_preserves_shape_and_gradients():
    layer = EncoderLayer(d_model=8, num_heads=2, d_ff=16, dropout=0.0)
    x = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.ones(2, 1, 1, 5, dtype=torch.bool)

    out = layer(x, mask)
    assert out.shape == x.shape

    out.sum().backward()
    assert x.grad is not None


def test_decoder_layer_handles_masks():
    layer = DecoderLayer(d_model=8, num_heads=2, d_ff=16, dropout=0.0)
    tgt = torch.randn(2, 4, 8, requires_grad=True)
    memory = torch.randn(2, 6, 8)
    tgt_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).expand(2, -1, -1, -1)
    memory_mask = torch.ones(2, 1, 1, 6, dtype=torch.bool)

    out = layer(tgt, memory, tgt_mask, memory_mask)
    assert out.shape == tgt.shape

    out.mean().backward()
    assert tgt.grad is not None
