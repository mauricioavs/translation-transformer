import torch

from multi_head_attention import MultiHeadAttention


def test_multi_head_attention_respects_boolean_mask():
    mha = MultiHeadAttention(d_model=4, num_heads=2, dropout=0.0)
    with torch.no_grad():
        eye = torch.eye(4)
        for layer in (mha.q_proj, mha.k_proj, mha.v_proj, mha.out_proj):
            layer.weight.copy_(eye)
            layer.bias.zero_()

    # Two tokens; mask blocks attention to the second token
    sequence = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    mask = torch.tensor([[[[True, False], [True, False]]]])

    output = mha(sequence, sequence, sequence, mask=mask)

    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(output[0, 0], expected, atol=1e-6)
    assert torch.allclose(output[0, 1], expected, atol=1e-6)
