import math

import torch

from positional_encoding import PositionalEncoding


def test_positional_encoding_matches_manual_computation():
    d_model = 6
    seq_len = 3
    module = PositionalEncoding(d_model=d_model, max_len=seq_len, dropout=0.0)

    input_embeddings = torch.zeros(1, seq_len, d_model)
    encoded = module(input_embeddings)

    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    expected = torch.zeros(seq_len, d_model)
    expected[:, 0::2] = torch.sin(position * div_term)
    expected[:, 1::2] = torch.cos(position * div_term)

    assert torch.allclose(encoded[0], expected, atol=1e-6)
