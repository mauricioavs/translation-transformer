import torch

from config import TransformerConfig
from transformer import Transformer


def test_transformer_forward_produces_logits():
    config = TransformerConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=32,
        dropout=0.0,
        pad_id=0,
        tie_embeddings=True,
        max_len=64,
    )
    model = Transformer(config)

    src = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    tgt = torch.tensor([[1, 4, 5, 0]], dtype=torch.long)

    logits = model(src, tgt)
    assert logits.shape == (1, tgt.size(1), config.vocab_size)

    # Ensure embeddings are tied when requested
    assert model.src_embedding.weight.data_ptr() == model.tgt_embedding.weight.data_ptr()
