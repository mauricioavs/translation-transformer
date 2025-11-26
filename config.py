from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Immutable configuration object for the Transformer model."""

    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    pad_id: int = 0
    tie_embeddings: bool = True
    max_len: int = 4096
