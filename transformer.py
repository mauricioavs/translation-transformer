import math
from typing import Optional

import torch
import torch.nn as nn

from config import TransformerConfig
from decoder import Decoder
from decoder_layer import DecoderLayer
from encoder import Encoder
from encoder_layer import EncoderLayer
from masks import create_decoder_mask, create_padding_mask
from positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """Encoder-decoder Transformer tailored for sequence-to-sequence translation."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.src_embedding = nn.Embedding(config.vocab_size, d_model, padding_idx=config.pad_id)
        self.tgt_embedding = nn.Embedding(config.vocab_size, d_model, padding_idx=config.pad_id)
        self.src_positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=config.max_len,
            dropout=config.dropout,
        )
        self.tgt_positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=config.max_len,
            dropout=config.dropout,
        )

        encoder_layers = [
            EncoderLayer(d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.num_encoder_layers)
        ]
        decoder_layers = [
            DecoderLayer(d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.num_decoder_layers)
        ]

        self.encoder = Encoder(encoder_layers, d_model)
        self.decoder = Decoder(decoder_layers, d_model)
        self.generator = nn.Linear(d_model, config.vocab_size)
        self.scale = math.sqrt(d_model)

        if config.tie_embeddings:
            self.tgt_embedding.weight = self.src_embedding.weight

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.src_embedding(src) * self.scale
        embeddings = self.src_positional_encoding(embeddings)
        return self.encoder(embeddings, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.tgt_embedding(tgt) * self.scale
        embeddings = self.tgt_positional_encoding(embeddings)
        return self.decoder(embeddings, memory, tgt_mask, memory_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if src_mask is None:
            src_mask = create_padding_mask(src, self.config.pad_id)
        if tgt_mask is None:
            tgt_mask = create_decoder_mask(tgt, self.config.pad_id, device=tgt.device)
        memory = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.generator(decoder_out)
