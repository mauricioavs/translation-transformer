from typing import Optional

import torch
import torch.nn as nn

from feed_forward import FeedForwardNetwork
from multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """Single encoder block combining self-attention, feed-forward, and residual paths."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        self_attention: Optional[MultiHeadAttention] = None,
        feed_forward: Optional[FeedForwardNetwork] = None,
    ):
        super().__init__()
        self.self_attn = self_attention or MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = feed_forward or FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
