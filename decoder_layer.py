from typing import Optional

import torch
import torch.nn as nn

from feed_forward import FeedForwardNetwork
from multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """Single decoder block including self-attention, cross-attention, and feed-forward stages."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        self_attention: Optional[MultiHeadAttention] = None,
        cross_attention: Optional[MultiHeadAttention] = None,
        feed_forward: Optional[FeedForwardNetwork] = None,
    ):
        super().__init__()
        self.self_attn = self_attention or MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = cross_attention or MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = feed_forward or FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x
