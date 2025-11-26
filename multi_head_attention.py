import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention with explicit projection layers per head."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.size()
        x = x.view(batch, length, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, length, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Tensor [batch, query_len, d_model].
            key: Tensor [batch, key_len, d_model].
            value: Tensor [batch, value_len, d_model].
            mask: Optional boolean tensor broadcastable to [batch, heads, query_len, key_len].

        Returns:
            Attended tensor with shape [batch, query_len, d_model].
        """
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("Attention mask must be of dtype bool.")
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = self._combine_heads(context)
        return self.out_proj(context)
