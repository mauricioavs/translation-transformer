import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape [batch, length, d_model].
            positions: Optional tensor with positional indices when using relative positions.

        Returns:
            Tensor with positional encodings added, same shape as ``x``.
        """
        if positions is not None:
            pe = self.pe[:, positions]
        else:
            pe = self.pe[:, : x.size(1)]
        x = x + pe
        return self.dropout(x)
