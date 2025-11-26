from typing import Iterable, Optional

import torch
import torch.nn as nn

from encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """Transformer encoder consisting of a stack of :class:`EncoderLayer` blocks."""

    def __init__(self, layers: Iterable[EncoderLayer], d_model: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
