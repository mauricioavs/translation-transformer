from typing import Iterable, Optional

import torch
import torch.nn as nn

from decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """Transformer decoder consisting of stacked :class:`DecoderLayer` blocks."""

    def __init__(self, layers: Iterable[DecoderLayer], d_model: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)
