from typing import Optional

import torch


def create_padding_mask(sequence: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Builds a mask highlighting valid (non-padding) positions.

    Returns:
        Boolean tensor with shape [batch, 1, 1, length].
    """
    if sequence.dim() != 2:
        raise ValueError("Padding mask expects input of shape [batch, length].")
    mask = sequence.ne(pad_id).unsqueeze(1).unsqueeze(2)
    return mask


def create_causal_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Lower triangular mask preventing attention to future positions."""
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def create_decoder_mask(
    tgt: torch.Tensor,
    pad_id: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Combines padding and causal masks for the decoder."""
    padding_mask = create_padding_mask(tgt, pad_id)
    seq_len = tgt.size(1)
    causal_mask = create_causal_mask(seq_len, device=device)
    return padding_mask & causal_mask
