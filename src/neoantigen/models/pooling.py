# src/neoantigen/models/pooling.py
"""Pooling layers shared by every backbone."""
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """
    Learned attention pooling: collapses a variable-length token sequence to a
    single vector using a softmax-weighted sum.

    Padding tokens are masked to -inf before softmax so they carry zero weight.
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.w = nn.Linear(d_in, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, L, D)  token embeddings
        mask: (B, L)     1 = real token, 0 = padding
        returns (B, D)
        """
        # Run in float32: softmax over long sequences can overflow and produce
        # NaN in float16 under torch.amp.autocast("cuda").
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            logits = self.w(x).squeeze(-1)                         # (B, L)
            logits = logits.masked_fill(mask == 0, float("-inf"))  # mask padding
            weights = torch.softmax(logits, dim=-1)                # (B, L)
            # nan_to_num guards the degenerate case where every position is
            # masked (softmax of all-(-inf) -> NaN); fall back to zero.
            weights = weights.nan_to_num(0.0)
            return torch.bmm(weights.unsqueeze(1), x).squeeze(1)   # (B, D)
