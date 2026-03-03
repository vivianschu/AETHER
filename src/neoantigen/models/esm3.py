import torch
import torch.nn as nn
from esm.models.esm3 import ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

ESM3_HID = 1536  # esm3_sm_open_v1 hidden dim


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
        logits = self.w(x).squeeze(-1)                        # (B, L)
        logits = logits.masked_fill(mask == 0, float("-inf"))  # mask padding
        weights = torch.softmax(logits, dim=-1)                # (B, L)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)


class ESM3Encoder(nn.Module):
    """
    ESM3 sequence backbone.

    Returns per-token hidden states and the corresponding attention mask so
    downstream modules can do their own pooling or cross-attention.
    """

    def __init__(self, model_id: str = "esm3_sm_open_v1", freeze: bool = True):
        super().__init__()
        self.model = ESM3.from_pretrained(model_id)
        self.tokenizer = EsmSequenceTokenizer()
        self.hidden_size = ESM3_HID

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, sequences: list) -> tuple:
        """
        sequences: list[str] of amino acid sequences
        returns:
            embeddings: (B, L, D)  per-token hidden states
            mask:       (B, L)     1 = real token, 0 = padding
        """
        encoded = self.tokenizer(
            sequences,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        output = self.model(sequence_tokens=input_ids)
        return output.embeddings, attention_mask  # (B, L, D), (B, L)
