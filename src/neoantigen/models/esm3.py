# src/neoantigen/models/esm3.py
import hashlib
import os

import torch
import torch.nn as nn
from esm.models.esm3 import ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from .pooling import AttentionPool  # re-exported for backward compatibility

ESM3_HID = 1536  # esm3_sm_open_v1 hidden dim

__all__ = ["ESM3Encoder", "AttentionPool", "ESM3_HID"]


class ESM3Encoder(nn.Module):
    """
    ESM3 sequence backbone with optional per-sequence embedding cache.

    Cache behaviour
    ---------------
    When ``use_cache=True`` and ``freeze=True``, per-sequence embeddings are
    stored in an in-memory dict (and optionally on disk under ``cache_dir``).
    The cache avoids re-running ESM3 on sequences seen in previous batches or
    epochs — particularly valuable for MHC alleles that repeat across samples.

    Calling ``invalidate_cache()`` clears the in-memory cache and disables
    further caching (used by progressive unfreezing, which changes the weights).
    """

    def __init__(
        self,
        model_id: str = "esm3_sm_open_v1",
        freeze: bool = True,
        use_cache: bool = False,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.model = ESM3.from_pretrained(model_id)
        self.tokenizer = EsmSequenceTokenizer()
        self.hidden_size = ESM3_HID
        self._model_id = model_id

        # Cache is only meaningful with a frozen backbone
        self._use_cache = use_cache and freeze
        self._mem_cache: dict[str, torch.Tensor] = {}  # seq -> (L, D) cpu tensor
        self._cache_dir = cache_dir if (use_cache and cache_dir) else None

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def invalidate_cache(self) -> None:
        """Clear in-memory cache and disable caching (call before unfreezing)."""
        self._mem_cache.clear()
        self._use_cache = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _seq_key(self, seq: str) -> str:
        return hashlib.md5(f"{self._model_id}:{seq}".encode()).hexdigest()

    def _disk_load(self, seq: str) -> torch.Tensor | None:
        if not self._cache_dir:
            return None
        path = os.path.join(self._cache_dir, f"{self._seq_key(seq)}.pt")
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def _disk_save(self, seq: str, emb: torch.Tensor) -> None:
        if not self._cache_dir:
            return
        os.makedirs(self._cache_dir, exist_ok=True)
        path = os.path.join(self._cache_dir, f"{self._seq_key(seq)}.pt")
        torch.save(emb.cpu(), path)

    def _encode_batch(self, sequences: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ESM3 on a list of sequences; returns (embeddings, mask)."""
        encoded = self.tokenizer(sequences, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        dev_type = self.device.type
        with torch.amp.autocast(
            device_type=dev_type,
            dtype=torch.bfloat16,
            enabled=(dev_type == "cuda"),
        ):
            output = self.model(sequence_tokens=input_ids)
        return output.embeddings.float(), attention_mask  # (B, L, D), (B, L)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, sequences: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sequences: list[str] of amino acid sequences
        returns:
            embeddings: (B, L, D)  per-token hidden states
            mask:       (B, L)     1 = real token, 0 = padding
        """
        if not self._use_cache:
            return self._encode_batch(sequences)

        # ── Resolve cached vs. uncached sequences ─────────────────────────────
        cached_embs: dict[int, torch.Tensor] = {}
        uncached_seqs: list[str] = []
        uncached_idxs: list[int] = []

        for i, seq in enumerate(sequences):
            emb = self._mem_cache.get(seq)
            if emb is None:
                emb = self._disk_load(seq)
                if emb is not None:
                    self._mem_cache[seq] = emb
            if emb is not None:
                cached_embs[i] = emb
            else:
                uncached_seqs.append(seq)
                uncached_idxs.append(i)

        # ── Encode uncached batch ─────────────────────────────────────────────
        if uncached_seqs:
            new_embs, new_masks = self._encode_batch(uncached_seqs)
            for j, idx in enumerate(uncached_idxs):
                seq_len = int(new_masks[j].sum().item())
                emb_j = new_embs[j, :seq_len].cpu()
                seq = uncached_seqs[j]
                self._mem_cache[seq] = emb_j
                self._disk_save(seq, emb_j)
                cached_embs[idx] = emb_j

        # ── Reassemble padded batch ───────────────────────────────────────────
        dev = self.device
        max_len = max(v.shape[0] for v in cached_embs.values())
        B = len(sequences)

        emb_batch = torch.zeros(B, max_len, self.hidden_size, device=dev)
        mask_batch = torch.zeros(B, max_len, dtype=torch.long, device=dev)

        for i, emb in cached_embs.items():
            L = emb.shape[0]
            emb_batch[i, :L] = emb.to(dev)
            mask_batch[i, :L] = 1

        return emb_batch, mask_batch
