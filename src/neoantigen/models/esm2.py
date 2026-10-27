# src/neoantigen/models/esm2.py
"""
ESM2 sequence backbone, loaded through HuggingFace ``transformers``.

Why a cache matters here
------------------------
In the MHCAttnNet class-I split there are 343,713 training rows but only
**152 distinct MHC sequences** (~275 residues each), against 160,404 distinct
peptides (~9 residues each). Encoding the MHC side row-by-row therefore repeats
the same forward pass thousands of times, and because the MHC sequence is ~30x
longer than the peptide it dominates the backbone cost.

``precompute()`` runs the frozen backbone once over the unique MHC sequences and
pins the result in memory. After that the cache is *sealed*: lookups still hit,
but sequences that miss (i.e. peptides) are encoded live and never stored, so
the 160k unique peptides cannot grow the cache without bound.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

# Hidden width of each public ESM2 checkpoint, used to size the projections
# without instantiating the model first.
ESM2_HIDDEN = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": 2560,
}


class ESM2Encoder(nn.Module):
    """
    Parameters
    ----------
    model_id : str
        HuggingFace ESM2 checkpoint name.
    freeze : bool
        Freeze all backbone weights. Caching is only sound when frozen, since a
        weight update would invalidate every stored embedding.
    max_length : int | None
        Truncate tokenized sequences to this many tokens (``None`` = no limit).
    cache_device : str | None
        Where to hold precomputed embeddings. ``None`` keeps them on the model's
        device (fastest); ``"cpu"`` trades a transfer per batch for GPU memory.
    cache_dtype : torch.dtype
        Storage dtype for cached embeddings. float16 halves memory at negligible
        cost, since these are frozen features feeding a trained projection.
    """

    def __init__(
        self,
        model_id: str = "facebook/esm2_t33_650M_UR50D",
        freeze: bool = True,
        max_length: int | None = None,
        cache_device: str | None = None,
        cache_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.model = EsmModel.from_pretrained(model_id, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.hidden_size = self.model.config.hidden_size
        self._model_id = model_id
        self._max_length = max_length

        self._frozen = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        # seq -> (L, D) tensor. Populated only by precompute().
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_sealed = False
        self._cache_device = cache_device
        self._cache_dtype = cache_dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def set_frozen(self, frozen: bool) -> None:
        """
        Flip the frozen flag and the state that depends on it.

        ``_frozen`` drives two things that requires_grad alone does not: the
        backbone being pinned to eval() (so ESM2's own dropout stays off), and
        the no_grad context in ``_encode_batch``. Setting requires_grad without
        this leaves a "frozen" backbone running dropout, or an unfrozen one
        whose parameters never receive a gradient.
        """
        self._frozen = frozen
        for p in self.model.parameters():
            p.requires_grad = not frozen
        self.model.eval() if frozen else self.model.train(self.training)

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode so dropout stays off."""
        super().train(mode)
        if self._frozen:
            self.model.eval()
        return self

    # -- Tokenize + encode ----------------------------------------------------

    def _tokenize(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs: dict = {"padding": True, "return_tensors": "pt"}
        if self._max_length is not None:
            kwargs.update(truncation=True, max_length=self._max_length)
        enc = self.tokenizer(sequences, **kwargs)
        return (
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
        )

    def _encode_batch(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ESM2 over a list of sequences; returns (embeddings, mask)."""
        input_ids, attention_mask = self._tokenize(sequences)
        # No autocast here: the caller's autocast context (if any) already
        # applies, and forcing bfloat16 would fight a fine-tuning run.
        # nullcontext, not enable_grad(): enable_grad would *re-enable* grad
        # inside a caller's no_grad block and build a 650M-param graph during
        # evaluation.
        ctx = torch.no_grad() if self._frozen else contextlib.nullcontext()
        with ctx:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state.float(), attention_mask

    # -- Cache ----------------------------------------------------------------

    @torch.no_grad()
    def precompute(
        self,
        sequences: list[str],
        batch_size: int = 8,
        verbose: bool = True,
    ) -> int:
        """
        Encode the unique entries of ``sequences`` and seal the cache.

        Returns the number of distinct sequences stored. Refuses to run on an
        unfrozen backbone, whose embeddings change every optimizer step.
        """
        if not self._frozen:
            raise RuntimeError(
                "precompute() requires a frozen backbone; cached embeddings "
                "would go stale as soon as the weights update."
            )

        uniq = sorted(set(sequences))
        dev = torch.device(self._cache_device) if self._cache_device else self.device

        was_training = self.model.training
        self.model.eval()
        for i in range(0, len(uniq), batch_size):
            chunk = uniq[i : i + batch_size]
            embs, masks = self._encode_batch(chunk)
            for j, seq in enumerate(chunk):
                n_tok = int(masks[j].sum().item())
                self._cache[seq] = embs[j, :n_tok].to(dev, self._cache_dtype)
            if verbose and (i // batch_size) % 20 == 0:
                print(
                    f"  precompute {min(i + batch_size, len(uniq))}/{len(uniq)}",
                    flush=True,
                )
        if was_training:
            self.model.train()

        self._cache_sealed = True
        if verbose:
            n_bytes = sum(v.numel() * v.element_size() for v in self._cache.values())
            print(
                f"  cached {len(self._cache)} unique sequences "
                f"({n_bytes / 1e6:.0f} MB on {dev})",
                flush=True,
            )
        return len(self._cache)

    def invalidate_cache(self) -> None:
        """Drop every cached embedding. Call before unfreezing the backbone."""
        self._cache.clear()
        self._cache_sealed = False

    def unfreeze_last_layers(self, n: int) -> list[nn.Parameter]:
        """
        Make the final ``n`` transformer layers trainable and return the newly
        trainable parameters so the caller can add them to the optimizer.

        Clears the cache first: once these weights move, stored embeddings no
        longer correspond to what the backbone would produce.
        """
        self.invalidate_cache()
        self._frozen = False

        layers = self.model.encoder.layer
        unfrozen: list[nn.Parameter] = []
        for layer in layers[max(0, len(layers) - n) :]:
            for p in layer.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    unfrozen.append(p)
        self.model.train()
        return unfrozen

    # -- Forward --------------------------------------------------------------

    def forward(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sequences: list[str] of amino acid sequences
        returns:
            embeddings: (B, L, D)  per-token hidden states
            mask:       (B, L)     1 = real token, 0 = padding
        """
        if not self._cache:
            return self._encode_batch(sequences)

        hits: dict[int, torch.Tensor] = {}
        miss_seqs: list[str] = []
        miss_idx: list[int] = []
        for i, seq in enumerate(sequences):
            emb = self._cache.get(seq)
            if emb is None:
                miss_seqs.append(seq)
                miss_idx.append(i)
            else:
                hits[i] = emb

        # Encode anything the cache does not hold. With a sealed cache these
        # results are used and discarded rather than stored.
        if miss_seqs:
            embs, masks = self._encode_batch(miss_seqs)
            for j, idx in enumerate(miss_idx):
                n_tok = int(masks[j].sum().item())
                emb_j = embs[j, :n_tok]
                if not self._cache_sealed:
                    self._cache[miss_seqs[j]] = emb_j.detach().to(
                        self._cache_device or self.device, self._cache_dtype
                    )
                hits[idx] = emb_j

        # Reassemble into one padded batch.
        dev = self.device
        max_len = max(v.shape[0] for v in hits.values())
        B = len(sequences)
        out = torch.zeros(B, max_len, self.hidden_size, device=dev)
        mask = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        for i, emb in hits.items():
            L = emb.shape[0]
            out[i, :L] = emb.to(dev, torch.float32)
            mask[i, :L] = 1
        return out, mask
