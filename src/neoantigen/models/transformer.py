# src/neoantigen/models/transformer.py
import torch
import torch.nn as nn

from .config import ModelConfig
from .esm3 import ESM3Encoder, AttentionPool


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention with a residual connection and layer norm.

    Queries come from ``x``; keys and values come from ``context``.
    Padding positions in ``context`` are excluded via key_padding_mask.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                # (B, Lq, D)
        context: torch.Tensor,          # (B, Lkv, D)
        key_padding_mask: torch.Tensor, # (B, Lkv)  True = ignore
    ) -> torch.Tensor:
        # Run MHA in float32: attention scores (Q @ K^T) overflow in float16
        # for 512-dim attention under torch.amp.autocast("cuda").
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            attn_out, _ = self.attn(
                query=x,
                key=context.float(),
                value=context.float(),
                key_padding_mask=key_padding_mask,
            )
            return self.norm(x + self.drop(attn_out))  # (B, Lq, D)


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with a residual connection and layer norm.

    Used after cross-attention so each modality can integrate the newly
    acquired cross-modal context across its own token positions.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                # (B, L, D)
        key_padding_mask: torch.Tensor, # (B, L)  True = ignore (pad positions)
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            attn_out, _ = self.attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
            )
            return self.norm(x + self.drop(attn_out))  # (B, L, D)


class InteractionLayer(nn.Module):
    """
    One round of bidirectional cross-attention + optional self-attention.

    Each layer:
      1. Peptide attends to MHC (cross-attn)
      2. MHC attends to updated peptide (cross-attn)
      3. Optional: self-attention within peptide tokens
      4. Optional: self-attention within MHC tokens

    Stacking N of these allows iterative refinement of both representations.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        use_self_attn: bool,
    ):
        super().__init__()
        self.pep_to_mhc = CrossAttention(d_model, num_heads, dropout)
        self.mhc_to_pep = CrossAttention(d_model, num_heads, dropout)
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.pep_self = SelfAttention(d_model, num_heads, dropout)
            self.mhc_self = SelfAttention(d_model, num_heads, dropout)

    def forward(
        self,
        pep_h: torch.Tensor,   # (B, Lp, D)
        mhc_h: torch.Tensor,   # (B, Lm, D)
        pep_kpm: torch.Tensor, # (B, Lp)  True = pad
        mhc_kpm: torch.Tensor, # (B, Lm)  True = pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pep_h = self.pep_to_mhc(pep_h, mhc_h, key_padding_mask=mhc_kpm)
        mhc_h = self.mhc_to_pep(mhc_h, pep_h, key_padding_mask=pep_kpm)
        if self.use_self_attn:
            pep_h = self.pep_self(pep_h, key_padding_mask=pep_kpm)
            mhc_h = self.mhc_self(mhc_h, key_padding_mask=mhc_kpm)
        return pep_h, mhc_h


class NeoantigenModel(nn.Module):
    """
    Neoantigen MHC-peptide binding predictor.

    Architecture
    ------------
    1. Shared ESM3 backbone encodes both sequences → per-token embeddings (B, L, 1536).
    2. Separate linear projections → d_proj for peptide and MHC (captures
       distinct positional roles of short peptides vs. long binding grooves).
    3. N stacked InteractionLayers: each does bidirectional cross-attention
       (peptide attends to MHC and vice versa), optionally followed by
       self-attention so tokens integrate the new cross-modal context.
    4. Learned attention pooling collapses each modality to a single vector.
    5. MLP head over [pep, mhc] or [pep, mhc, pep*mhc] → scalar binding logit.

    Configuration is driven by a ``ModelConfig`` instance to support ablations.
    """

    def __init__(self, cfg: ModelConfig | None = None, **kwargs):
        super().__init__()
        # Backward-compat: accept old-style keyword args from predict.py / checkpoints
        if cfg is None:
            cfg = ModelConfig(
                model_name=kwargs.get("model_name", "esm3_sm_open_v1"),
                freeze_backbone=kwargs.get("freeze_backbone", True),
                d_proj=kwargs.get("d_proj", 512),
                num_heads=kwargs.get("num_heads", 8),
                dropout=kwargs.get("dropout", 0.1),
            )
        self.cfg = cfg

        self.encoder = ESM3Encoder(
            model_id=cfg.model_name,
            freeze=cfg.freeze_backbone,
            use_cache=cfg.use_embedding_cache,
            cache_dir=cfg.cache_dir if cfg.use_embedding_cache else None,
        )
        D = self.encoder.hidden_size  # 1536

        # Separate projections for peptide vs. MHC tokens
        self.pep_proj = nn.Linear(D, cfg.d_proj)
        self.mhc_proj = nn.Linear(D, cfg.d_proj)

        # Stacked bidirectional interaction layers
        self.layers = nn.ModuleList([
            InteractionLayer(cfg.d_proj, cfg.num_heads, cfg.dropout, cfg.use_self_attn)
            for _ in range(cfg.num_interaction_layers)
        ])

        # Learned attention pooling (separate per modality)
        self.pep_pool = AttentionPool(cfg.d_proj)
        self.mhc_pool = AttentionPool(cfg.d_proj)

        # Prediction head: optionally richer with element-wise product
        head_in = cfg.d_proj * 3 if cfg.use_product_head else cfg.d_proj * 2
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 1),
        )

    def forward(self, peptides: list, mhc_seqs: list) -> torch.Tensor:
        # ── 1. Encode ──────────────────────────────────────────────────────────
        pep_tok, pep_mask = self.encoder(peptides)   # (B, Lp, D), (B, Lp)
        mhc_tok, mhc_mask = self.encoder(mhc_seqs)  # (B, Lm, D), (B, Lm)

        # ── 2. Project ─────────────────────────────────────────────────────────
        pep_h = self.pep_proj(pep_tok)  # (B, Lp, d_proj)
        mhc_h = self.mhc_proj(mhc_tok) # (B, Lm, d_proj)

        # nn.MultiheadAttention key_padding_mask: True = ignore (pad positions)
        pep_kpm = (pep_mask == 0)  # (B, Lp)
        mhc_kpm = (mhc_mask == 0) # (B, Lm)

        # ── 3. Stacked interaction layers ──────────────────────────────────────
        for layer in self.layers:
            pep_h, mhc_h = layer(pep_h, mhc_h, pep_kpm, mhc_kpm)

        # ── 4. Attention pooling ───────────────────────────────────────────────
        pep_emb = self.pep_pool(pep_h, pep_mask)  # (B, d_proj)
        mhc_emb = self.mhc_pool(mhc_h, mhc_mask) # (B, d_proj)

        # ── 5. Predict ─────────────────────────────────────────────────────────
        if self.cfg.use_product_head:
            x = torch.cat([pep_emb, mhc_emb, pep_emb * mhc_emb], dim=1)  # (B, 3*d_proj)
        else:
            x = torch.cat([pep_emb, mhc_emb], dim=1)                      # (B, 2*d_proj)

        return self.head(x).squeeze(-1)  # (B,)
