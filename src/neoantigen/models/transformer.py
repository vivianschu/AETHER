# src/neoantigen/models/transformer.py
import torch
import torch.nn as nn

from .config import ModelConfig
from .pooling import AttentionPool


def build_backbone(cfg: ModelConfig) -> nn.Module:
    """
    Construct the sequence backbone named by ``cfg.model_name``.

    ESM3 is imported lazily: it needs EvolutionaryScale's ``esm`` package, which
    is not required for the ESM2 path and should not break the import if absent.
    """
    if cfg.model_name.startswith("facebook/esm2"):
        from .esm2 import ESM2Encoder

        return ESM2Encoder(
            model_id=cfg.model_name,
            freeze=cfg.freeze_backbone,
            max_length=cfg.max_length,
            cache_device=cfg.cache_device,
        )

    if cfg.model_name.startswith("esm3"):
        from .esm3 import ESM3Encoder

        return ESM3Encoder(
            model_id=cfg.model_name,
            freeze=cfg.freeze_backbone,
            use_cache=cfg.use_embedding_cache,
            cache_dir=cfg.cache_dir if cfg.use_embedding_cache else None,
        )

    raise ValueError(
        f"Unrecognized model_name {cfg.model_name!r}. Expected a "
        f"'facebook/esm2_*' or 'esm3*' checkpoint."
    )


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
    1. Shared ESM2 backbone encodes both sequences → per-token embeddings (B, L, D).
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
            known = {"model_name", "freeze_backbone", "d_proj", "num_heads", "dropout"}
            unknown = set(kwargs) - known
            if unknown:
                # Silently ignoring these builds a different architecture than
                # the caller asked for, and the state dict still loads cleanly.
                raise TypeError(
                    f"NeoantigenModel got unsupported keyword(s) {sorted(unknown)}. "
                    f"Pass a ModelConfig instead of keyword arguments."
                )
            cfg = ModelConfig(
                model_name=kwargs.get("model_name", ModelConfig.model_name),
                freeze_backbone=kwargs.get("freeze_backbone", True),
                d_proj=kwargs.get("d_proj", 512),
                num_heads=kwargs.get("num_heads", 8),
                dropout=kwargs.get("dropout", 0.1),
            )
        self.cfg = cfg

        self.encoder = build_backbone(cfg)
        D = self.encoder.hidden_size  # 1280 for esm2_t33_650M

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

        # ── Phase-2 TCR arm ───────────────────────────────────────────────────
        # Built only when requested, so a phase-1 checkpoint loads unchanged.
        if cfg.use_tcr_head:
            self.tcr_proj = nn.Linear(D, cfg.d_proj)
            self.tcr_layers = nn.ModuleList([
                InteractionLayer(cfg.d_proj, cfg.num_heads, cfg.dropout,
                                 cfg.use_self_attn)
                for _ in range(cfg.num_tcr_interaction_layers)
            ])
            self.tcr_pool = AttentionPool(cfg.d_proj)
            self.tcr_head = nn.Sequential(
                nn.Linear(cfg.d_proj * 3, 256),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(256, 1),
            )

    def precompute_mhc(self, mhc_seqs: list, batch_size: int = 8) -> int:
        """
        Pin frozen embeddings for the unique MHC sequences in ``mhc_seqs``.

        The class-I split holds ~150 distinct MHC sequences across ~344k rows,
        so this removes almost all of the backbone's work. Returns the number of
        sequences cached, or 0 if the backbone does not support caching.
        """
        if not hasattr(self.encoder, "precompute"):
            return 0
        return self.encoder.precompute(mhc_seqs, batch_size=batch_size)

    def encode_pmhc(self, peptides: list, mhc_seqs: list):
        """
        Shared trunk: encode, project, and run the interaction layers.

        Returns the refined token states and their masks, so both heads work
        from the same peptide-MHC representation.
        """
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

        return pep_h, mhc_h, pep_mask, mhc_mask

    def forward(self, peptides: list, mhc_seqs: list,
                cdr3b_seqs: list | None = None) -> torch.Tensor:
        """
        Dispatch to the objective implied by the arguments.

        Without ``cdr3b_seqs`` this is phase 1 -- does this peptide bind this
        MHC -- and the binding branch below never sees a receptor. Binding
        precedes any T cell, so CDR3b there could only carry assay-selection
        bias. With ``cdr3b_seqs`` it is the phase-2 recognition objective,
        which is a different label, not a better-informed binding score.

        Routing through ``forward`` rather than calling ``forward_tcr``
        directly is what keeps DDP's gradient hooks attached.
        """
        if cdr3b_seqs is not None:
            return self.forward_tcr(peptides, mhc_seqs, cdr3b_seqs)

        pep_h, mhc_h, pep_mask, mhc_mask = self.encode_pmhc(peptides, mhc_seqs)

        # ── 4. Attention pooling ───────────────────────────────────────────────
        pep_emb = self.pep_pool(pep_h, pep_mask)  # (B, d_proj)
        mhc_emb = self.mhc_pool(mhc_h, mhc_mask) # (B, d_proj)

        # ── 5. Predict ─────────────────────────────────────────────────────────
        if self.cfg.use_product_head:
            x = torch.cat([pep_emb, mhc_emb, pep_emb * mhc_emb], dim=1)  # (B, 3*d_proj)
        else:
            x = torch.cat([pep_emb, mhc_emb], dim=1)                      # (B, 2*d_proj)

        return self.head(x).squeeze(-1)  # (B,)

    def forward_tcr(self, peptides: list, mhc_seqs: list,
                    cdr3b_seqs: list) -> torch.Tensor:
        """
        Phase-2 objective: does this receptor recognize this peptide-MHC?

        The CDR3b tokens cross-attend to the peptide-MHC complex -- the two
        token sets concatenated, since a CDR3 loop contacts peptide and groove
        as one surface. Returns a recognition logit, never a binding one.
        """
        if not self.cfg.use_tcr_head:
            raise RuntimeError(
                "forward_tcr requires use_tcr_head=True in the ModelConfig."
            )

        pep_h, mhc_h, pep_mask, mhc_mask = self.encode_pmhc(peptides, mhc_seqs)

        tcr_tok, tcr_mask = self.encoder(cdr3b_seqs)  # (B, Lt, D), (B, Lt)
        tcr_h = self.tcr_proj(tcr_tok)                # (B, Lt, d_proj)

        cplx_h = torch.cat([pep_h, mhc_h], dim=1)           # (B, Lp+Lm, d_proj)
        cplx_mask = torch.cat([pep_mask, mhc_mask], dim=1)  # (B, Lp+Lm)
        tcr_kpm = (tcr_mask == 0)
        cplx_kpm = (cplx_mask == 0)

        for layer in self.tcr_layers:
            tcr_h, cplx_h = layer(tcr_h, cplx_h, tcr_kpm, cplx_kpm)

        tcr_emb = self.tcr_pool(tcr_h, tcr_mask)            # (B, d_proj)
        # Pool the peptide half of the complex, so the pMHC side of this head
        # has the same geometry as the phase-1 head's peptide embedding.
        pmhc_emb = self.pep_pool(cplx_h[:, : pep_h.shape[1]], pep_mask)

        x = torch.cat([pmhc_emb, tcr_emb, pmhc_emb * tcr_emb], dim=1)
        return self.tcr_head(x).squeeze(-1)  # (B,)

    def freeze_trunk(self, freeze: bool = True) -> int:
        """
        Freeze everything the binding logit depends on, for phase-2 training.

        Keeps phase-1 predictions bit-identical while the TCR arm trains, so a
        phase-2 run cannot quietly move benchmark numbers. Returns the number
        of parameter tensors touched.
        """
        # The encoder needs its own flag flipped, not just requires_grad:
        # that is what pins it to eval() and picks the no_grad path.
        if hasattr(self.encoder, "set_frozen"):
            self.encoder.set_frozen(freeze)

        trunk = [self.encoder, self.pep_proj, self.mhc_proj, self.layers,
                 self.pep_pool, self.mhc_pool, self.head]
        n = 0
        for module in trunk:
            for p in module.parameters():
                p.requires_grad = not freeze
                n += 1
        return n
