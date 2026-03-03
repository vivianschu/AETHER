import torch
import torch.nn as nn
from .esm3 import ESM3Encoder, AttentionPool


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention with a residual connection and layer norm.

    Queries come from `x`; keys and values come from `context`.
    Padding positions in `context` are excluded via key_padding_mask.
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
        x: torch.Tensor,               # (B, Lq, D)   query sequence
        context: torch.Tensor,         # (B, Lkv, D)  key/value sequence
        key_padding_mask: torch.Tensor, # (B, Lkv)    True = ignore position
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


class NeoantigenModel(nn.Module):
    """
    Neoantigen MHC-peptide binding predictor.

    Architecture
    ------------
    1. Shared ESM3 backbone encodes both peptide and MHC sequences into
       per-token embeddings (B, L, 1536).
    2. Separate linear projections reduce each to d_proj so the model can
       learn distinct representations for short peptides vs. long MHC grooves.
    3. Bidirectional cross-attention lets peptide tokens attend to MHC
       context (and vice versa), modelling direct physical interaction.
    4. Learned attention pooling collapses each token sequence to a single
       vector, weighting positions by their binding relevance.
    5. A small MLP head produces a scalar binding logit.
    """

    def __init__(
        self,
        model_name: str = "esm3_sm_open_v1",
        freeze_backbone: bool = True,
        d_proj: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared ESM3 backbone (run once per sequence)
        self.encoder = ESM3Encoder(model_id=model_name, freeze=freeze_backbone)
        D = self.encoder.hidden_size  # 1536

        # Separate projections for peptide vs. MHC tokens
        self.pep_proj = nn.Linear(D, d_proj)
        self.mhc_proj = nn.Linear(D, d_proj)

        # Bidirectional cross-attention
        self.pep_to_mhc = CrossAttention(d_proj, num_heads, dropout)  # pep queries mhc
        self.mhc_to_pep = CrossAttention(d_proj, num_heads, dropout)  # mhc queries pep

        # Learned attention pooling (separate per modality)
        self.pep_pool = AttentionPool(d_proj)
        self.mhc_pool = AttentionPool(d_proj)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_proj * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, peptides: list, mhc_seqs: list) -> torch.Tensor:
        # ── 1. Encode ──────────────────────────────────────────────────────
        pep_tok, pep_mask = self.encoder(peptides)   # (B, Lp, D), (B, Lp)
        mhc_tok, mhc_mask = self.encoder(mhc_seqs)  # (B, Lm, D), (B, Lm)

        # ── 2. Project ─────────────────────────────────────────────────────
        pep_h = self.pep_proj(pep_tok)  # (B, Lp, d_proj)
        mhc_h = self.mhc_proj(mhc_tok) # (B, Lm, d_proj)

        # nn.MultiheadAttention key_padding_mask: True = ignore (pad positions)
        pep_kpm = (pep_mask == 0)  # (B, Lp)
        mhc_kpm = (mhc_mask == 0) # (B, Lm)

        # ── 3. Cross-attention ─────────────────────────────────────────────
        pep_ctx = self.pep_to_mhc(pep_h, mhc_h, key_padding_mask=mhc_kpm)  # (B, Lp, d_proj)
        mhc_ctx = self.mhc_to_pep(mhc_h, pep_h, key_padding_mask=pep_kpm)  # (B, Lm, d_proj)

        # ── 4. Attention pooling ───────────────────────────────────────────
        pep_emb = self.pep_pool(pep_ctx, pep_mask)  # (B, d_proj)
        mhc_emb = self.mhc_pool(mhc_ctx, mhc_mask) # (B, d_proj)

        # ── 5. Predict ─────────────────────────────────────────────────────
        x = torch.cat([pep_emb, mhc_emb], dim=1)  # (B, 2*d_proj)
        return self.head(x).squeeze(-1)            # (B,)
