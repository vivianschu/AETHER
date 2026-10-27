# src/neoantigen/models/config.py
import json
from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    """Architecture hyperparameters for NeoantigenModel.

    Designed for ablation studies: toggle one field at a time to isolate
    the contribution of each architectural component.

    Ablation knobs
    --------------
    num_interaction_layers : int
        Number of stacked bidirectional cross-attention rounds (default 1).
        Increasing this lets peptide and MHC representations iteratively
        refine each other.
    use_self_attn : bool
        If True, each interaction layer also runs self-attention within each
        modality *after* cross-attention, so tokens can integrate the
        newly-acquired cross-modal context before the next round.
    use_product_head : bool
        If True, the MLP head sees [pep, mhc, pep*mhc] instead of [pep, mhc].
        The element-wise product is a lightweight interaction term that has
        proved useful in NLI and siamese network tasks.
    precompute_mhc : bool
        Run the frozen backbone once over the unique MHC sequences and pin the
        result. The class-I split has ~150 distinct MHC sequences across ~344k
        rows, so this removes nearly all of the backbone's per-step cost.
        Only valid when freeze_backbone=True; disabled on progressive unfreezing.
    use_embedding_cache : bool
        ESM3-only legacy cache (in-memory, optionally on disk).
    """

    # ── Backbone ──────────────────────────────────────────────────────────────
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    freeze_backbone: bool = True
    max_length: int | None = None     # truncate tokenized sequences; None = no limit

    # ── Cache ─────────────────────────────────────────────────────────────────
    precompute_mhc: bool = True       # pin frozen embeddings for unique MHC seqs
    cache_device: str | None = None   # None = model device; "cpu" to save VRAM
    use_embedding_cache: bool = False  # ESM3-only legacy cache
    cache_dir: str = ".cache/esm3"

    # ── Projection ────────────────────────────────────────────────────────────
    d_proj: int = 512

    # ── Interaction layers ────────────────────────────────────────────────────
    num_heads: int = 8
    dropout: float = 0.1
    num_interaction_layers: int = 1   # stacked bidirectional cross-attn rounds
    use_self_attn: bool = False        # self-attn within each modality post cross-attn

    # ── Head ──────────────────────────────────────────────────────────────────
    use_product_head: bool = False     # [pep, mhc, pep*mhc] instead of [pep, mhc]

    # ── TCR recognition head (phase 2) ────────────────────────────────────────
    # A second head over the same trunk, predicting whether a given receptor
    # recognizes the peptide-MHC complex. It is a *separate objective*, not an
    # input to binding: CDR3b never reaches the binding logit, because the TCR
    # is downstream of binding and would only leak which pairs were assayed.
    use_tcr_head: bool = False
    num_tcr_interaction_layers: int = 1  # cross-attn rounds between CDR3b and pMHC

    # ── Serialization ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
