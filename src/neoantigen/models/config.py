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
    use_embedding_cache : bool
        Cache ESM3 per-token embeddings in memory (and optionally on disk).
        Only valid when freeze_backbone=True; automatically disabled on
        progressive unfreezing.
    """

    # ── Backbone ──────────────────────────────────────────────────────────────
    model_name: str = "esm3_sm_open_v1"
    freeze_backbone: bool = True

    # ── Cache ─────────────────────────────────────────────────────────────────
    use_embedding_cache: bool = False
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
