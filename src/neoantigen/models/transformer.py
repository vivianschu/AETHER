import torch
import torch.nn as nn
from .esm import ESMEncoder

class NeoantigenModel(nn.Module):
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", freeze_backbone=True):
        super().__init__()

        # ONE shared encoder (shared weights). Run it twice.
        self.shared_encoder = ESMEncoder(model_name=model_name, freeze=freeze_backbone)

        d = self.shared_encoder.model.config.hidden_size  # 1280 for esm2_t33_650M

        self.head = nn.Sequential(
            nn.Linear(d * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)  # logits
        )

    def forward(self, peptides, mhc_seqs):
        pep_emb = self.shared_encoder(peptides)   # (B, D)
        mhc_emb = self.shared_encoder(mhc_seqs)   # (B, D)

        x = torch.cat([pep_emb, mhc_emb], dim=1)  # (B, 2D)
        logits = self.head(x).squeeze(-1)         # (B,)
        return logits