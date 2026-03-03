from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class ESMEncoder(nn.Module):
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", freeze=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, sequences):
        """
        sequences: list[str]
        returns: (B, D) pooled embeddings
        """
        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Move to same device as model
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        outputs = self.model(**tokens)  # last_hidden_state: (B, L, D)
        last_hidden = outputs.last_hidden_state
        attention_mask = tokens["attention_mask"]  # (B, L), 1=real, 0=pad

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)                  # (B, D)
        denom = mask.sum(dim=1).clamp_min(1.0)                    # (B, 1)
        pooled = summed / denom

        return pooled

# If want a quick ablation later:
    # Try pooled = last_hidden[:, 0, :] (CLS token) and compare validation AUC/MAE.
    # But default = masked mean.