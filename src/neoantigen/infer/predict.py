import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from neoantigen.models.transformer import NeoantigenModel


def convert_qualitative_measurement(x):
    if x == "Positive":
        return 1.0
    elif x == "Negative":
        return 0.0
    elif x == "Positive-Low":
        return 0.25
    elif x == "Positive-High":
        return 0.75
    elif x == "Positive-Intermediate":
        return 0.5
    else:
        return np.nan


class PredictDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        required = ["epitope", "mhc_sequence"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing}")

        self.df = df.copy()
        self.df["epitope"] = self.df["epitope"].astype(str).str.upper().str.replace(" ", "")
        self.df["mhc_sequence"] = self.df["mhc_sequence"].astype(str).str.upper().str.replace(" ", "")

        # optional label column
        if "mhc_binding" in self.df.columns:
            self.df["mhc_binding_label"] = self.df["mhc_binding"].apply(convert_qualitative_measurement)
        else:
            self.df["mhc_binding_label"] = np.nan

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "epitope": row["epitope"],
            "mhc_sequence": row["mhc_sequence"],
            "idx": idx,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[int]]:
    peptides = [b["epitope"] for b in batch]
    mhcs = [b["mhc_sequence"] for b in batch]
    idxs = [b["idx"] for b in batch]
    return peptides, mhcs, idxs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model_name = ckpt_args.get("model_name", "facebook/esm2_t33_650M_UR50D")
    freeze_backbone = ckpt_args.get("freeze_backbone", True)

    model = NeoantigenModel(model_name=model_name, freeze_backbone=freeze_backbone).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = PredictDataset(args.input_csv)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    pred_logit = np.zeros(len(ds), dtype=np.float32)
    pred_prob = np.zeros(len(ds), dtype=np.float32)

    use_amp = args.amp and device.type == "cuda"

    for peptides, mhcs, idxs in loader:
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(peptides, mhcs)
        else:
            logits = model(peptides, mhcs)

        probs = torch.sigmoid(logits)

        logits_np = logits.detach().cpu().numpy().astype(np.float32)
        probs_np = probs.detach().cpu().numpy().astype(np.float32)

        for i, row_idx in enumerate(idxs):
            pred_logit[row_idx] = logits_np[i]
            pred_prob[row_idx] = probs_np[i]

    out_df = ds.df.copy()
    out_df["pred_logit"] = pred_logit
    out_df["pred_prob"] = pred_prob

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()