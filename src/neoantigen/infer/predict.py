import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from neoantigen.models.transformer import NeoantigenModel
from neoantigen.train.dataloader import resolve_sequence_columns


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
        # Resolves peptide/MHC columns by alias, so both the MHCAttnNet schema
        # (peptide / mhc_amino_acid) and the AETHER schema (epitope /
        # mhc_sequence) load unchanged, and cleans the sequences the same way
        # training did.
        raw = pd.read_csv(csv_path)
        # Keep the input row number: clean_sequences drops rows whose peptide
        # or MHC is empty after cleaning and reindexes, so the output CSV can
        # be shorter than the input with no way to realign it by position.
        raw = raw.assign(input_row=np.arange(len(raw)))
        self.df = resolve_sequence_columns(raw)
        if len(self.df) < len(raw):
            print(f"WARNING: dropped {len(raw) - len(self.df):,} of {len(raw):,} "
                  f"input rows with an empty peptide or MHC sequence. The "
                  f"output carries an `input_row` column to realign them.",
                  flush=True)

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
    parser.add_argument("--no_precompute_mhc", action="store_true",
                        help="Skip pinning embeddings for the unique MHC sequences.")
    parser.add_argument("--precompute_batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    from neoantigen.models.config import ModelConfig

    # Load ModelConfig from checkpoint (new-style) or fall back to legacy args dict
    if "model_cfg" in ckpt:
        model_cfg = ModelConfig.from_dict(ckpt["model_cfg"])
        # Always freeze for inference; the cache is rebuilt below from this
        # file's own MHC sequences rather than the training set's.
        model_cfg.freeze_backbone = True
        model_cfg.use_embedding_cache = False
        model = NeoantigenModel(model_cfg).to(device)
    else:
        # Legacy checkpoints: reconstruct every architecture field from the
        # saved args. Several of them -- num_heads above all -- change no
        # tensor shape, so load_state_dict reports a clean load while the
        # weights are reinterpreted and the predictions shift.
        ckpt_args = ckpt.get("args", {})
        fields = ("model_name", "d_proj", "num_heads", "dropout", "max_length",
                  "num_interaction_layers", "use_self_attn", "use_product_head")
        model_cfg = ModelConfig.from_dict(
            {k: ckpt_args[k] for k in fields if k in ckpt_args}
        )
        model_cfg.freeze_backbone = True
        model_cfg.use_embedding_cache = False
        missing = [k for k in fields if k not in ckpt_args]
        if missing:
            print(f"WARNING: checkpoint does not record {missing}; "
                  f"using ModelConfig defaults for them.", flush=True)
        model = NeoantigenModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = PredictDataset(args.input_csv)

    # Same trick as training: an input file typically covers a handful of
    # alleles across many peptides, so encode each MHC sequence once.
    if not args.no_precompute_mhc:
        n = model.precompute_mhc(ds.df["mhc_sequence"].tolist(),
                                 batch_size=args.precompute_batch_size)
        if n:
            print(f"Precomputed {n} unique MHC embeddings.")
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
            # bfloat16 matches training; torch.cuda.amp.autocast is deprecated.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(peptides, mhcs)
        else:
            logits = model(peptides, mhcs)

        logits = logits.float()
        probs = torch.sigmoid(logits)

        logits_np = logits.detach().cpu().numpy().astype(np.float32)
        probs_np = probs.detach().cpu().numpy().astype(np.float32)

        for i, row_idx in enumerate(idxs):
            pred_logit[row_idx] = logits_np[i]
            pred_prob[row_idx] = probs_np[i]

    out_df = ds.df.copy()
    out_df["pred_logit"] = pred_logit
    out_df["pred_prob"] = pred_prob

    # dirname is empty for a bare filename, which makedirs rejects.
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()