import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# If your package layout differs, adjust this import
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


class NeoantigenDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        required = ["epitope", "mhc_sequence", "mhc_binding"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing}")

        df = df[required].copy()
        df["y"] = df["mhc_binding"].apply(convert_qualitative_measurement)

        # Drop NaNs in label or sequences
        df = df.dropna(subset=["epitope", "mhc_sequence", "y"]).reset_index(drop=True)

        # Basic cleanup (optional)
        df["epitope"] = df["epitope"].astype(str).str.upper().str.replace(" ", "")
        df["mhc_sequence"] = df["mhc_sequence"].astype(str).str.upper().str.replace(" ", "")

        self.epitopes = df["epitope"].tolist()
        self.mhc_seqs = df["mhc_sequence"].tolist()
        self.y = df["y"].astype(float).to_numpy()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "epitope": self.epitopes[idx],
            "mhc_sequence": self.mhc_seqs[idx],
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[List[str], List[str], torch.Tensor]:
    peptides = [b["epitope"] for b in batch]
    mhcs = [b["mhc_sequence"] for b in batch]
    y = torch.stack([b["y"] for b in batch], dim=0)
    return peptides, mhcs, y


def compute_auc(y_true_bin: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    Returns ROC AUC if both classes are present; otherwise returns None.
    """
    # Avoid adding heavy deps if you don't want; sklearn is the simplest.
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None

    if len(np.unique(y_true_bin)) < 2:
        return None
    return float(roc_auc_score(y_true_bin, y_score))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    auc_label_threshold: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    n = 0

    all_probs = []
    all_y = []

    for peptides, mhcs, y in loader:
        y = y.to(device)
        logits = model(peptides, mhcs)  # (B,)
        loss = loss_fn(logits, y)

        probs = torch.sigmoid(logits)

        mae = torch.abs(probs - y).mean()
        mse = torch.square(probs - y).mean()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_mae += float(mae.item()) * bs
        total_mse += float(mse.item()) * bs
        n += bs

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
    all_y = np.concatenate(all_y, axis=0) if all_y else np.array([])

    metrics = {
        "loss": total_loss / max(n, 1),
        "mae": total_mae / max(n, 1),
        "mse": total_mse / max(n, 1),
    }

    if all_y.size > 0:
        y_bin = (all_y >= auc_label_threshold).astype(np.int32)
        auc = compute_auc(y_bin, all_probs)
        if auc is not None:
            metrics["auc"] = auc

    return metrics


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val: float, args: argparse.Namespace):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=str, required=True,
                        help="Path containing train.csv/valid.csv/test.csv")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze ESM backbone (recommended first run).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--auc_label_threshold", type=float, default=0.5,
                        help="Binarize label for AUC: y_bin = (y >= threshold).")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (cuda only).")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_csv = os.path.join(args.split_dir, "train.csv")
    valid_csv = os.path.join(args.split_dir, "valid.csv")
    test_csv = os.path.join(args.split_dir, "test.csv")

    train_ds = NeoantigenDataset(train_csv)
    valid_ds = NeoantigenDataset(valid_csv)
    test_ds = NeoantigenDataset(test_csv)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeoantigenModel(model_name=args.model_name, freeze_backbone=args.freeze_backbone).to(device)

    # Soft-label BCE on logits
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    os.makedirs(args.out_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.out_dir, "best.pt")
    last_ckpt_path = os.path.join(args.out_dir, "last.pt")
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")

    best_val = float("inf")

    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        n = 0

        for peptides, mhcs, y in train_loader:
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    logits = model(peptides, mhcs)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                if args.grad_clip is not None and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(peptides, mhcs)
                loss = loss_fn(logits, y)
                loss.backward()
                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            bs = y.size(0)
            running_loss += float(loss.item()) * bs
            n += bs

        train_loss = running_loss / max(n, 1)
        val_metrics = evaluate(model, valid_loader, device, loss_fn, args.auc_label_threshold)

        # Log
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"epoch {epoch:02d} | train_loss {train_loss:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | val_mae {val_metrics['mae']:.4f} | val_mse {val_metrics['mse']:.4f}"
            + (f" | val_auc {val_metrics['auc']:.4f}" if "auc" in val_metrics else " | val_auc NA")
        )

        # Save last
        save_checkpoint(last_ckpt_path, model, optimizer, epoch, best_val, args)

        # Save best by val_loss (you can swap to val_mae if you prefer)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_val, args)

    # Final test eval using best checkpoint
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, device, loss_fn, args.auc_label_threshold)
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()