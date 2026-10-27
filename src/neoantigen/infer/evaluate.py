# src/neoantigen/infer/evaluate.py
"""
Score a trained checkpoint on a held-out split.

``train.py`` already runs the test pass with ``best.pt`` as soon as training
ends. This is the standalone version, for when you want to:

  * re-run the test pass after an interrupted run,
  * score another dataset. Unseen-allele claims require a checkpoint trained
    with those alleles excluded; changing only the evaluation split is invalid,
  * compare two checkpoints on identical data.

It writes the same artifacts the training run does, so
``figures/02_training_progress.ipynb`` can plot the result unchanged.

Usage
-----
    python -m neoantigen.infer.evaluate \
        --ckpt outputs/classI_esm2_650m/best.pt \
        --data_dir data/mhcattnnet/classI_peptide_split \
        --split test \
        --out_dir outputs/classI_esm2_650m_peptide_split
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import hashlib
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
import torch.nn as nn

from neoantigen.models.config import ModelConfig
from neoantigen.models.transformer import NeoantigenModel
from neoantigen.train.dataloader import DataConfig, get_dataloaders, load_splits
from neoantigen.train.train import evaluate
from neoantigen.metrics import per_allele_metrics
from neoantigen.provenance import file_sha256


def verify_unseen_alleles(ckpt, checkpoint_path, target):
    """Verify against the checkpoint's actual training rows, not a new split's train.csv."""
    saved = ckpt.get("args", {})
    manifest_path = Path(checkpoint_path).parent / "run_manifest.json"
    if not manifest_path.exists() or file_sha256(manifest_path) != ckpt.get("run_manifest_sha256"):
        raise ValueError("Unseen-allele evaluation requires the matching run_manifest.json beside the checkpoint")
    manifest = json.loads(manifest_path.read_text())
    cfg = DataConfig(**{k:saved[k] for k in
        ["data_dir", "csv_url", "seed", "split_group", "max_train_rows", "max_eval_rows"] if k in saved})
    train, _, _ = load_splits(cfg)
    digest = hashlib.sha256(train.to_csv(index=False).encode()).hexdigest()
    if digest != manifest["split_sha256"]["train"]:
        raise ValueError("Checkpoint training data no longer match their recorded hash")
    if "mhc_allele" not in train or "mhc_allele" not in target:
        raise ValueError("Allele identities are required for unseen-allele evaluation")
    shared_alleles = set(train.mhc_allele) & set(target.mhc_allele)
    shared_sequences = set(train.mhc_sequence) & set(target.mhc_sequence)
    if shared_alleles or shared_sequences:
        raise ValueError(f"Evaluation is not unseen-allele: {len(shared_alleles)} allele names and "
                         f"{len(shared_sequences)} encoded MHC sequences occurred in checkpoint training")
    return True


def main():
    p = argparse.ArgumentParser(
        description="Score a checkpoint on a held-out split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="Path to best.pt / last.pt")
    p.add_argument("--data_dir", default=None,
                   help="Folder with train.csv / val.csv / test.csv")
    p.add_argument("--csv_url", default=None, help="Single CSV instead of --data_dir")
    p.add_argument("--split_group", choices=["peptide", "allele", "pair"], default=None)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--seed", type=int, default=None,
                   help="split seed; defaults to the seed in the checkpoint")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", default=None,
                   help="Filename prefix for the artifacts (default: the split name)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--auc_label_threshold", type=float, default=0.5)
    p.add_argument("--no_precompute_mhc", dest="precompute_mhc",
                   action="store_false", default=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix = args.prefix or args.split

    # ── Model ─────────────────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "model_cfg" not in ckpt:
        raise ValueError(
            f"{args.ckpt} has no model_cfg — it predates the current trainer. "
            "Rebuild the model manually or retrain."
        )
    model_cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    model_cfg.freeze_backbone = True  # always frozen for scoring
    model = NeoantigenModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {args.ckpt} (epoch {ckpt.get('epoch', '?')}, "
          f"backbone {model_cfg.model_name})", flush=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    # The split seed has to match the run being scored. When the split is
    # derived rather than read from files (--csv_url, or a --data_dir with no
    # val.csv), a different seed re-partitions the data and the "held-out"
    # rows then include pairs the checkpoint trained on.
    seed = args.seed if args.seed is not None else ckpt.get("args", {}).get("seed", 1337)
    print(f"Split seed: {seed}"
          + ("" if args.seed is not None else " (from checkpoint)"), flush=True)
    data_cfg = DataConfig(
        split_group=args.split_group or ckpt.get("args", {}).get("split_group", "peptide"),
        csv_url=args.csv_url,
        data_dir=args.data_dir,
        seed=seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, valid_loader, test_loader, splits = get_dataloaders(data_cfg)
    loader = {"train": train_loader, "valid": valid_loader, "test": test_loader}[args.split]
    split_df = splits[args.split]

    unseen_allele_verified = False
    if data_cfg.split_group == "allele" and args.split != "train":
        unseen_allele_verified = verify_unseen_alleles(ckpt, args.ckpt, split_df)

    if args.split == "train":
        # train_loader is built with shuffle=True, so evaluate() returns rows
        # in loader order while split_df is in file order. The length check
        # below passes (drop_last=False) and every row would be mis-paired.
        from torch.utils.data import DataLoader

        from neoantigen.train.dataloader import NeoantigenDataset, collate_fn

        loader = DataLoader(
            NeoantigenDataset(split_df), batch_size=args.batch_size * 2,
            shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers,
        )

    if args.precompute_mhc:
        model.precompute_mhc(split_df["mhc_sequence"].tolist())

    # ── Evaluate ──────────────────────────────────────────────────────────────
    amp_ctx = (
        torch.amp.autocast(device.type, dtype=torch.bfloat16)
        if (args.amp and device.type == "cuda")
        else contextlib.nullcontext()
    )
    metrics, probs, y = evaluate(
        model, loader, device, nn.BCEWithLogitsLoss(),
        args.auc_label_threshold, amp_ctx, return_preds=True,
    )

    # ── Write the same artifacts a training run does ──────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    metrics.update(unseen_allele_verified=unseen_allele_verified,
                   checkpoint_sha256=file_sha256(args.ckpt), checkpoint_epoch=ckpt.get("epoch"),
                   run_manifest_sha256=ckpt.get("run_manifest_sha256"))
    if "mhc_allele" in split_df:
        per_allele_metrics((y >= args.auc_label_threshold).astype(int), probs,
                          split_df.mhc_allele).to_csv(
            os.path.join(args.out_dir, f"{prefix}_per_allele.csv"), index=False)
    metrics_path = os.path.join(args.out_dir, f"{prefix}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    preds = pd.DataFrame({"y_true": y, "y_pred_prob": probs})
    if len(split_df) == len(preds):
        preds["peptide"] = split_df["epitope"].to_numpy()
        if "mhc_allele" in split_df.columns:
            preds["mhc_allele"] = split_df["mhc_allele"].to_numpy()
    preds_path = os.path.join(args.out_dir, f"{prefix}_predictions.csv")
    preds.to_csv(preds_path, index=False)

    print(f"\n{args.split} metrics:", json.dumps(metrics, indent=2), flush=True)
    print(f"Wrote {metrics_path}\n      {preds_path}", flush=True)


if __name__ == "__main__":
    main()
