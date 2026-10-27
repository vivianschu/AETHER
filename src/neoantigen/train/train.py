# src/neoantigen/train/train.py
"""
Train AETHER (ESM2 backbone + bidirectional cross-attention) on peptide-MHC
binding.

All plotting lives in notebooks under ``figures/``. This script's job is to
emit the artifacts those notebooks read:

    <out_dir>/metrics.jsonl        one JSON row per evaluation, appended live
    <out_dir>/config.json          the full resolved run configuration
    <out_dir>/best.pt, last.pt     checkpoints
    <out_dir>/test_metrics.json    final test metrics
    <out_dir>/test_predictions.csv per-row y_true / y_pred_prob (+ allele)

``metrics.jsonl`` is flushed after every evaluation, so the training-progress
notebook can be re-run mid-training to see the curves so far.
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from neoantigen.models.config import ModelConfig
from neoantigen.models.transformer import NeoantigenModel

from .dataloader import DataConfig, get_dataloaders, unique_mhc_sequences, audit_splits
from neoantigen.metrics import ranking_metrics, per_allele_metrics
from neoantigen.provenance import write_manifest, file_sha256
from .tcr_dataloader import TCRDataConfig, get_tcr_dataloaders

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


# ── Metrics ───────────────────────────────────────────────────────────────────

def binary_metrics(y_true_bin: np.ndarray, y_score: np.ndarray,
                   score_threshold: float = 0.5) -> Dict[str, float]:
    """
    ROC AUC, average precision, PPV@100 and accuracy, with scored-row support.

    ``score_threshold`` cuts the predicted probability and should match the
    threshold used to binarize the labels. With graded targets and a label cut
    of 0.25, a model predicting 0.25 on Positive-Low rows is perfectly
    calibrated, yet a fixed 0.5 probability cut counts every one of them as a
    false negative -- accuracy then disagrees with an AUC of 1.0.
    """
    out = ranking_metrics(y_true_bin, y_score)
    if "auroc" in out:
        out["auc"] = out["auroc"]  # compatibility with existing notebooks
    if len(y_true_bin):
        out["acc"] = float(np.mean(y_true_bin == (y_score >= score_threshold)))
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn: nn.Module,
    label_threshold: float,
    amp_ctx,
    return_preds: bool = False,
    max_batches: int = 0,
):
    model.eval()

    total_loss = total_mae = total_mse = 0.0
    n = 0
    all_probs, all_y = [], []

    # A batch is (peptides, mhcs, y) for binding, (peptides, mhcs, cdr3b, y)
    # for TCR recognition; the model dispatches on the extra field.
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        *inputs, y = batch
        y = y.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(*inputs)
        logits = logits.float()
        loss = loss_fn(logits, y)
        probs = torch.sigmoid(logits)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_mae += float(torch.abs(probs - y).mean().item()) * bs
        total_mse += float(torch.square(probs - y).mean().item()) * bs
        n += bs

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    probs_arr = np.concatenate(all_probs) if all_probs else np.array([])
    y_arr = np.concatenate(all_y) if all_y else np.array([])

    metrics: Dict[str, float] = {
        "loss": total_loss / max(n, 1),
        "mae": total_mae / max(n, 1),
        "mse": total_mse / max(n, 1),
    }
    metrics.update(binary_metrics((y_arr >= label_threshold).astype(np.int32),
                                  probs_arr, score_threshold=label_threshold))

    frame = getattr(loader.dataset, "df", None)
    if frame is not None and "mhc_allele" in frame:
        # Validation/test loaders are sequential. Partial evaluations use their prefix.
        metrics.update(ranking_metrics((y_arr >= label_threshold).astype(int),
                       probs_arr, frame["mhc_allele"].to_numpy()[:len(y_arr)]))
    model.train()
    if return_preds:
        return metrics, probs_arr, y_arr
    return metrics


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, epoch, best_score, args, model_cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    underlying = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model_state": underlying.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "args": vars(args),
            "model_cfg": model_cfg.to_dict(),
            "run_manifest_sha256": getattr(args, "run_manifest_sha256", None),
            "validation_metrics": getattr(args, "validation_metrics", None),
        },
        path,
    )


# ── Distributed setup ─────────────────────────────────────────────────────────

def setup_distributed():
    """
    Initialize DDP from torchrun environment variables.

    Returns (rank, world_size, is_distributed); (0, 1, False) for a plain run.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False
    local_rank = int(os.environ["LOCAL_RANK"])
    # Rank 0 serializes a multi-GB checkpoint at each epoch boundary while the
    # other ranks wait in the next collective. The default watchdog aborts them
    # (SIGABRT) if that takes too long, so allow for slow shared storage.
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=int(os.environ.get("AETHER_DDP_TIMEOUT_S", 7200))),
    )
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), True


# ── LR scheduler ──────────────────────────────────────────────────────────────

def build_scheduler(args, optimizer, steps_per_epoch):
    """"none" (constant), "cosine" (optional linear warmup), or "onecycle"."""
    if args.lr_scheduler == "none":
        return None

    total_steps = max(1, args.epochs * steps_per_epoch)

    if args.lr_scheduler == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR

        return OneCycleLR(
            optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs
        )

    if args.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            SequentialLR,
        )

        warmup = min(args.warmup_steps, total_steps - 1)
        if warmup > 0:
            return SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                    CosineAnnealingLR(optimizer, T_max=total_steps - warmup),
                ],
                milestones=[warmup],
            )
        return CosineAnnealingLR(optimizer, T_max=total_steps)

    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler!r}")


# ── Arguments ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train AETHER for peptide-MHC binding prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="JSON config file; values become defaults, overridable by CLI flags.")

    # Data
    p.add_argument("--data_dir", type=str, default=None,
                   help="Folder with train.csv / val.csv / test.csv (preferred).")
    p.add_argument("--csv_url", type=str, default=None,
                   help="Single CSV (path or URL) to split on (epitope, mhc) pairs.")
    p.add_argument("--split_group", choices=["peptide", "allele", "pair"], default="peptide",
                   help="Grouping for derived splits and leakage checks for pre-split data")
    p.add_argument("--out_dir", type=str, required=True,
                   help="New run directory; an existing path is rejected to prevent mixed runs")

    # ── Phase 2: TCR recognition ──────────────────────────────────────────────
    p.add_argument("--task", type=str, default="binding", choices=["binding", "tcr"],
                   help="binding = phase 1 (peptide+MHC). tcr = phase 2 "
                        "(peptide+MHC+CDR3b), a separate objective and head.")
    p.add_argument("--tcr_csv", type=str, default=None,
                   help="cedar_tcr.csv; required when --task tcr")
    p.add_argument("--tcr_neg_ratio", type=float, default=5.0,
                   help="shuffled negatives per observed positive, drawn within "
                        "each split against other real epitopes")
    p.add_argument("--tcr_split", type=str, default="epitope",
                   choices=["epitope", "random"],
                   help="epitope = test epitopes unseen in training (the real "
                        "task). random = diagnostic only.")
    p.add_argument("--tcr_max_pos_per_epitope", type=int, default=1000,
                   help="cap positives contributed by any one epitope (0 = off). "
                        "A negative needs a receptor seen against another "
                        "epitope, so a dominant epitope cannot reach the "
                        "negative ratio and ends up with a higher positive "
                        "rate than the rest -- which the model reads off "
                        "instead of the receptor.")
    p.add_argument("--tcr_drop_viral", action="store_true",
                   help="drop viral-antigen rows (95%% of the corpus; usually keep them)")
    p.add_argument("--tcr_neoantigen_only", action="store_true",
                   help="restrict to neoantigen rows -- a cancer-only eval set")
    p.add_argument("--init_from", type=str, default=None,
                   help="checkpoint to initialise from, e.g. a phase-1 best.pt")
    p.add_argument("--freeze_trunk", action="store_true",
                   help="train only the TCR arm, leaving phase-1 predictions "
                        "bit-identical")
    p.add_argument("--auc_label_threshold", type=float, default=0.5)
    p.add_argument("--max_train_rows", type=int, default=0,
                   help=">0 subsamples the train split — useful for smoke tests.")
    p.add_argument("--max_eval_rows", type=int, default=0,
                   help=">0 subsamples val/test.")

    # Model
    p.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--no_freeze_backbone", dest="freeze_backbone",
                   action="store_false", default=True,
                   help="Fine-tune the whole backbone (disables MHC precompute).")
    p.add_argument("--max_length", type=int, default=None,
                   help="Truncate tokenized sequences to this many tokens.")
    p.add_argument("--d_proj", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_interaction_layers", type=int, default=2,
                   help="Stacked bidirectional cross-attention rounds.")
    p.add_argument("--use_self_attn", action="store_true",
                   help="Self-attention within each modality after cross-attention.")
    p.add_argument("--num_tcr_interaction_layers", type=int, default=2,
                   help="cross-attn rounds between CDR3b and the pMHC complex")
    p.add_argument("--use_product_head", action="store_true",
                   help="Head sees [pep, mhc, pep*mhc] instead of [pep, mhc].")

    # Caching
    p.add_argument("--no_precompute_mhc", dest="precompute_mhc",
                   action="store_false", default=True,
                   help="Skip pinning frozen embeddings for the unique MHC sequences.")
    p.add_argument("--cache_device", type=str, default=None,
                   help="Where to hold cached MHC embeddings ('cpu' to save VRAM).")
    p.add_argument("--precompute_batch_size", type=int, default=8)

    # Training
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--pos_weight", type=str, default="none",
                   help="'none', 'auto' (n_neg/n_pos), or a float, for BCE class weighting.")

    # Evaluation cadence
    p.add_argument("--eval_every_steps", type=int, default=0,
                   help=">0 also validates mid-epoch every N steps (epochs are long).")
    p.add_argument("--eval_max_batches", type=int, default=50,
                   help="Batches per mid-epoch validation; end-of-epoch always runs in full.")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="stop after this many epochs with no improvement in "
                        "--best_metric (0 = off). With few distinct epitopes, "
                        "the TCR head starts fitting its training split long "
                        "before the epoch budget runs out.")
    p.add_argument("--best_metric", type=str, default="within_allele_auroc",
                   choices=["auc", "auprc", "within_allele_auroc", "loss"],
                   help="Validation metric used to pick best.pt.")

    # LR schedule
    p.add_argument("--lr_scheduler", type=str, default="cosine",
                   choices=["none", "cosine", "onecycle"])
    p.add_argument("--warmup_steps", type=int, default=200)

    # Progressive unfreezing
    p.add_argument("--unfreeze_after_epoch", type=int, default=-1,
                   help="Unfreeze the last N backbone layers after this epoch. -1 = off.")
    p.add_argument("--unfreeze_num_layers", type=int, default=4)

    # Figures
    p.add_argument("--render_figures", action="store_true",
                   help="Execute figures/02_training_progress.ipynb after every "
                        "epoch and once more after the test pass. Plotting code "
                        "stays in the notebook; this only runs it.")
    p.add_argument("--figures_dir", type=str, default=None,
                   help="Where the plotting notebooks live (default: <repo>/figures).")

    # Weights & Biases
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="AETHER")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_log_every", type=int, default=50)

    return p


def parse_args() -> argparse.Namespace:
    """Defaults < --config JSON < explicit CLI flags."""
    parser = build_parser()
    pre, _ = parser.parse_known_args()
    if pre.config:
        with open(pre.config) as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("_comment", None)
        valid = {k: v for k, v in cfg_dict.items() if k in vars(pre)}
        unknown = set(cfg_dict) - set(valid)
        if unknown:
            print(f"WARNING: ignoring unknown config keys: {sorted(unknown)}", flush=True)
        parser.set_defaults(**valid)
    return parser.parse_args()


def resolve_pos_weight(spec: str, train_df: pd.DataFrame,
                       threshold: float = 0.5) -> Optional[float]:
    """
    Resolve --pos_weight.

    "auto" balances the two classes as the loss actually sees them. With
    continuous targets BCEWithLogitsLoss weights the ``y·log σ`` term, so the
    positive mass is ``Σy`` rather than the count above a threshold -- for
    evenly spread grades those differ (0.5N vs 0.4N) and weighting by the
    count biases the boundary. Hard 0/1 labels give the same answer either way.
    """
    if spec in ("none", "", None):
        return None
    if spec == "auto":
        y = train_df["y"].to_numpy(dtype=np.float64)
        graded = bool(((y > 0) & (y < 1)).any())
        if graded:
            pos_mass, neg_mass = float(y.sum()), float((1.0 - y).sum())
        else:
            pos_mass = float((y >= threshold).sum())
            neg_mass = float(len(y) - pos_mass)
        return neg_mass / max(pos_mass, 1.0)
    return float(spec)


# ── Metric logging ────────────────────────────────────────────────────────────

class MetricLog:
    """Appends one JSON row per evaluation and flushes, so notebooks see it live."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Start a fresh file per run. Appending across restarts silently
        # interleaves runs -- outputs/classI_esm2_650m/metrics.jsonl holds 11
        # "val" rows whose epoch counter resets -- and the figure notebooks
        # read it as one curve. The previous file is kept alongside.
        if os.path.exists(path) and os.path.getsize(path) > 0:
            stamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y%m%d-%H%M%S")
            shutil.move(path, f"{os.path.splitext(path)[0]}.{stamp}.jsonl")
        open(path, "w").close()

    def append(self, row: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()


# ── Figure rendering ──────────────────────────────────────────────────────────

class FigureRenderer:
    """
    Executes the plotting notebook out of process.

    All plotting code stays in ``figures/``; this only runs the notebook and
    points it at the current run via ``AETHER_RUN_DIR``. Renders are launched in
    the background so a training epoch is never waiting on matplotlib, and a
    request arriving while a previous render is still going is dropped rather
    than queued — the next epoch will draw the newer data anyway.
    """

    NOTEBOOK = "02_training_progress.ipynb"

    def __init__(self, out_dir: str, figures_dir: str, enabled: bool):
        self.enabled = enabled
        self.out_dir = os.path.abspath(out_dir)
        self.figures_dir = os.path.abspath(figures_dir)
        self._proc: Optional[subprocess.Popen] = None

        if not enabled:
            return
        nb = os.path.join(self.figures_dir, self.NOTEBOOK)
        if not os.path.exists(nb):
            print(f"WARNING: {nb} not found — figure rendering disabled.", flush=True)
            self.enabled = False

    def render(self, wait: bool = False) -> None:
        if not self.enabled:
            return

        if self._proc is not None and self._proc.poll() is None:
            if not wait:
                return  # still drawing the previous one; skip this round
            self._proc.wait()

        env = {
            **os.environ,
            "AETHER_RUN_DIR": self.out_dir,
            "MPLBACKEND": "Agg",
            # The notebook only reads CSVs; keep it off the training GPU.
            "CUDA_VISIBLE_DEVICES": "",
        }
        cmd = [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=1800",
            self.NOTEBOOK,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=self.figures_dir, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except OSError as e:
            print(f"WARNING: could not start figure render ({e}).", flush=True)
            self.enabled = False
            return

        if wait:
            _, err = self._proc.communicate()
            if self._proc.returncode != 0:
                tail = (err or b"").decode(errors="replace").strip().splitlines()[-5:]
                print("WARNING: figure render failed:\n  " + "\n  ".join(tail), flush=True)
            else:
                print(f"Figures updated in {self.figures_dir}/", flush=True)

    def close(self) -> None:
        """Let an in-flight background render finish before the process exits."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.wait()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    rank, world_size, is_distributed = setup_distributed()
    # On one node these coincide; across nodes rank 4 would select cuda:4 on a
    # 4-GPU host, or land on a device other than the one NCCL posts to.
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available() else torch.device("cpu")
    )

    if is_main:
        # mkdir is atomic: concurrent processes cannot reserve the same run.
        os.makedirs(args.out_dir, exist_ok=False)
    if is_distributed:
        dist.barrier()

    # ── Data ──────────────────────────────────────────────────────────────────
    data_cfg = DataConfig(
        csv_url=args.csv_url,
        data_dir=args.data_dir,
        split_group=args.split_group,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        rank=rank,
        world_size=world_size,
    )
    if args.task == "tcr":
        if not args.tcr_csv:
            raise SystemExit("--task tcr requires --tcr_csv")
        tcr_cfg = TCRDataConfig(
            tcr_csv=args.tcr_csv,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            neg_ratio=args.tcr_neg_ratio,
            max_pos_per_epitope=args.tcr_max_pos_per_epitope,
            split_mode=args.tcr_split,
            drop_viral=args.tcr_drop_viral,
            neoantigen_only=args.tcr_neoantigen_only,
            max_train_rows=args.max_train_rows,
            max_eval_rows=args.max_eval_rows,
            rank=rank,
            world_size=world_size,
        )
        train_loader, valid_loader, test_loader, splits = get_tcr_dataloaders(tcr_cfg)
    else:
        train_loader, valid_loader, test_loader, splits = get_dataloaders(data_cfg)

    if is_main:
        audit = audit_splits(splits, args.split_group) if args.task != "tcr" else {"task": "tcr"}
        args.run_manifest_sha256 = write_manifest(args.out_dir, args, splits, audit)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = ModelConfig(
        model_name=args.model_name,
        freeze_backbone=args.freeze_backbone,
        max_length=args.max_length,
        precompute_mhc=args.precompute_mhc,
        cache_device=args.cache_device,
        d_proj=args.d_proj,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_interaction_layers=args.num_interaction_layers,
        use_self_attn=args.use_self_attn,
        use_product_head=args.use_product_head,
        use_tcr_head=(args.task == "tcr"),
        num_tcr_interaction_layers=args.num_tcr_interaction_layers,
    )
    model = NeoantigenModel(model_cfg).to(device)

    # Phase 2 starts from the phase-1 trunk. strict=False because the TCR arm
    # is new and the phase-1 checkpoint has no weights for it.
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        # save_checkpoint writes "model_state"; accept a bare state dict too.
        state = ckpt.get("model_state", ckpt.get("model", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded = len(state) - len(unexpected)
        # Silence here would mean phase 2 trained from scratch on a random
        # trunk and still looked like it was working, so fail loudly instead.
        if loaded == 0:
            raise SystemExit(
                f"--init_from {args.init_from}: no tensor matched the model. "
                f"Checkpoint top-level keys: {sorted(ckpt)[:8]}"
            )
        if is_main:
            new = sorted({m.split('.')[0] for m in missing})
            print(f"\nInitialised from {args.init_from}: {loaded} tensors loaded, "
                  f"{len(missing)} new ({', '.join(new) or 'none'})"
                  + (f", {len(unexpected)} unexpected" if unexpected else ""),
                  flush=True)
    if args.freeze_trunk:
        n = model.freeze_trunk(True)
        if is_main:
            print(f"Froze {n} trunk tensors; training the TCR arm only.", flush=True)

    if is_main:
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nBackbone : {args.model_name} (hidden {model.encoder.hidden_size})")
        print(f"Model    : {total_p:,} total params, {train_p:,} trainable")
        print(f"Head     : {args.num_interaction_layers} interaction layer(s), "
              f"d_proj={args.d_proj}, self_attn={args.use_self_attn}, "
              f"product_head={args.use_product_head}", flush=True)

    # Precompute the MHC side: ~150 distinct sequences carry ~344k rows, and the
    # MHC sequence is ~30x longer than the peptide, so this is where the
    # backbone's time would otherwise go.
    if args.precompute_mhc and args.freeze_backbone:
        mhc_seqs = unique_mhc_sequences(splits)
        if is_main:
            print(f"\nPrecomputing MHC embeddings for {len(mhc_seqs)} unique sequences...",
                  flush=True)
        model.precompute_mhc(mhc_seqs, batch_size=args.precompute_batch_size)
    elif is_main and args.precompute_mhc:
        print("MHC precompute skipped: backbone is not frozen.", flush=True)

    if is_distributed:
        # forward_tcr does not touch the binding head (or the MHC pool), so
        # with a trainable trunk those parameters receive no gradient and DDP
        # raises "Expected to have finished reduction in the prior iteration".
        # With --freeze_trunk they are excluded from the reducer anyway, so the
        # scan is only needed when the trunk is trainable.
        tcr_unused = args.task == "tcr" and not args.freeze_trunk
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=tcr_unused)
        if is_main and tcr_unused:
            print("DDP: find_unused_parameters=True (TCR task with a trainable "
                  "trunk leaves the binding head without gradients).", flush=True)
    # Evaluation runs on rank 0 alone, so it must use the unwrapped module:
    # a forward through the DDP wrapper broadcasts buffers, and that collective
    # has no counterpart on the other ranks (hang, then watchdog SIGABRT).
    eval_model = model.module if is_distributed else model

    # ── Loss / optimizer ──────────────────────────────────────────────────────
    pw = resolve_pos_weight(args.pos_weight, splits["train"],
                            threshold=args.auc_label_threshold)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pw, device=device) if pw is not None else None
    )
    if is_main and pw is not None:
        print(f"BCE pos_weight : {pw:.4f}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(args, optimizer, steps_per_epoch=len(train_loader))

    # bfloat16 shares float32's exponent range, so no gradient scaler is needed.
    def amp_ctx():
        return (
            torch.amp.autocast(device.type, dtype=torch.bfloat16)
            if (args.amp and device.type == "cuda")
            else contextlib.nullcontext()
        )

    # ── Output dir ────────────────────────────────────────────────────────────
    metric_log = None
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump({**vars(args), "model_cfg": model_cfg.to_dict()}, f, indent=2)
        metric_log = MetricLog(os.path.join(args.out_dir, "metrics.jsonl"))

    # Rank 0 only: one renderer per run, not one per process.
    default_figures = os.path.join(os.path.dirname(__file__), "..", "..", "..", "figures")
    renderer = FigureRenderer(
        out_dir=args.out_dir,
        figures_dir=args.figures_dir or default_figures,
        enabled=bool(args.render_figures) and is_main,
    )

    # ── Weights & Biases ──────────────────────────────────────────────────────
    use_wandb = is_main and bool(args.wandb) and wandb is not None and args.wandb_mode != "disabled"
    if is_main and args.wandb and wandb is None:
        print("WARNING: --wandb set but wandb is not installed. Skipping.", flush=True)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"aether_{datetime.now():%Y%m%d-%H%M%S}",
            dir=args.out_dir,
            config={**vars(args), "model_cfg": model_cfg.to_dict()},
            tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
            mode=args.wandb_mode,
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    # best_metric may be unavailable on an epoch -- binary_metrics returns {}
    # when the thresholded labels are single-class -- and the fallback is loss,
    # where lower is better. Tracking which metric the incumbent score came
    # from keeps the comparison on one scale instead of maximizing a loss.
    best_metric_used = args.best_metric
    epochs_since_best = 0
    maximize = args.best_metric in ("auc", "auprc", "within_allele_auroc")
    best_score = -float("inf") if maximize else float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")
    global_step = 0

    def is_better(score: float, higher_is_better: bool) -> bool:
        return score > best_score if higher_is_better else score < best_score

    if is_main:
        print(f"\nTraining: {args.epochs} epochs x {len(train_loader)} steps "
              f"(batch {args.batch_size}, world_size {world_size})\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        # ── Progressive unfreezing ────────────────────────────────────────────
        if args.unfreeze_after_epoch >= 0 and epoch == args.unfreeze_after_epoch + 1:
            if is_distributed:
                # DDP snapshots its parameter set at construction, filtering on
                # requires_grad. Layers unfrozen afterwards are not in the
                # reducer, so their gradients are never all-reduced and each
                # rank silently trains its own copy on its own shard.
                raise SystemExit(
                    "--unfreeze_after_epoch is not supported under torchrun: "
                    "layers unfrozen after DDP is built never sync their "
                    "gradients. Run progressive unfreezing on a single GPU."
                )
            underlying = model.module if isinstance(model, DDP) else model
            encoder = underlying.encoder
            if hasattr(encoder, "unfreeze_last_layers"):
                new_params = encoder.unfreeze_last_layers(args.unfreeze_num_layers)
                if new_params:
                    # A tenth of the head LR: these weights are already good, and
                    # the cached MHC embeddings they produced are now discarded.
                    group = {
                        "params": new_params,
                        "lr": args.lr * 0.1,
                        "weight_decay": args.weight_decay,
                    }
                    if args.lr_scheduler == "onecycle":
                        # OneCycleLR indexes these per group and only sets them
                        # for groups that existed when it was constructed.
                        group.update(max_lr=args.lr * 0.1,
                                     initial_lr=args.lr * 0.1,
                                     min_lr=0.0)
                    optimizer.add_param_group(group)
                if is_main:
                    n = sum(p.numel() for p in new_params)
                    print(f"[epoch {epoch}] Unfroze last {args.unfreeze_num_layers} "
                          f"backbone layers ({n:,} params, lr={args.lr * 0.1:.2e}). "
                          f"MHC cache cleared.", flush=True)

        running_loss, n_seen = 0.0, 0

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            *inputs, y = batch
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx():
                logits = model(*inputs)
            loss = loss_fn(logits.float(), y)

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            bs = y.size(0)
            running_loss += float(loss.item()) * bs
            n_seen += bs

            if is_main and step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running_loss / max(n_seen, 1):.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

            if use_wandb and args.wandb_log_every > 0 and step % args.wandb_log_every == 0:
                wandb.log({
                    "train/loss_step": float(loss.item()),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                }, step=global_step)

            # ── Mid-epoch validation on a capped number of batches ────────────
            if (is_main and args.eval_every_steps > 0
                    and step % args.eval_every_steps == 0):
                vm = evaluate(eval_model, valid_loader, device, loss_fn,
                              args.auc_label_threshold, amp_ctx(),
                              max_batches=args.eval_max_batches)
                metric_log.append({
                    "phase": "val_partial", "epoch": epoch, "step": step,
                    "global_step": global_step,
                    "train_loss": running_loss / max(n_seen, 1),
                    "lr": optimizer.param_groups[0]["lr"],
                    **{f"val_{k}": v for k, v in vm.items()},
                })
                print(f"  [mid-epoch] val_loss={vm['loss']:.4f}"
                      + (f" val_auc={vm['auc']:.4f}" if "auc" in vm else ""), flush=True)
                if use_wandb:
                    wandb.log({f"val_partial/{k}": v for k, v in vm.items()},
                              step=global_step)

        # ── End of epoch: full validation ─────────────────────────────────────
        # Every rank must reach this all_reduce, so it sits outside the is_main
        # block: running_loss covers this rank's shard only, and without the
        # reduction the logged curve is 1/world_size of the data, noisier than
        # it looks and plotted against a full-dataset val_loss.
        train_loss = running_loss / max(n_seen, 1)
        if is_distributed:
            t = torch.tensor([running_loss, float(n_seen)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            train_loss = float(t[0] / torch.clamp(t[1], min=1.0))
            dist.barrier()

        if is_main:
            vm = evaluate(eval_model, valid_loader, device, loss_fn,
                          args.auc_label_threshold, amp_ctx())

            metric_log.append({
                "phase": "val", "epoch": epoch, "step": len(train_loader),
                "global_step": global_step, "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"val_{k}": v for k, v in vm.items()},
            })

            print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} | "
                  f"val_loss {vm['loss']:.4f}"
                  + (f" | val_auc {vm['auc']:.4f}" if "auc" in vm else "")
                  + (f" | val_auprc {vm['auprc']:.4f}" if "auprc" in vm else ""),
                  flush=True)

            if use_wandb:
                wandb.log({"epoch": epoch, "train/loss": float(train_loss),
                           **{f"val/{k}": float(v) for k, v in vm.items()}},
                          step=global_step)

            if args.best_metric in vm:
                metric_now, score, higher = args.best_metric, vm[args.best_metric], maximize
            else:
                # Fall back to loss, which is minimized whatever best_metric says.
                metric_now, score, higher = "loss", vm["loss"], False
                print(f"  WARNING: {args.best_metric!r} unavailable this epoch "
                      f"(single-class labels?); selecting on loss instead.",
                      flush=True)

            if metric_now != best_metric_used:
                # Scales are not comparable; restart the comparison rather than
                # ranking an AUC against a loss.
                best_metric_used = metric_now
                best_score = -float("inf") if higher else float("inf")

            improved = is_better(score, higher)
            if improved:
                best_score = score

            # Written after best_score is updated so the checkpoint records the
            # score it actually achieved, not the previous epoch's.
            args.validation_metrics = vm
            save_checkpoint(last_path, model, optimizer, epoch, best_score, args, model_cfg)
            if improved:
                # Identical weights to the file just written: copying the bytes
                # halves the per-epoch I/O, which for a 650M backbone is GBs.
                shutil.copyfile(last_path, best_path)
                print(f"  -> new best ({metric_now}={score:.4f}), saved best.pt",
                      flush=True)

            # Refresh the training curves in the background; the epoch does not
            # wait on it.
            renderer.render(wait=False)

        # ── Early stopping ────────────────────────────────────────────────────
        # Decided on rank 0, which is the only rank that validates, then
        # broadcast: if rank 0 alone left the loop the others would block in
        # the next epoch's all-reduce forever.
        stop = torch.zeros(1, dtype=torch.int32, device=device)
        if is_main:
            epochs_since_best = 0 if improved else epochs_since_best + 1
            if args.early_stop_patience and epochs_since_best >= args.early_stop_patience:
                stop[0] = 1
                print(f"\nEarly stop: {best_metric_used} has not improved on "
                      f"{best_score:.4f} for {epochs_since_best} epoch(s). "
                      f"best.pt holds epoch {epoch - epochs_since_best}.",
                      flush=True)
        if is_distributed:
            dist.broadcast(stop, src=0)
            dist.barrier()
        if int(stop.item()):
            break

    # ── Test with the best checkpoint ─────────────────────────────────────────
    if is_main:
        underlying = model.module if isinstance(model, DDP) else model
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            underlying.load_state_dict(ckpt["model_state"])
            print(f"\nLoaded best.pt from epoch {ckpt['epoch']} for testing.", flush=True)

        test_metrics, test_probs, test_y = evaluate(
            eval_model, test_loader, device, loss_fn, args.auc_label_threshold,
            amp_ctx(), return_preds=True,
        )

        # Link the evaluated predictions and metrics to the exact checkpoint bytes.
        checkpoint_hash = file_sha256(best_path)
        test_metrics.update(checkpoint_sha256=checkpoint_hash,
                            checkpoint_epoch=ckpt["epoch"],
                            run_manifest_sha256=args.run_manifest_sha256)
        if "mhc_allele" in splits["test"]:
            per_allele_metrics((test_y >= args.auc_label_threshold).astype(int),
                              test_probs, splits["test"].mhc_allele).to_csv(
                os.path.join(args.out_dir, "test_per_allele.csv"), index=False)
        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        metric_log.append({"phase": "test", "epoch": ckpt["epoch"],
                           "global_step": global_step,
                           **{f"test_{k}": v for k, v in test_metrics.items()}})

        # Per-row predictions, with the allele kept so the results notebook can
        # break performance down per HLA allele.
        preds = pd.DataFrame({"y_true": test_y, "y_pred_prob": test_probs})
        test_df = splits["test"]
        if len(test_df) == len(preds):
            preds["peptide"] = test_df["epitope"].to_numpy()
            if "mhc_allele" in test_df.columns:
                preds["mhc_allele"] = test_df["mhc_allele"].to_numpy()
        preds.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

        artifacts = {name: file_sha256(os.path.join(args.out_dir, name)) for name in
                     ["best.pt", "config.json", "run_manifest.json", "metrics.jsonl",
                      "test_metrics.json", "test_predictions.csv"]}
        with open(os.path.join(args.out_dir, "artifact_manifest.json"), "w") as f:
            json.dump(artifacts, f, indent=2)
        print("\nTest metrics:", json.dumps(test_metrics, indent=2), flush=True)
        print(f"Artifacts written to {args.out_dir}/", flush=True)

        # Final render: blocking, so the process does not exit with the figures
        # still mid-draw and the test panels missing.
        if renderer.enabled:
            print("Rendering final figures...", flush=True)
            renderer.render(wait=True)
        else:
            print("Plot them with the notebooks in figures/.", flush=True)

        if use_wandb:
            wandb.log({f"test/{k}": float(v) for k, v in test_metrics.items()
                       if isinstance(v, (int, float))},
                      step=global_step)
            wandb.finish()

    renderer.close()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
