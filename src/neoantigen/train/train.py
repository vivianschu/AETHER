# src/neoantigen/train/train.py

import contextlib
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .dataloader import DataConfig, get_dataloaders
from neoantigen.models.transformer import NeoantigenModel

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def compute_auc(y_true_bin: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    Returns ROC AUC if both classes are present; otherwise None.
    """
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
    loader,
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

    metrics: Dict[str, float] = {
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


def save_checkpoint(path: str, model: nn.Module, optimizer, epoch: int, best_val: float, args: argparse.Namespace):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Unwrap DDP to get the underlying state dict
    underlying = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model_state": underlying.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "args": vars(args),
        },
        path,
    )


def setup_distributed():
    """
    Initialize DDP from torchrun environment variables.
    Returns (rank, world_size, is_distributed).
    Safe to call in single-GPU runs — returns (0, 1, False) when not launched
    via torchrun.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_url",
        type=str,
        default="https://huggingface.co/datasets/wanglab/aether/resolve/main/mhc_set.csv",
    )
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="esm3_sm_open_v1")
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--auc_label_threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--no_stratify", action="store_true")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="AETHER")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated tags")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_log_every", type=int, default=50, help="Log training step metrics every N steps")

    args = parser.parse_args()

    # ── Distributed setup (must come before device assignment) ──────────────
    rank, world_size, is_distributed = setup_distributed()
    is_main = (rank == 0)

    # Different seed per rank so DistributedSampler shuffles differently each epoch
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_cfg = DataConfig(
        csv_url=args.csv_url,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stratify=(not args.no_stratify),
        stratify_threshold=args.auc_label_threshold,
        rank=rank,
        world_size=world_size,
    )
    train_loader, valid_loader, test_loader, full_df = get_dataloaders(data_cfg)
    if is_main:
        print(f"Loaded {len(full_df)} usable rows after dropping NaNs.", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NeoantigenModel(model_name=args.model_name, freeze_backbone=args.freeze_backbone).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ── Loss / optimizer ──────────────────────────────────────────────────────
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # bfloat16 has the same exponent range as float32 so gradient scaling is
    # unnecessary (and unsupported for bfloat16 params like ESM3's weights).
    amp_ctx: contextlib.AbstractContextManager = (
        torch.amp.autocast(device.type, dtype=torch.bfloat16)
        if (args.amp and device.type == "cuda")
        else contextlib.nullcontext()
    )

    # ── Output dir (rank 0 only) ──────────────────────────────────────────────
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # ── wandb (rank 0 only) ───────────────────────────────────────────────────
    use_wandb = is_main and bool(args.wandb) and (wandb is not None) and args.wandb_mode != "disabled"
    if is_main and args.wandb and wandb is None:
        print("WARNING: --wandb set but wandb is not installed. Proceeding without wandb.", flush=True)

    if use_wandb:
        run_name = args.wandb_run_name or f"mhc_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            dir=args.out_dir,
            config=vars(args),
            tags=tags,
            mode=args.wandb_mode,
        )

        wandb.config.update(
            {
                "device": str(device),
                "cuda": torch.cuda.is_available(),
                "torch_version": torch.__version__,
                "world_size": world_size,
                "effective_batch_size": args.batch_size * world_size,
                "num_trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "num_params_total": int(sum(p.numel() for p in model.parameters())),
                "dataset_rows_total": int(len(full_df)),
                "train_steps_per_epoch": int(len(train_loader)),
                "val_steps_per_epoch": int(len(valid_loader)),
            },
            allow_val_change=True,
        )

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Advance DistributedSampler so each epoch gets a different shuffle
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0
        n = 0

        if is_main:
            print(f"Starting epoch {epoch}...", flush=True)

        for step, (peptides, mhcs, y) in enumerate(train_loader, start=1):
            global_step += 1

            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                logits = model(peptides, mhcs)
                loss = loss_fn(logits, y)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = y.size(0)
            running_loss += float(loss.item()) * bs
            n += bs

            if is_main and step % 100 == 0:
                print(f"epoch {epoch} step {step} loss={loss.item():.4f}", flush=True)

            if use_wandb and (args.wandb_log_every > 0) and (step % args.wandb_log_every == 0):
                wandb.log(
                    {
                        "train/loss_step": float(loss.item()),
                        "train/epoch": epoch,
                        "train/step": step,
                        "train/global_step": global_step,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

        # ── End-of-epoch: validate and checkpoint on rank 0 ──────────────────
        if is_distributed:
            dist.barrier()  # wait for all ranks to finish the epoch

        if is_main:
            train_loss = running_loss / max(n, 1)
            val_metrics = evaluate(model, valid_loader, device, loss_fn, args.auc_label_threshold)

            row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")

            if use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": float(train_loss),
                    "val/loss": float(val_metrics["loss"]),
                    "val/mae": float(val_metrics["mae"]),
                    "val/mse": float(val_metrics["mse"]),
                    "train/global_step": global_step,
                }
                if "auc" in val_metrics:
                    log_dict["val/auc"] = float(val_metrics["auc"])
                wandb.log(log_dict, step=global_step)

            print(
                f"epoch {epoch:02d} | train_loss {train_loss:.4f} | "
                f"val_loss {val_metrics['loss']:.4f} | val_mae {val_metrics['mae']:.4f} | val_mse {val_metrics['mse']:.4f}"
                + (f" | val_auc {val_metrics['auc']:.4f}" if "auc" in val_metrics else " | val_auc NA"),
                flush=True,
            )

            save_checkpoint(last_path, model, optimizer, epoch, best_val, args)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(best_path, model, optimizer, epoch, best_val, args)

                if use_wandb:
                    artifact = wandb.Artifact(name="best-checkpoint", type="model")
                    artifact.add_file(best_path)
                    wandb.log_artifact(artifact)

        if is_distributed:
            dist.barrier()  # ensure checkpoint is written before next epoch starts

    # ── Test with best checkpoint (rank 0 only) ───────────────────────────────
    if is_distributed:
        dist.barrier()

    if is_main:
        underlying = model.module if isinstance(model, DDP) else model
        ckpt = torch.load(best_path, map_location=device)
        underlying.load_state_dict(ckpt["model_state"])
        test_metrics = evaluate(model, test_loader, device, loss_fn, args.auc_label_threshold)

        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

        print("Test metrics:", test_metrics, flush=True)

        if use_wandb:
            wandb.log({f"test/{k}": float(v) for k, v in test_metrics.items()}, step=global_step)
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
