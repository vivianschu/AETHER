# src/neoantigen/train/train.py

import contextlib
import os
import re
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
from .plotting import plot_test_results
from neoantigen.models.config import ModelConfig
from neoantigen.models.transformer import NeoantigenModel

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_auc(y_true_bin: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Returns ROC AUC if both classes are present; otherwise None."""
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
    return_preds: bool = False,
) -> Dict[str, float] | tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()

    total_loss = total_mae = total_mse = 0.0
    n = 0
    all_probs, all_y = [], []

    for peptides, mhcs, y in loader:
        y = y.to(device)
        logits = model(peptides, mhcs)
        loss = loss_fn(logits, y)
        probs = torch.sigmoid(logits)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_mae += float(torch.abs(probs - y).mean().item()) * bs
        total_mse += float(torch.square(probs - y).mean().item()) * bs
        n += bs

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs) if all_probs else np.array([])
    all_y = np.concatenate(all_y) if all_y else np.array([])

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

    if return_preds:
        return metrics, all_probs, all_y
    return metrics


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
    model_cfg: ModelConfig,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    underlying = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model_state": underlying.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "args": vars(args),
            "model_cfg": model_cfg.to_dict(),
        },
        path,
    )


# ── Distributed setup ─────────────────────────────────────────────────────────

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


# ── LR scheduler ─────────────────────────────────────────────────────────────

def build_scheduler(args, optimizer, steps_per_epoch):
    """
    Build an LR scheduler based on ``args.lr_scheduler``.

    Options
    -------
    "none"     – no scheduler (constant LR)
    "cosine"   – cosine annealing with optional linear warmup
    "onecycle" – OneCycleLR (warmup + cosine anneal in one pass)
    """
    sched = args.lr_scheduler
    if sched == "none":
        return None

    total_steps = args.epochs * steps_per_epoch

    if sched == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR
        return OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
        )

    if sched == "cosine":
        warmup = min(args.warmup_steps, total_steps)
        if warmup > 0:
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            warmup_sched = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
            )
            cosine_sched = CosineAnnealingLR(optimizer, T_max=total_steps - warmup)
            return SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup],
            )
        else:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=total_steps)

    raise ValueError(f"Unknown lr_scheduler: {sched!r}. Choose from: none, cosine, onecycle")


# ── Progressive unfreezing ────────────────────────────────────────────────────

def _unfreeze_last_n_layers(
    esm_model: nn.Module,
    n: int,
    optimizer,
    lr: float,
    weight_decay: float,
) -> int:
    """
    Unfreeze the last ``n`` transformer layers of ``esm_model`` and register
    them as a new optimizer parameter group (with lr / 10 to avoid disrupting
    already-converged projection/head weights).

    Returns the number of newly unfrozen parameters.
    """
    # Group frozen parameters by layer index found in their name
    layer_params: dict[int, list[nn.Parameter]] = {}
    for name, param in esm_model.named_parameters():
        if not param.requires_grad:
            m = re.search(r'\.(\d+)\.', name)
            if m:
                idx = int(m.group(1))
                layer_params.setdefault(idx, []).append(param)

    if not layer_params:
        print("WARNING: No numbered transformer layers found in ESM3 — cannot unfreeze.", flush=True)
        return 0

    max_idx = max(layer_params.keys())
    to_unfreeze = []
    for idx in range(max(0, max_idx - n + 1), max_idx + 1):
        for p in layer_params.get(idx, []):
            p.requires_grad = True
            to_unfreeze.append(p)

    if to_unfreeze:
        optimizer.add_param_group({
            "params": to_unfreeze,
            "lr": lr * 0.1,   # smaller LR for pretrained weights
            "weight_decay": weight_decay,
        })

    return len(to_unfreeze)


# ── Argument parsing with JSON config support ─────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train NeoantigenModel for MHC-peptide binding prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config file (loaded before all other args) ────────────────────────────
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config file. Values are used as defaults and can "
             "be overridden by individual CLI flags.",
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--csv_url", type=str,
        default="https://huggingface.co/datasets/wanglab/aether/resolve/main/mhc_set.csv",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--no_stratify", action="store_true")
    parser.add_argument("--auc_label_threshold", type=float, default=0.5)

    # ── Model architecture ────────────────────────────────────────────────────
    parser.add_argument("--model_name", type=str, default="esm3_sm_open_v1")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--d_proj", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--num_interaction_layers", type=int, default=1,
        help="Number of stacked bidirectional cross-attention rounds.",
    )
    parser.add_argument(
        "--use_self_attn", action="store_true",
        help="Add self-attention within each modality after cross-attention.",
    )
    parser.add_argument(
        "--use_product_head", action="store_true",
        help="Concat [pep, mhc, pep*mhc] instead of [pep, mhc] for the MLP head.",
    )

    # ── Embedding cache ───────────────────────────────────────────────────────
    parser.add_argument(
        "--use_embedding_cache", action="store_true",
        help="Cache ESM3 embeddings in memory (and optionally on disk) to avoid "
             "re-encoding sequences seen in previous batches/epochs.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=".cache/esm3",
        help="Directory for on-disk ESM3 embedding cache (used with --use_embedding_cache).",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # ── LR scheduler ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--lr_scheduler", type=str, default="none",
        choices=["none", "cosine", "onecycle"],
        help="LR scheduling strategy.",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100,
        help="Linear warmup steps (used with --lr_scheduler cosine).",
    )

    # ── Progressive unfreezing ────────────────────────────────────────────────
    parser.add_argument(
        "--unfreeze_after_epoch", type=int, default=-1,
        help="Unfreeze the last N ESM3 layers after this epoch. -1 = disabled.",
    )
    parser.add_argument(
        "--unfreeze_num_layers", type=int, default=4,
        help="Number of ESM3 transformer layers to unfreeze.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--plot_test", action="store_true", default=True,
        help="Generate scatter / distribution / ROC plots for the test set.",
    )

    # ── Weights & Biases ──────────────────────────────────────────────────────
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AETHER")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_log_every", type=int, default=50)

    return parser


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args with optional JSON config file support.

    Loading order (later entries win):
      1. Parser defaults
      2. Values from --config JSON file
      3. Explicit CLI flags
    """
    parser = build_parser()

    # First pass: get --config path only
    pre, remaining = parser.parse_known_args()

    if pre.config:
        with open(pre.config) as f:
            cfg_dict = json.load(f)
        # Inject JSON values as argparse defaults so they are overrideable by CLI
        parser.set_defaults(**cfg_dict)

    # Second pass: full parse (CLI flags override JSON defaults)
    args = parser.parse_args()
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Distributed setup (must come before device assignment) ────────────────
    rank, world_size, is_distributed = setup_distributed()
    is_main = (rank == 0)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    # ── Data ──────────────────────────────────────────────────────────────────
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
    model_cfg = ModelConfig(
        model_name=args.model_name,
        freeze_backbone=args.freeze_backbone,
        use_embedding_cache=args.use_embedding_cache,
        cache_dir=args.cache_dir,
        d_proj=args.d_proj,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_interaction_layers=args.num_interaction_layers,
        use_self_attn=args.use_self_attn,
        use_product_head=args.use_product_head,
    )
    model = NeoantigenModel(model_cfg).to(device)

    if is_main:
        total_p = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {total_p:,} total params, {trainable_p:,} trainable.", flush=True)
        if args.num_interaction_layers > 1:
            print(f"  stacked interaction layers : {args.num_interaction_layers}", flush=True)
        if args.use_self_attn:
            print("  self-attention after cross-attn : ON", flush=True)
        if args.use_product_head:
            print("  product head [pep, mhc, pep*mhc] : ON", flush=True)
        if args.use_embedding_cache:
            print(f"  embedding cache : ON  (cache_dir={args.cache_dir})", flush=True)

    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ── Loss / optimizer ──────────────────────────────────────────────────────
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = build_scheduler(args, optimizer, steps_per_epoch=len(train_loader))
    if is_main and scheduler is not None:
        print(f"LR scheduler : {args.lr_scheduler}  "
              f"(warmup_steps={args.warmup_steps})", flush=True)

    # bfloat16 has the same exponent range as float32 so gradient scaling is
    # unnecessary (and unsupported for bfloat16 params like ESM3's weights).
    amp_ctx: contextlib.AbstractContextManager = (
        torch.amp.autocast(device.type, dtype=torch.bfloat16)
        if (args.amp and device.type == "cuda")
        else contextlib.nullcontext()
    )

    # ── Output dir ────────────────────────────────────────────────────────────
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            combined = {**vars(args), "model_cfg": model_cfg.to_dict()}
            json.dump(combined, f, indent=2)

    # ── Weights & Biases ──────────────────────────────────────────────────────
    use_wandb = (
        is_main
        and bool(args.wandb)
        and (wandb is not None)
        and args.wandb_mode != "disabled"
    )
    if is_main and args.wandb and wandb is None:
        print("WARNING: --wandb set but wandb is not installed. Skipping.", flush=True)

    if use_wandb:
        run_name = args.wandb_run_name or f"mhc_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            dir=args.out_dir,
            config={**vars(args), "model_cfg": model_cfg.to_dict()},
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
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        # ── Progressive unfreezing ─────────────────────────────────────────────
        if (
            args.unfreeze_after_epoch >= 0
            and epoch == args.unfreeze_after_epoch + 1
            and is_main
        ):
            underlying = model.module if isinstance(model, DDP) else model
            encoder = underlying.encoder
            encoder.invalidate_cache()  # embeddings will change after unfreezing
            n_params = _unfreeze_last_n_layers(
                encoder.model,
                args.unfreeze_num_layers,
                optimizer,
                args.lr,
                args.weight_decay,
            )
            print(
                f"[epoch {epoch}] Progressive unfreezing: {n_params} backbone params "
                f"added to optimizer (lr={args.lr * 0.1:.2e}).",
                flush=True,
            )

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
            if scheduler is not None:
                scheduler.step()

            bs = y.size(0)
            running_loss += float(loss.item()) * bs
            n += bs

            if is_main and step % 100 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch {epoch} step {step} loss={loss.item():.4f} lr={cur_lr:.2e}",
                    flush=True,
                )

            if use_wandb and args.wandb_log_every > 0 and step % args.wandb_log_every == 0:
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

        # ── End-of-epoch: validate and checkpoint (rank 0 only) ───────────────
        if is_distributed:
            dist.barrier()

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
                f"val_loss {val_metrics['loss']:.4f} | val_mae {val_metrics['mae']:.4f} | "
                f"val_mse {val_metrics['mse']:.4f}"
                + (f" | val_auc {val_metrics['auc']:.4f}" if "auc" in val_metrics else " | val_auc NA"),
                flush=True,
            )

            save_checkpoint(last_path, model, optimizer, epoch, best_val, args, model_cfg)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(best_path, model, optimizer, epoch, best_val, args, model_cfg)

                if use_wandb:
                    artifact = wandb.Artifact(name="best-checkpoint", type="model")
                    artifact.add_file(best_path)
                    wandb.log_artifact(artifact)

        if is_distributed:
            dist.barrier()

    # ── Test with best checkpoint (rank 0 only) ───────────────────────────────
    if is_distributed:
        dist.barrier()

    if is_main:
        underlying = model.module if isinstance(model, DDP) else model
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        underlying.load_state_dict(ckpt["model_state"])

        test_metrics, test_probs, test_y = evaluate(
            model, test_loader, device, loss_fn, args.auc_label_threshold,
            return_preds=True,
        )

        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        print("Test metrics:", test_metrics, flush=True)

        if args.plot_test:
            plot_test_results(test_y, test_probs, args.out_dir, args.auc_label_threshold)

        if use_wandb:
            wandb.log({f"test/{k}": float(v) for k, v in test_metrics.items()}, step=global_step)
            if args.plot_test:
                plots_dir = os.path.join(args.out_dir, "plots")
                for fname in ("scatter.png", "score_distribution.png", "roc_curve.png", "residuals.png"):
                    fpath = os.path.join(plots_dir, fname)
                    if os.path.exists(fpath):
                        wandb.log({f"test/{fname.split('.')[0]}": wandb.Image(fpath)}, step=global_step)
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
