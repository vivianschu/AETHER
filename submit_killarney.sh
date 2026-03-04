#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM submission script for AETHER MHC-binding training on Killarney (L40S)
#
# Launch (2 GPUs):  sbatch scripts/submit_killarney.sh
# Launch (4 GPUs):  sbatch --gres=gpu:l40s:4 --cpus-per-task=64 --mem=128G \
#                          scripts/submit_killarney.sh
#
# LR linear-scaling rule: with N GPUs the effective batch size is N×batch_size.
# Scale --lr proportionally, e.g. 2 GPUs → 4e-4, 4 GPUs → 8e-4.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=aether_mhc
#SBATCH --account=aip-wanglab
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         # 16 cores per L40S (standard node)
#SBATCH --mem=64G                  # 32 GB per GPU headroom
#SBATCH --gres=gpu:l40s:2
#SBATCH --output=/scratch/%u/aether/logs/%x_%j.out
#SBATCH --error=/scratch/%u/aether/logs/%x_%j.err
#SBATCH -D /project/6101773/vchu/AETHER

# ── Environment ───────────────────────────────────────────────────────────────
source ~/venv/aether_sys/bin/activate

NUM_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
# Fallback: count from gres string if SLURM_JOB_GPUS is unset
if [ -z "$SLURM_JOB_GPUS" ]; then
    NUM_GPUS=2
fi

# ── Output / log dirs ─────────────────────────────────────────────────────────
OUT_DIR="/scratch/$USER/aether/mhc_binding_esm3_nofreeze"
mkdir -p "$OUT_DIR"
mkdir -p "/scratch/$USER/aether/logs"

echo "Job $SLURM_JOB_ID | GPUs: $NUM_GPUS | Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Launch DDP via torchrun ───────────────────────────────────────────────────
# torchrun sets LOCAL_RANK / RANK / WORLD_SIZE automatically.
# --lr is scaled linearly with the number of GPUs (linear scaling rule).
LR=$(python -c "print(f'{2e-4 * $NUM_GPUS:.1e}')")

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m neoantigen.train.train \
    --out_dir        "$OUT_DIR" \
    --csv_url        /project/6101773/vchu/AETHER/data/model/mhc_set.csv \
    --batch_size     8 \
    --epochs         50 \
    --lr             "$LR" \
    --amp \
    --num_workers    8 \
    --wandb \
    --wandb_project  aether \
    --wandb_run_name "mhc_esm3_${NUM_GPUS}gpu" \
    --wandb_tags     "killarney,l40s,esm3,ddp,nofreeze"
