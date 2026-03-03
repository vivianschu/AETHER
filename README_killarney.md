## Killarney Instructions
0. Run this from anywhere
```bash
module load slurm
AIP_ACCOUNT="aip-wanglab"
cd /project/6101773/vchu/AETHER

srun \
  --account="$AIP_ACCOUNT" \
  -D /project/6101773/vchu/AETHER \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:l40s:1 \
  --mem=20G \
  --cpus-per-task=4 \
  --time=24:00:00 \
  --pty bash
```

1. Check if cuda is available
```bash
module load gcc arrow/21.0.0
module load python/3.12 cuda/12.2
source ~/venv/aether_sys/bin/activate
nvidia-smi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

export WANDB_API_KEY=""
wandb login

export WANDB_CACHE_DIR=/scratch/vchu/wandb_cache
export WANDB_DIR=/scratch/vchu/wandb

```

2. Run inside the interactive shell
```bash
# without wandb
python -m neoantigen.train.train \
  --out_dir /scratch/$USER/aether/mhc_binding_esm3 \
  --csv_url /project/6101773/vchu/AETHER/data/model/mhc_set.csv \
  --freeze_backbone \
  --batch_size 8 \
  --epochs 10 \
  --lr 2e-4 \
  --amp \
  --num_workers 0

# with wandb
python -m neoantigen.train.train \
  --out_dir /scratch/$USER/aether/mhc_binding_esm3 \
  --csv_url /project/6101773/vchu/AETHER/data/model/mhc_set.csv \
  --freeze_backbone \
  --batch_size 8 \
  --epochs 50 \
  --lr 2e-4 \
  --amp \
  --num_workers 0 \
  --wandb \
  --wandb_project aether \
  --wandb_run_name test_mhc_esm3_freeze \
  --wandb_tags "killarney,l40s,esm3,freeze"
```