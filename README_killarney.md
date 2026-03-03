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
  --time=4:00:00 \
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

export WANDB_API_KEY="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"
wandb login

```

2. Run inside the interactive shell
```bash
export TOKENIZERS_PARALLELISM=false

python -m neoantigen.train.train \
  --out_dir /scratch/$USER/aether/mhc_binding_esm_shared_hf \
  --csv_url /project/6101773/vchu/AETHER/data/model/mhc_set.csv \
  --freeze_backbone \
  --batch_size 8 \
  --epochs 10 \
  --lr 2e-4 \
  --amp \
  --num_workers 0
```


3. Batch job `train.slurm`
```bash
cat > train.slrm <<'SLRM'
#!/bin/bash
#SBATCH --job-name=aether_mhc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module load gcc arrow
source ~/venv/aether_sys/bin/activate

# Confirm resources
echo "Job running on: $(hostname)"
nvidia-smi || true
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

python -m neoantigen.train.train \
  --out_dir /home/vchu/scratch/aether/mhc_binding_esm_shared_hf \
  --csv_url /home/vchu/projects/6101773/vchu/AETHER/data/model/mhc_set.csv \
  --freeze_backbone \
  --batch_size 8 \
  --epochs 10 \
  --lr 2e-4 \
  --amp
SLRM
```

4. Submit batch job
```bash
AIP_ACCOUNT="$(ls -d ~/projects/*/ | head -n 1 | sed 's|/$||' | sed 's|./||')"
WORKDIR="$HOME/projects/aip-wanglab/vchu/AETHER"

sbatch \
  --account "$AIP_ACCOUNT" \
  -D "$WORKDIR" \
  train.slrm
```