```bash
# If the HF file is private/gated:
export HF_TOKEN=

# Train with frozen ESM3 backbone (recommended first run)
python -m neoantigen.train.train \
  --out_dir /scratch/vchu/aether/mhc_esm3_freeze \
  --csv_url /home/vchu/projects/aip-wanglab/vchu/AETHER/data/model/mhc_set.csv \
  --freeze_backbone \
  --batch_size 8 \
  --epochs 10 \
  --lr 2e-4 \
  --amp

# Train with wandb logging
python -m neoantigen.train.train \
  --out_dir /scratch/vchu/aether/mhc_esm3_freeze \
  --csv_url /home/vchu/projects/aip-wanglab/vchu/AETHER/data/model/mhc_set.csv \
  --freeze_backbone \
  --batch_size 8 \
  --epochs 50 \
  --lr 2e-4 \
  --amp \
  --wandb \
  --wandb_project aether \
  --wandb_run_name mhc_esm3_freeze \
  --wandb_tags "esm3,freeze"

# Tip: start with --freeze_backbone (fast + stable). Unfreeze for fine-tuning.
```
