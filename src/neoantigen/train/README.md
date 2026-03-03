```bash
# If the HF file is private/gated:
export HF_TOKEN=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

python -m neoantigen.train.train \
  --out_dir /scratch/vchu/aether/mhc_binding_esm_shared_hf \
  --csv_url /home/vchu/projects/aip-wanglab/vchu/AETHER/data/model/mhc_set.csv \
  --freeze_backbone \
  --batch_size 8 \
  --epochs 10 \
  --lr 2e-4 \
  --amp
# Tip: first run with --freeze_backbone (fast + stable). Then unfreeze for finetuning.
```