```bash
python train.py \
  --split_dir /Users/vivianschu/repos/AETHER/data/model/splits/mhc-80-10-10 \
  --out_dir /Users/vivianschu/repos/AETHER/outputs/mhc_binding_esm_shared \
  --freeze_backbone \
  --batch_size 8 \
  --epochs 10 \
  --lr 2e-4 \
  --amp
# Tip: first run with --freeze_backbone (fast + stable). Then unfreeze for finetuning.
```