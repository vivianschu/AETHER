```bash
# If the HF file is private/gated:
export HF_TOKEN=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

python train.py \
  --out_dir /Users/vivianschu/repos/AETHER/outputs/mhc_binding_esm_shared_hf \
  --freeze_backbone \
  --batch_size 8 \
  --epochs 10 \
  --lr 2e-4 \
  --amp
# Tip: first run with --freeze_backbone (fast + stable). Then unfreeze for finetuning.
```