```bash
python predict.py \
  --ckpt /Users/vivianschu/repos/AETHER/outputs/mhc_binding_esm_shared/best.pt \
  --input_csv /Users/vivianschu/repos/AETHER/data/model/splits/mhc-80-10-10/test.csv \
  --output_csv /Users/vivianschu/repos/AETHER/outputs/mhc_binding_esm_shared/test_preds.csv \
  --amp
```