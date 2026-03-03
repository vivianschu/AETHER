```bash
python -m neoantigen.infer.predict \
  --ckpt /scratch/vchu/aether/mhc_esm3_freeze/best.pt \
  --input_csv /home/vchu/projects/aip-wanglab/vchu/AETHER/data/model/mhc_set.csv \
  --output_csv /scratch/vchu/aether/mhc_esm3_freeze/preds.csv \
  --amp
```

Input CSV must have columns: `epitope`, `mhc_sequence`.
Optional column `mhc_binding` (numeric 0–1) is preserved in output if present.

Output CSV adds `pred_logit` and `pred_prob` columns.
