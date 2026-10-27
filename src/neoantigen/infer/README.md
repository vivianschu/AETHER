# Inference

```bash
export PYTHONPATH=src

python -m neoantigen.infer.predict \
  --ckpt outputs/classI_esm2_650m/best.pt \
  --input_csv data/mhcattnnet/classI/test.csv \
  --output_csv outputs/classI_esm2_650m/preds.csv \
  --amp
```

## Input

The peptide and MHC columns are resolved by alias, so both schemas load
unchanged:

| meaning | accepted column names |
|---|---|
| peptide | `epitope`, `peptide`, `pep`, `peptide_seq` |
| MHC sequence | `mhc_sequence`, `mhc_amino_acid`, `mhc_seq`, `mhc` |

A label column (`mhc_binding`) is optional and is carried through to the output
if present.

## Output

The input columns plus `pred_logit` and `pred_prob`.

## Notes

- The architecture is rebuilt from the `model_cfg` stored in the checkpoint, so
  no model flags need to be repeated here.
- The unique MHC sequences in the input are encoded once up front, the same way
  training does. `--no_precompute_mhc` turns that off.

### Attention exports

```bash
python -m neoantigen.infer.explain --ckpt outputs/my_clean_peptide_run/best.pt \
  --input_csv data/mhcattnnet/classI_peptide_split/test.csv \
  --out_dir outputs/my_attention_export --max_rows 10 --device cpu --plot
```

This exports interaction attention averaged across heads and learned pooling
weights to one compressed NPZ per input row, with JSON token axes, input identity,
prediction, and checkpoint hash. Token axes include ESM's special tokens. It
observes existing weights through hooks and leaves model predictions unchanged.
Attention describes the model's internal weighting; it is not a causal
explanation or evidence of immunogenicity. The input needs peptide and MHC
sequences, using the same aliases as ordinary prediction.

An unseen-allele evaluation must use a model trained with those alleles excluded.
Changing the test CSV for an existing peptide-split checkpoint does not establish
allele generalization. Use the separately trained `heldout_allele.json` protocol.

With `--split_group allele`, standalone evaluation verifies the evaluation alleles and encoded MHC sequences against the checkpoint’s own training rows. It requires the matching `run_manifest.json` beside the checkpoint and verifies that the source training data still match their saved hash. A peptide-trained checkpoint cannot be relabelled as unseen-allele merely by supplying a different split directory.

`--plot` also renders peptide pooling and the final peptide-to-MHC attention for the 20 MHC token positions with highest mean attention. JSON/NPZ exports retain every token and layer.
