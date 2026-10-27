# Training

```bash
# From the repo root, with src/ on the path (or after `pip install -e .`)
export PYTHONPATH=src
```

## Main class-I run

```bash
python -m neoantigen.train.train \
  --config configs/classI_esm2_650m.json \
  --out_dir outputs/classI_esm2_650m
```

~15-20 min/epoch on one A6000: the MHC embeddings are precomputed once, so each
step only pushes short peptides through ESM2.

## Quick end-to-end check (~2 min)

```bash
python -m neoantigen.train.train \
  --config configs/smoke_test.json \
  --out_dir outputs/smoke
```

ESM2-35M on a 4k-row subsample. Produces every artifact a real run does, so it
is the fastest way to confirm a change did not break the pipeline. The accuracy
it reports is not meaningful.

## Useful flags

| Flag | Why |
|---|---|
| `--data_dir DIR` | folder with `train.csv` / `val.csv` / `test.csv` (preferred) |
| `--csv_url PATH_OR_URL` | single CSV instead, grouped by peptide by default |
| `--model_name` | any ESM2 checkpoint, e.g. `facebook/esm2_t12_35M_UR50D` for fast iteration |
| `--no_precompute_mhc` | disable the MHC embedding cache (useful to measure what it buys) |
| `--no_freeze_backbone` | fine-tune the whole backbone; disables precompute |
| `--unfreeze_after_epoch N` | train the head first, then unfreeze the last few ESM2 layers |
| `--best_metric within_allele_auroc\|auc\|auprc\|loss` | what selects `best.pt` (default `within_allele_auroc`) |
| `--pos_weight auto` | weight the BCE by `n_neg/n_pos`; only matters for calibration, not ranking |
| `--max_train_rows N` | subsample for a quick run |
| `--eval_every_steps N` | mid-epoch validation — epochs here are ~5,400 steps |
| `--amp` | bfloat16 autocast |

## Multi-GPU

```bash
torchrun --nproc_per_node=4 -m neoantigen.train.train \
  --config configs/classI_esm2_650m.json \
  --out_dir outputs/classI_ddp
```

Validation and checkpointing run on rank 0.

## Output

```
<out_dir>/metrics.jsonl        one JSON row per evaluation, flushed as it goes
<out_dir>/config.json          the full resolved configuration
<out_dir>/best.pt, last.pt     checkpoints
<out_dir>/test_metrics.json    final test metrics
<out_dir>/test_predictions.csv per-row y_true / y_pred_prob / peptide / allele
```

The script draws no figures. Plot a run with `figures/02_training_progress.ipynb`,
which can be re-run mid-training because `metrics.jsonl` is flushed after every
evaluation.

## Audited training protocols (2026-10-27)

The binding configs now use `data/mhcattnnet/classI_peptide_split`. Single-CSV
training also groups by peptide by default. Repeated pairs and contradictory
labels are removed before deriving a split. Rows explicitly marked `label_conflict` (as in the CEDAR builder) are excluded during loading. Pre-split inputs are audited after
sequence cleaning and before any subsampling; leaked groups or repeated pairs
raise an error. `--split_group allele` enforces allele-name and encoded-MHC-sequence disjointness instead. Identical MHC input sequences are kept together even when their allele names differ.
The explicit `pair` option is weaker and should be described as such.

Use a new output directory for **every** run. The trainer reserves it with an
atomic mkdir before loading data and rejects any existing path. This prevents
two jobs from writing the same checkpoints or metric stream. Failed runs also
keep their reserved directory; choose a new name when retrying.

```bash
# Four GPUs, batch 16 per GPU = total batch 64, 10 epochs.
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train_clean.sh \
  configs/clean_peptide.json outputs/my_clean_peptide_run

# Create once, then train a separate model on training alleles only.
python -m neoantigen.data.regroup --in_dir data/mhcattnnet/classI \
  --out_dir data/mhcattnnet/classI_allele_sequence_split --group_by allele
CUDA_VISIBLE_DEVICES=4,5,6,7 scripts/train_clean.sh \
  configs/heldout_allele.json outputs/my_heldout_allele_run
```

Both configs select checkpoints by support-weighted within-allele validation
AUROC, over alleles with both labels. Single-class alleles have undefined AUROC
and are excluded with support counts recorded. Benchmark summaries additionally
require at least 50 scored rows per allele by default. Validation evaluates the
entire split at each epoch; the clean launcher disables mid-epoch sampling.
Held-out-allele training permits shared peptides across alleles: it measures
transfer to unseen alleles, not simultaneous unseen-peptide/allele transfer.

Outputs include resolved `config.json`, `metrics.jsonl`, `best.pt`, `last.pt`,
`test_metrics.json`, `test_predictions.csv`, and `test_per_allele.csv`.
`run_manifest.json` records the Git state, source/data hashes, software versions,
GPU visibility, world size and split audit. `source_snapshot/` preserves the
actual Python source even when the working tree is dirty. Checkpoints embed the
manifest hash and validation metrics. Final test metrics identify the exact
best-checkpoint SHA-256 and epoch; `artifact_manifest.json` hashes the final
predictions and metrics too. Store these together when copying a run elsewhere.
A rerun uses a different process topology from the historical single-GPU runs;
the seed/configuration is recorded, but bitwise identity is not claimed.

AUPRC is average precision (step integration). PPV@100 is precision among the
highest 100 finite scores, or all scored rows if fewer than 100 exist; the actual
count is recorded. Ties at the cutoff receive their expected positive count under
random tie breaking. Per-allele CSVs also report AUPRC and PPV@100. Pooled PPV can
be high for an allele-prior model, so use the per-allele results for triage within
a patient/allele rather than treating pooled precision as peptide discrimination.
