# AETHER

Current audited results: [2026-10-27 benchmark report](results/20261027/README.md), with clean checkpoints, both split protocols, NetMHCpan 4.2d, BigMHC, and per-allele metrics.


**A**ffinity **E**valuation for **T**-cell **H**LA **E**pitope Recognition.

AETHER predicts whether a peptide binds an MHC allele. An **ESM2** backbone
encodes the peptide and the MHC allele sequence, bidirectional cross-attention
lets the two representations read each other, and an MLP head emits a binding
logit. Label is binary (1 = binder, 0 = non-binder); the headline metric is
AUROC, with AUPRC and a per-allele breakdown alongside.

---

## Contents

- [Install](#install)
- [Tutorial: train a model](#tutorial-train-a-model)
- [Tutorial: make predictions](#tutorial-make-predictions)
- [Tracking a run](#tracking-a-run)
- [Evaluating on a held-out split](#evaluating-on-a-held-out-split)
- [Benchmarks](#benchmarks)
- [Data](#data)
- [Architecture](#architecture)
- [Known issues and caveats](#known-issues-and-caveats)
- [Repository layout](#repository-layout)

---

## Install

The environment is managed with [pixi](https://pixi.sh). Install pixi once:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
exec $SHELL            # pick up ~/.pixi/bin on PATH
```

Then, from the repo root:

```bash
pixi install
```

That reads [`pixi.toml`](pixi.toml), resolves everything into `.pixi/` in the
repo, and installs `neoantigen` in editable mode — so no `PYTHONPATH` fiddling
and no conda activate. It pulls a CUDA build of PyTorch (`[system-requirements]
cuda = "12.0"`), which is a few GB on first run.

Run anything through pixi:

```bash
pixi run python -c "import torch; print(torch.cuda.is_available())"
pixi shell             # or drop into a subshell with the env active
```

Predefined tasks (see `[tasks]` in `pixi.toml`):

| Task | What it does |
|---|---|
| `pixi run smoke` | 2-minute end-to-end check on a subsample |
| `pixi run train` | the main class-I run, rendering figures as it goes |
| `pixi run regroup` | rebuild the splits without train/test leakage |
| `pixi run evaluate` | score a checkpoint on a held-out split |
| `pixi run benchmark` | run the reference models and compare |
| `pixi run figures` | regenerate the training-results figures |
| `pixi run lab` | Jupyter Lab in `figures/` |

<details>
<summary>Without pixi</summary>

```bash
pip install -r requirements.txt
export PYTHONPATH=src        # or: pip install -e .
```

All commands below work the same, minus the `pixi run` prefix.
</details>

---

## Tutorial: train a model

### 1. Check the pipeline works (~2 minutes)

```bash
pixi run smoke
```

ESM2-35M on a 4,000-row subsample for 2 epochs. It produces every artifact a
real run does, so it is the fastest way to confirm nothing is broken. **The
accuracy it reports is not meaningful** — it exists to exercise the code path.

You should see it finish with something like:

```
epoch 02 | train_loss 0.4873 | val_loss 0.4590 | val_auc 0.7557 | val_auprc 0.9099
  -> new best (auc=0.7557), saved best.pt
Loaded best.pt from epoch 2 for testing.
Test metrics: { "auc": 0.728, ... }
```

### 2. Build a leakage-free split (do this once)

The splits that ship with the dataset leak — see
[Known issues](#known-issues-and-caveats). Build honest ones:

```bash
pixi run regroup
```

This pools the original splits, drops duplicates and contradictory labels, and
re-partitions so a peptide never appears in two splits. It prints a leakage
check that must read `clean`:

```
  dropped 135,127 exact duplicate rows
  dropped 8,248 rows across 4,124 (peptide, allele) pairs with contradictory labels
  leakage check
    test (peptide, allele) pairs also in train : 0
    shared peptide between train and test      : 0
    clean
```

Output lands in `data/mhcattnnet/classI_peptide_split/`.

### 3. Train

```bash
pixi run python -m neoantigen.train.train \
  --config configs/classI_esm2_650m.json \
  --data_dir data/mhcattnnet/classI_peptide_split \
  --out_dir outputs/classI_peptide_split \
  --render_figures
```

About 15–20 minutes per epoch on one A6000. The MHC embeddings are precomputed
once up front, so each step only pushes short peptides through ESM2 — that is
what makes batch-64 practical.

`--render_figures` is the integration you want: after **every epoch** it
re-executes `figures/02_training_progress.ipynb` in the background (training
never waits on it), and once more — blocking — after the test pass, so the
process does not exit with the figures half-drawn.

Useful flags:

| Flag | Why |
|---|---|
| `--model_name facebook/esm2_t12_35M_UR50D` | small backbone for fast iteration |
| `--max_train_rows 20000` | subsample for a quick run |
| `--eval_every_steps 1000` | mid-epoch validation (epochs are ~5,400 steps) |
| `--best_metric within_allele_auroc\|auc\|auprc\|loss` | what selects `best.pt` (default `within_allele_auroc`) |
| `--unfreeze_after_epoch 3` | train the head first, then unfreeze the last ESM2 layers |
| `--no_precompute_mhc` | disable the MHC cache, e.g. to measure what it buys |
| `--pos_weight auto` | weight BCE by `n_neg/n_pos`; affects calibration, not ranking |
| `--amp` | bfloat16 autocast |

Multi-GPU:

```bash
pixi run torchrun --nproc_per_node=4 -m neoantigen.train.train \
  --config configs/classI_esm2_650m.json --out_dir outputs/classI_ddp
```

Validation, checkpointing and figure rendering happen on rank 0.

### 4. What a run writes

```
outputs/<run>/metrics.jsonl        one JSON row per evaluation, flushed live
outputs/<run>/config.json          the full resolved configuration
outputs/<run>/best.pt, last.pt     checkpoints
outputs/<run>/test_metrics.json    test metrics, written as soon as training ends
outputs/<run>/test_predictions.csv per-row y_true / y_pred_prob / peptide / allele
```

The held-out test pass runs automatically with `best.pt` the moment the last
epoch finishes — no separate command needed.

---

## Tutorial: make predictions

```bash
pixi run python -m neoantigen.infer.predict \
  --ckpt outputs/classI_peptide_split/best.pt \
  --input_csv my_peptides.csv \
  --output_csv my_predictions.csv \
  --amp
```

### Input

A CSV with a peptide column and an MHC sequence column. Names are resolved by
alias, so both schemas work unchanged:

| meaning | accepted column names |
|---|---|
| peptide | `epitope`, `peptide`, `pep`, `peptide_seq` |
| MHC sequence | `mhc_sequence`, `mhc_amino_acid`, `mhc_seq`, `mhc` |

```csv
peptide,mhc_amino_acid,mhc_allele
KRVTEILGF,SHSMRYFHTSVSRPGRGEPRFITVGYVDDTLFVRF...,HLA-B*27:02
AANPHSFVF,SHSMRYFDTAVSRPGRGEPRFISVGYVDDTQFVRF...,HLA-C*07:02
```

You need the **allele protein sequence**, not just the allele name. Pull them
from `data/HLA/` (IPD-IMGT/HLA) or copy the `mhc_amino_acid` column for the
allele out of `data/mhcattnnet/classI/train.csv`.

### Output

Your input columns plus `pred_logit` and `pred_prob`. `pred_prob` is the
predicted probability of binding, in `[0, 1]`:

```csv
peptide,mhc_amino_acid,mhc_allele,pred_logit,pred_prob
ATFSVPMEK,SHSMRYFYTSVSRPG...,HLA-A*11:01,0.2246,0.5559
```

The architecture is rebuilt from the `model_cfg` stored in the checkpoint, so
no model flags need repeating. The unique MHC sequences in your file are
encoded once up front, as in training; `--no_precompute_mhc` turns that off.

> **Rank, don't threshold.** `pred_prob` is not calibrated against any
> biological rate — the training set is 77% binders, which no real screen is.
> Use it to rank candidates for a given allele and take the top *k*. Check
> `figures/test_calibration.png` before reading a probability literally.

---

## Tracking a run

All plotting lives in notebooks under [`figures/`](figures/); `train.py` draws
nothing and only writes the metrics they read. That keeps figures restylable
without retraining.

| Notebook | What it plots |
|---|---|
| `01_dataset_overview.ipynb` | class balance, peptide length, MHC redundancy, allele coverage |
| `02_training_progress.ipynb` | loss and AUROC/AUPRC curves, ROC/PR, calibration, per-allele AUROC |

**While training**, with `--render_figures`, the figures refresh themselves
after every epoch. Without it, re-run the notebook whenever you like —
`metrics.jsonl` is flushed after each evaluation, so you always get the curves
so far:

```bash
pixi run figures                                  # the default run
AETHER_RUN_DIR=outputs/my_run pixi run figures    # a specific run
```

Before the first epoch finishes there are no full validations yet, so the
notebook falls back to the mid-epoch checks and says so in the titles. Those use
a capped batch count and are noisier. Test-set panels skip themselves until the
run reaches the test stage.

---

## Evaluating on a held-out split

Training already runs the test pass. Use `evaluate.py` to score a checkpoint on
a *different* split — most importantly the unseen-allele one:

```bash
pixi run python -m neoantigen.infer.evaluate \
  --ckpt outputs/classI_peptide_split/best.pt \
  --data_dir data/mhcattnnet/classI/new_mhc_split \
  --split test \
  --out_dir outputs/classI_peptide_split \
  --prefix unseen_allele --amp
```

It writes `<prefix>_metrics.json` and `<prefix>_predictions.csv` in the same
format as a training run.

Expect a drop here. In a smoke run the same checkpoint scored **0.738 on its own
test split and 0.677 on unseen alleles** — the gap is the honest measure of how
far the model generalizes beyond the alleles it was trained on, and it is the
number worth quoting for a pan-allele claim.

---

## Benchmarks

```bash
pixi run python -m neoantigen.benchmark.run \
  --data_dir data/mhcattnnet/classI_peptide_split --split test \
  --out_dir outputs/benchmarks \
  --include aether=outputs/classI_peptide_split/test_predictions.csv

pixi run benchmark-figures     # -> figures/benchmark_*.png
```

Every method writes `<out_dir>/<name>_predictions.csv` in the same schema as a
training run, so an AETHER run folds in with `--include` and one notebook plots
them all.

### What each baseline is for

| Method | Question it answers |
|---|---|
| `majority` | sanity check — AUROC is 0.5 by construction |
| `allele_prior`, `length_prior`, `allele_length_prior` | **does the model read the peptide at all?** These predict from label frequency and never see the sequence |
| `pssm` | does it beat the classical per-allele scoring matrix? |
| `blosum_lr` | does it beat logistic regression on BLOSUM-encoded peptides? |
| `esm2_probe` | **does the architecture earn its keep?** Same frozen ESM2 embeddings, mean-pooled, linear head — the only difference from AETHER is the cross-attention, attention pooling and MLP |
| `pepmhc_archive` | did AETHER improve on the model it replaced? Scores the checkpoint in `archive_PEPMHC/` |
| `mhcflurry` | published external reference — see below |
| `netmhcpan` | field standard. License-gated — install the binary from DTU, then `--netmhcpan_bin` |

### Published tools

| Tool | Access | Status here |
|---|---|---|
| [MHCflurry 2.x](https://github.com/openvax/mhcflurry) | `pip install mhcflurry`, open source | **working** — see below |
| [NetMHCpan 4.1](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/) | academic licence, manual download | adapter written, needs the binary |
| [IEDB standalone tools](https://tools.iedb.org/mhci/download) | free tarball | bundles SMM, SMMPMBEC, ANN, consensus |
| [MHCnuggets](https://github.com/KarchinLab/mhcnuggets) | `pip install mhcnuggets` | not wired up yet |
| [MixMHCpred](https://github.com/GfellerLab/MixMHCpred) | free academic download | not wired up yet |

#### Running MHCflurry

MHCflurry needs TensorFlow, which should not share an environment with the
PyTorch training stack. So it runs from its own venv via a standalone script
that imports nothing from `neoantigen`:

```bash
python -m venv ~/venv/mhctools
~/venv/mhctools/bin/pip install mhcflurry
~/venv/mhctools/bin/mhcflurry-downloads fetch models_class1_presentation
```

Then a two-pass run. The first pass writes `_target_rows.csv` — the prepared
rows, already cleaned and in scoring order — so the external tool scores exactly
the same rows as everything else instead of re-implementing the cleaning:

```bash
# 1. baselines, which also dumps _target_rows.csv
pixi run python -m neoantigen.benchmark.run \
  --data_dir data/mhcattnnet/classI_peptide_split --split test \
  --out_dir outputs/benchmarks

# 2. score those rows with MHCflurry, in its own environment
~/venv/mhctools/bin/python src/neoantigen/benchmark/run_mhcflurry.py \
  --target_rows outputs/benchmarks/_target_rows.csv \
  --out_csv outputs/benchmarks/mhcflurry_predictions.csv

# 3. fold it back in
pixi run python -m neoantigen.benchmark.run \
  --data_dir data/mhcattnnet/classI_peptide_split --split test \
  --out_dir outputs/benchmarks \
  --include mhcflurry=outputs/benchmarks/mhcflurry_predictions.csv
```

One API trap worth knowing: `Class1PresentationPredictor.predict()` reads
`alleles` as the allele set of *one sample* (at most 6) and returns the
strongest prediction across them — it does **not** pair alleles with peptides
element-wise the way the affinity predictor does. `run_mhcflurry.py` therefore
scores one allele at a time, so each score is a genuine (peptide, allele) pair
rather than a max over a set.

### Reading the results

Compare `within_allele_auroc_common`: support-weighted AUROC within each
eligible allele, using the rows scored by every method. The default benchmark
requires at least 50 scored rows and both labels per allele. Coverage,
`n_common`, eligible alleles and excluded support accompany the result.
Pooled AUROC remains available as `auroc_common`, but can reward allele
separation instead of peptide ranking.

The [2026-10-27 audited report](results/20261027/README.md) contains the clean
AETHER reruns, NetMHCpan 4.2d, BigMHC EL, MHCflurry and the classical baselines,
with checkpoint hashes and per-allele comparisons. It supersedes the historical
summary formerly shown here, whose AETHER checkpoint provenance was unresolved.

The allele-prior baseline's approximately 0.32 gap demonstrates the problem with
pooled AUROC; that gap is model dependent and is not a universal correction to
subtract from other models. AUPRC is average precision. PPV@100 is also recorded,
with expected credit for score ties at the cutoff. Allele priors can achieve
near-perfect pooled PPV by selecting high-prevalence alleles, so inspect the
per-allele PPV tables when assessing peptide triage.

## Data

`data/mhcattnnet/` — see its [README](data/mhcattnnet/README.md) for schema and
statistics.

```
data/mhcattnnet/
├── classI/                   original row split — regroup before training
│   ├── train.csv  val.csv  test.csv
│   └── new_mhc_split/        train/test only; test alleles unseen in training
├── classI_peptide_split/     peptide-disjoint default, built by `pixi run regroup`
├── classI_allele_sequence_split/  held-out allele names and encoded MHC sequences
├── classII/
└── 5_cross_val/              5-fold splits for classI and classII
```

The original class-I row split has 343,713 training rows. The deduplicated,
peptide-disjoint default has 279,527 training rows, 32,777 validation rows and
35,339 test rows. The stricter allele protocol has 274,923 training rows,
35,857 validation rows and 36,863 test rows. Machine-readable counts, overlap
checks and input hashes are in `results/20261027/data_audit.json`.

---

## Architecture

1. **ESM2 backbone** (`facebook/esm2_t33_650M_UR50D`, 1280-dim), shared between
   the two sequences and frozen by default.
2. **Separate linear projections** to `d_proj` for peptide and MHC — a
   9-residue peptide and a 275-residue groove play different positional roles.
3. **N interaction layers**: peptide attends to MHC, MHC attends to the updated
   peptide, optionally followed by self-attention within each modality.
4. **Learned attention pooling** per modality, padding masked out.
5. **MLP head** over `[pep, mhc]`, or `[pep, mhc, pep*mhc]` with
   `--use_product_head`.

### Why the MHC side is precomputed

The original 343,713 training rows carry only **152 distinct MHC sequences**, each ~275
residues against a ~9-residue peptide. Encoding the MHC row-by-row repeats the
same forward pass ~2,000× per sequence, on the side that is ~30× longer — so
that is where the backbone's entire time budget would go.

AETHER runs the frozen backbone once over the unique MHC sequences and pins the
result (`--precompute_mhc`, default on). The cache then *seals*: the 160k unique
peptides are encoded live and never grow it. Unfreezing clears it automatically,
since those embeddings would otherwise go stale.

`figures/01_dataset_overview.ipynb` plots the evidence.

---

## Known issues and caveats

Read this before quoting a number.

**1. The shipped splits leak.** The original `classI/` split partitions by row,
so 63.1% of test peptides and **35.8% of test (peptide, allele) pairs appear
verbatim in train**, and 27.5% of the pooled rows are exact duplicates. Any
AUROC measured on that split is inflated by memorisation. Use
`pixi run regroup` and train on `classI_peptide_split/`.

**2. 77% positives is not a real base rate.** In a genuine screen, a small
fraction of candidate peptides bind a given allele. The class balance here comes
from assay-selection bias — peptides get tested because someone suspected they
bind. Consequences: accuracy is near-useless (a constant "binder" scores 0.774),
AUPRC looks flattering against a 0.77 baseline, and predicted probabilities are
not calibrated to anything biological. For screening, what matters is precision
in the top *k*, not global AUROC.

**3. Binding ≠ presentation ≠ immunogenicity.** This model predicts *binding*
only. The path to a T-cell response also needs proteasomal cleavage, TAP
transport, stable surface presentation, and TCR recognition. Mass-spec eluted
ligand data is a better proxy for presentation than binding affinity, and
immunogenicity is harder still.

**4. The full MHC sequence is probably more than is needed.** Sequences here run
180–347 residues, meaning some entries are partial, and binding-rate varies by
length bucket (0.69–0.88) — so length is partly predictive and the model can use
it as an allele fingerprint rather than learning binding chemistry. Established
methods use a ~34-residue pseudo-sequence of the polymorphic groove positions
instead. Worth trying.

**5. The binary label discards information.** The underlying IEDB measurements
are IC50 values. Binarizing at a threshold throws away the gradient; training on
`1 - log50k(IC50)` as a regression target typically does better and lets you
keep AUROC by thresholding at eval time.

---

## Repository layout

```
AETHER/
├── pixi.toml                 environment and tasks
├── configs/                  run configurations (JSON)
├── data/
│   ├── mhcattnnet/           peptide-MHC binding splits — the training data
│   ├── CEDAR/                Cancer Epitope Database (T-cell assay data)
│   ├── HLA/                  HLA allele protein FASTA (IPD-IMGT/HLA)
│   └── model/                processed splits and export scripts
├── figures/                  all plotting, as notebooks; PNGs written here
├── src/neoantigen/
│   ├── models/               esm2.py backbone, transformer.py cross-attention
│   ├── data/                 regroup.py — leakage-free splits
│   ├── benchmark/            baselines.py, external.py, run.py
│   ├── train/                train.py, dataloader.py
│   └── infer/                predict.py, evaluate.py
├── outputs/                  run artifacts (gitignored)
└── archive_PEPMHC/           retired Siamese n-gram model, kept for reference
```

### Developer notes

- Backbone is **ESM2** via HuggingFace `transformers`. The ESM3 encoder
  (`models/esm3.py`) remains selectable with `--model_name esm3_sm_open_v1` but
  needs EvolutionaryScale's `esm` package; it is imported lazily so its absence
  never breaks the ESM2 path.
- Dataset column names resolve by alias, so both the MHCAttnNet schema
  (`peptide` / `mhc_amino_acid` / `bind`) and the older AETHER schema
  (`epitope` / `mhc_sequence` / `mhc_binding`) load without edits.
- `val.csv` is optional. If a split directory has only train/test, a validation
  slice is carved out of train, grouped by (peptide, MHC) pair so nothing leaks.

## Clean benchmarking update — 2026-10-27

The default binding configs now point to the peptide-disjoint MHCAttnNet split.
See [training protocols](src/neoantigen/train/README.md#audited-training-protocols-2026-10-27)
for separate peptide and held-out-allele runs, fresh-directory enforcement, and
checkpoint-linked metrics. New benchmark adapters use the locally downloaded
NetMHCpan 4.2d standard EL predictor and BigMHC EL ensemble, alongside MHCflurry
and classical baselines. Benchmark ranking uses support-weighted within-allele
AUROC on the common scored rows, with AUPRC, PPV@100, coverage and per-allele CSVs.

```bash
python -m neoantigen.benchmark.run \
  --data_dir data/mhcattnnet/classI_peptide_split \
  --out_dir outputs/my_benchmark \
  --methods majority allele_prior allele_length_prior pssm blosum_lr netmhcpan bigmhc \
  --netmhcpan_bin scripts/netmhcpan42 --bigmhc_dir tools/bigmhc \
  --include aether=outputs/my_clean_peptide_run/test_predictions.csv
```

NetMHCpan and BigMHC are evaluated on standard-residue 8–15mers; declined rows
remain missing and reduce coverage. NetMHCpan uses `1 - %Rank_EL/100` as a ranking
score, without `-neo` or `-pathogen`. BigMHC uses EL presentation scores, accepts
only exact supported allele identities (no fuzzy substitution), and realigns
its sorted output by peptide–allele key. Binding and presentation are different
targets; these are reference rankings against the binding labels. The pretrained
external tools may have seen overlapping IEDB/CEDAR records; local split hygiene
does not establish that their pretraining data were independent.

MHCflurry can run in `.venvs/mhctools` with the standalone
`src/neoantigen/benchmark/run_mhcflurry.py`, consuming the benchmark's exact
`_target_rows.csv`. Fold its predictions back in using `--include`. Included
predictions must have peptide/allele identity and matching labels and order.
`_common_rows.csv` records the shared scored population. Within-allele AUROC
excludes single-class alleles and, by default, alleles with fewer than 50 scored
rows; both exclusions and eligible support are explicit in the summary.

The working tree already includes an optional TCR head and a separate phase-two
CEDAR pipeline. That is not a validated immunogenicity/presentation model. No
untrained output heads or language-generated biological claims were added.
[Attention export](src/neoantigen/infer/README.md#attention-exports) now makes
existing interaction/pooling weights available for inspection. CEDAR's small,
conflicted and overlapping labels remain a data limitation; deduplication and
peptide grouping address leakage, not label quality or external validity.

The published peptide split is frozen in
`results/20261027/peptide_split_assignments.csv.gz`. `pixi run regroup` restores
these assignments and their original row order. To rebuild into a separate
folder directly:

```bash
python -m neoantigen.data.regroup --in_dir data/mhcattnnet/classI \
  --out_dir outputs/restored_peptide_split \
  --assignments_file results/20261027/peptide_split_assignments.csv.gz
```

The current randomized regrouping defaults produce a different, also disjoint
partition of the same cleaned corpus. Use the frozen assignments to reproduce
the reported experiments; training seed 1337 alone does not specify this
pre-existing split. `peptide_split_restoration.json` records matching canonical
hashes for all three restored files.
