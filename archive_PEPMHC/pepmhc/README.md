# `pepmhc` — inference for the trained class-I binder

Before this package the repo had no way to make a prediction: every script under
`MHCAttnNet_ft/src/` is a *training* entry point, each one ending in a
`trainer.train()` or `fit(...)` call, and the only line that loads a checkpoint
(`model.load_state_dict(torch.load(config.model_name))`, `train.py:537`) is
commented out. This package adds the missing path.

```python
from pepmhc.predictor import Predictor

p = Predictor()
p.predict(["KRVTEILGF", "AANPHSFVF"], "A*02:01:01")   # many peptides, one allele
p.predict_pairs(df.peptide, df.allele)                # paired, elementwise
p.supported_alleles()                                 # the 161 it can score
```

Scores are `P(bind)` in `[0, 1]`; `NaN` means the peptide or the allele could
not be used (see *Limits*). **Read "What the scores are worth" before using
them.**

## The checkpoint

`MHCAttnNet_dataset/models/classI/check.pt` — 64 tensors, 132 MB, class I.
It is **not** MHCAttnNet: there are no LSTM and no attention-layer weights.
It is a Siamese transformer, and no single class in the source tree matches it,
because the tree moved on after the checkpoint was written:

| tensor group | shape | matches |
|---|---|---|
| `peptide_embedding` | `(23, 100)` | 1-gram peptide vocab |
| `mhc_embedding` | `(365, 100)` | 3-gram MHC vocab |
| `{pep,mhc}_pos_encoder.pos_encoding` | `(500, 1, 100)` | `transformer.py`'s `max_len=500` |
| `{pep,mhc}_transformer_encoder.layers.{0,1}` | `d_model=100`, `ff=512` | `transformer.py`'s `dim_feedforward=512`, with `num_layers=2` |
| `{peptide,mhc}_linear0` | `(2048, 4500)`, `(2048, 12000)` | `mlp.py`'s flattened route — **dead weights**, nothing consumes their output |
| `{peptide,mhc}_linear` | `(64, 100)` | `mlp.py`'s `Linear(100, LINEAR1_OUT)`, so pooling is a mean over positions |
| `hidden_linear`, `out_linear` | `(64, 128)`, `(2, 64)` | both files |

`model.py` reproduces that, `*_linear0` included, so the state dict loads with
`strict=True` and nothing is silently re-initialised.

## The vocabularies

Nothing about the vocabularies was saved with the weights, and they cannot be
read back out of the embedding matrices: those were fine-tuned, so they no
longer match the `n-gram-vectors.txt` files they were initialised from (a
nearest-neighbour match assigns `L`, `Y`, `D` and `N` to two rows each at
distances of 0.5–1.4 — useless).

`vocab.py` reconstructs them from the training CSVs instead, replicating
`torchtext.legacy`'s ordering: specials first, then the counter listed
alphabetically and *stably* re-sorted by frequency, descending, counted over
`classI/train.csv` + `classI/val.csv` (`data_loader.IEDB.splits` defaults to
`config.base_path`). Three independent checks agree:

* sizes land on exactly 23 and 365, the numbers baked into the checkpoint;
* the peptide order comes out `L V A S I F R E T K P Y G D Q N H M W C X`,
  matching the comment in `data_loader.py` ("`L (2), V (3), A (4) ....X (22)`");
* index 0 is the only all-zero row in both embeddings — `<unk>`, the one token
  that never appears and so never received a gradient, while `<pad>` at index 1
  did move.

The allele → MHC-sequence table comes from the same CSVs, which carry the full
alpha-chain sequence next to each allele name.

## Two quirks, reproduced on purpose

The training code has two bugs that are part of the trained function, so
`model.py` keeps them:

1. `PositionalEncoding.forward` slices `pos_encoding[:token_embedding.size(0)]`
   on a **batch-first** tensor, so the offset added is indexed by a sample's
   position *in the batch*, not by sequence position.
2. `nn.TransformerEncoder` is left at `batch_first=False` and fed a batch-first
   tensor, so self-attention runs **across the batch**, mixing unrelated
   peptides.

Together these make a score depend on the batch it was scored in. `Predictor`
therefore pins `batch_size=32` (`config.batch_size`) and scores a fixed,
sorted list of pairs, so results are reproducible. `workflow/validate_model.py`
measures the residual sensitivity.

Rewiring the encoder to the semantically correct `(seq, batch, feature)` layout
(`Predictor(seq_first=True)`) was tested and scores *worse*, which is itself
evidence that the quirks are what the weights were trained under.

## Limits

* **Class I only**, and only the **161 alleles** present in the training CSVs
  (`p.supported_alleles()`). The model consumes the MHC *amino-acid sequence*,
  not an allele name, so an allele with no sequence cannot be scored. Those
  return `NaN` rather than being mapped to a near neighbour.
* `normalize_allele` folds the HLA caller's 3-field, prefix-less names
  (`A*01:01:01`) onto the training table's 2-field names (`HLA-A*01:01`).
  Null alleles (`A*01:04N`) are dropped — not expressed, so not presentable.
* Peptides are truncated to 45 residues and must be plain amino acids;
  modifications have to be stripped before scoring.

## What the scores are worth

Measured on the held-out `classI/test.csv` (the split the torchtext training
path held out), after sweeping all 24 combinations of the hyperparameters the
checkpoint does not record. Best config: `seq_first=False, nhead=10,
activation="relu", relu_on_output=False`. **Read this before using the scores.**

| | |
|---|---|
| ROC AUC | **0.673** |
| PRC AUC | 0.864 (positive rate 0.775) |
| Accuracy | 0.775 — exactly the positive rate |
| Precision / Recall | 0.775 / **1.000** |
| Score mean ± sd | 0.9808 ± 0.0017 |
| Score range, all 20,000 rows | **0.0102** |

Three things follow, and all 24 configurations showed them:

1. **The checkpoint predicts "bind" for every peptide.** Recall is exactly
   1.000 and precision is exactly the base rate. Its own 0.5 boundary is below
   the minimum score it ever produces, so `score > 0.5` is not a usable call.
2. **The output is saturated.** Every score across the whole held-out split
   falls in a band 0.0102 wide, centred on 0.98.
3. **The score is mostly a function of the allele, not the peptide.** Over a
   200-peptide × 20-allele grid (`workflow/diagnose_model.py`), 95% of the
   score variance is between alleles and **4.9% is between peptides with the
   allele held fixed**. The per-peptide spread across alleles has a median
   (0.01687) equal to its maximum (0.01690) — the allele adds a near-constant
   offset that barely depends on the peptide.

On top of that, the batch quirks are not negligible at this scale: re-shuffling
which peptides share a batch moves a score by up to 0.0043, which is **42% of
the model's entire dynamic range**.

So the AUC of 0.673 is largely the allele offset carrying information about how
often that allele binds in this dataset, not peptide-level discrimination. For
comparison, `MHCAttNet_log` records ~0.92 for the project's other models, and
`train.py` writes this file at `epoch % 5 == 0`, so it may well be an early
checkpoint that never converged.

**Use `pct_rank`, not the raw score.** `workflow/background.py` divides out the
per-allele offset by ranking each peptide against random natural peptides on the
same allele (netMHCpan's convention: `%rank <= 2` binder, `<= 0.5` strong).
That makes the output interpretable and comparable across alleles, but it
cannot create discrimination the weights do not have. Treat the ranks as a
weak prior to order candidates, and not as evidence of presentation.
