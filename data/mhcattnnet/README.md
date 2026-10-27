# MHCAttnNet peptide–MHC binding dataset

Moved here from `archive_PEPMHC/MHCAttnNet_dataset/` so the data AETHER actually
trains on lives under `data/`. The archive keeps only what belongs to the
retired model: the *n*-gram vector files (`1-gram-vectors.txt` … `4-gram-vectors.txt`)
and the old `models/` checkpoints, none of which AETHER reads.

## Layout

```
data/mhcattnnet/
├── classI/              # MHC class I — the default training set
│   ├── train.csv  val.csv  test.csv
│   └── new_mhc_split/   # train/test only, split so test alleles are unseen
├── classII/             # MHC class II
│   └── train.csv  val.csv  test.csv
└── 5_cross_val/         # 5-fold splits, {1..5}_{train,val,test}.csv
    ├── classI/
    └── classII/
```

## Schema

Every CSV has four columns:

| column            | meaning                                              |
|-------------------|------------------------------------------------------|
| `peptide`         | peptide amino acid sequence                           |
| `mhc_amino_acid`  | full MHC protein sequence for the allele              |
| `bind`            | **label** — 1 = binder, 0 = non-binder                |
| `mhc_allele`      | allele name, e.g. `HLA-A*02:01`                       |

`src/neoantigen/train/dataloader.py` resolves these names by alias, so the files
load unchanged.

## Class I statistics

| split | rows    | positive | unique peptides | unique MHC seqs | unique alleles |
|-------|---------|----------|-----------------|-----------------|----------------|
| train | 343,713 | 77.4%    | 160,404         | 152             | 161            |
| val   |  73,652 | 77.1%    |  54,086         | 143             | 152            |
| test  |  73,653 | 77.4%    |  54,104         | 141             | 150            |

Peptides are 3–43 residues (median 9); MHC sequences are 180–347 residues
(median 275).

Two properties of this table drive the training design:

1. **~150 unique MHC sequences carry ~344k rows.** Re-encoding the MHC side
   every step repeats the same forward pass thousands of times, and the MHC
   sequence is ~30× longer than the peptide, so that is where the backbone's
   time would go. AETHER precomputes those embeddings once
   (`--precompute_mhc`, on by default with a frozen backbone).
2. **The classes are imbalanced (77% binders).** Accuracy is therefore a weak
   measure — a constant "binder" prediction scores 0.774. Use AUROC (and AUPRC)
   instead; `--pos_weight auto` is available if calibrated probabilities matter.

Note that `train`/`val`/`test` here split on rows, so a peptide may recur across
splits with a different allele. `classI/new_mhc_split/` is the stricter test:
its test alleles are held out from training, which measures generalization to
unseen MHC alleles rather than to unseen peptides.
