# AETHER poster figures — self-contained bundle

Everything needed to regenerate the seven poster panels, with no dependency on
the rest of the repo. Copy this folder anywhere and it still works.

## Running it

Open `04_poster_figures.ipynb` and select the **`AETHER (bigrna-torch)`**
kernel, or regenerate every PNG headlessly:

```bash
cd figures/poster
~/conda/envs/bigrna-torch/bin/python -m nbconvert \
    --to notebook --execute --inplace 04_poster_figures.ipynb
```

Run it **from this directory**. The notebook locates its inputs by walking up
from the working directory until it finds a `data/` folder, which is why the
`data/` and `outputs/` trees below are mirrored here — the notebook is an
unmodified copy of `figures/04_poster_figures.ipynb` and needs no edits.

If you move the folder to a machine without that conda environment, any Python
with `pandas`, `numpy`, `scikit-learn` and `matplotlib` will do. Nothing here
needs torch, a GPU, or network access.

## What is here

```
poster/
├── 04_poster_figures.ipynb          the notebook (unmodified copy)
├── poster_*.png                     the seven rendered panels, 600 dpi
├── outputs/benchmarks/              model scores
│   ├── benchmark_meta.json          split name, common-subset row count
│   ├── benchmark_summary.csv        the metrics table
│   └── <method>_predictions.csv     per-row y_true / y_pred_prob / peptide / allele
└── data/mhcattnnet/classI/          the raw split
    └── train.csv  val.csv  test.csv
```

The `data/` copies are only read by two panels — `poster_leakage` (peptide and
pair overlap between train and test, plus the duplicate-row count) and
`poster_dataset` (the 77% positive rate). They are also the bulk of the
bundle: ~157 MB apparent, of which `train.csv` is ~103 MB. If you need
something lighter to carry around, those two panels could read a small
precomputed JSON instead — ask and I will wire that up.

`pepmhc_archive` is deliberately absent. The notebook excludes it via
`POSTER_EXCLUDE`, so it is not needed to produce these figures.

## The figures

| File | Role |
|---|---|
| `poster_metric_decomposition.png` | **hero** — pooled vs within-allele AUROC, with the +0.32 giveaway called out |
| `poster_leakage.png` | why the shipped split cannot be used as-is |
| `poster_dataset.png` | the 77% base rate |
| `poster_benchmark.png` | all methods on within-allele AUROC |
| `poster_roc.png` | ROC curves |
| `poster_architecture.png` | what AETHER is |
| `poster_per_allele.png` | where it wins and loses, against MHCflurry |

## The numbers these show

Clean peptide-grouped split, 35,339 test rows, 66 alleles with >= 50 rows.
Compare on **within-allele AUROC** — AUROC computed per allele then averaged by
support. Pooled AUROC on this dataset is inflated by ~0.32, because 77% of rows
are binders and alleles differ in how often they bind.

| method | pooled | within-allele |
|---|---|---|
| **AETHER** | **0.964** | **0.933** |
| MHCflurry | 0.941 | 0.920 |
| PSSM (per allele) | 0.911 | 0.901 |
| ESM2 linear probe | 0.867 | 0.720 |
| BLOSUM + logistic | 0.868 | 0.710 |
| Allele x length prior | 0.862 | 0.658 |
| Allele prior | 0.819 | 0.500 |
| Majority class | 0.500 | 0.500 |

MHCflurry declined 536 of 35,339 rows (535 outside its 8-15mer range, 1 with a
non-standard residue), so it is compared at 98.5% coverage on the common subset.

Two caveats worth carrying onto the poster:

- AETHER beats MHCflurry on **32 of 66 alleles** — close to a coin flip by
  count. Its margin comes from the high-support alleles. The pooled number
  should not be read as uniform superiority.
- Two training processes wrote to the source run directory concurrently, so
  which of them produced the weights behind `aether_predictions.csv` cannot be
  established. Both were 10-epoch runs on this split with identical config. A
  clean re-run into a fresh directory would settle it.
