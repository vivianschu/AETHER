# Figures

All plotting for AETHER lives here, as notebooks. `train.py` draws nothing — it
writes metrics, and these notebooks turn them into figures, so plots can be
restyled or regenerated without re-running training.

| Notebook | What it plots |
|---|---|
| `01_dataset_overview.ipynb` | the peptide-MHC dataset: class balance, peptide length, MHC redundancy, allele coverage |
| `02_training_progress.ipynb` | a training run: loss and AUROC/AUPRC curves, ROC/PR, calibration, per-allele AUROC |
| `03_benchmark_comparison.ipynb` | AETHER against the reference models: AUROC bars, ROC overlay, precision@k, per-allele scatter |
| `04_poster_figures.ipynb` | poster-sized panels — larger type, heavier marks, written as `poster_*.png` at 600 dpi / 10 in wide |
| `05_results_20261027.ipynb` | audited 2026-10-27 runs: multi-peptide 9-mer attention, substitutions, motifs, matched ROC/PR, reliability, lengths, per-allele comparisons; PNG in `results/20261027/plots` |

Run them from this directory. Each cell writes one standalone 600 dpi PNG here.

## Which kernel to use

A kernel named **`AETHER (bigrna-torch)`** is registered and is the one to pick
from the kernel dropdown in VS Code or Jupyter. It points at the conda
environment that has torch, transformers, pandas, scikit-learn and matplotlib:

```
/home/vivian.chu/conda/envs/bigrna-torch/bin/python
```

It was registered with:

```bash
~/conda/envs/bigrna-torch/bin/python -m ipykernel install \
    --user --name aether --display-name "AETHER (bigrna-torch)"
```

Re-run that if it ever disappears from the list. Do **not** use the `scgpt`
kernel — that environment has no `transformers` and the notebooks will fail on
import.

To launch Jupyter rather than use the IDE:

```bash
cd figures
~/conda/envs/bigrna-torch/bin/jupyter lab
```

Once `pixi install` has been run, `pixi run lab` does the same inside the pixi
environment instead, and that becomes the preferred route.

### Running a notebook headlessly

This is what `--render_figures` does, and it is the quickest way to refresh
every PNG without opening anything:

```bash
cd figures
AETHER_RUN_DIR=../outputs/classI_peptide_split \
  ~/conda/envs/bigrna-torch/bin/python -m nbconvert \
  --to notebook --execute --inplace 02_training_progress.ipynb
```

Both run-specific notebooks take an environment variable so they can be pointed
at a particular run without editing:

```bash
AETHER_RUN_DIR=../outputs/my_run     jupyter nbconvert --execute --inplace 02_training_progress.ipynb
AETHER_BENCH_DIR=../outputs/my_bench jupyter nbconvert --execute --inplace 03_benchmark_comparison.ipynb
AETHER_BENCH_DIR=../outputs/my_bench jupyter nbconvert --execute --inplace 04_poster_figures.ipynb
```

## Poster figures

`04_poster_figures.ipynb` writes `poster_*.png` at 600 dpi and 10 in wide —
scale them down into any column. Change `POSTER_WIDTH` in its style cell for a
different base width. The narrative is findings-forward:

| File | Role |
|---|---|
| `poster_metric_decomposition.png` | **hero** — pooled vs within-allele AUROC |
| `poster_leakage.png` | why the shipped split cannot be used as-is |
| `poster_dataset.png` | the 77% base rate |
| `poster_benchmark.png` | all methods on within-allele AUROC |
| `poster_roc.png` | ROC curves |
| `poster_architecture.png` | what AETHER is |
| `poster_per_allele.png` | where it wins and loses |

The architecture schematic is drawn in matplotlib rather than a vector editor so
it stays in sync with the model and re-exports with everything else.

## Watching a run in progress

`02_training_progress.ipynb` is safe to re-run while training is going:
`metrics.jsonl` is appended and flushed after every evaluation, so each run of
the notebook picks up the curves so far. Set `RUN_DIR` at the top to the run you
want.

Before the first epoch finishes there are no full validations yet, so the
notebook falls back to the mid-epoch checks (`--eval_every_steps`) and says so
in the titles. Those use a capped number of batches and are noisier than the
end-of-epoch numbers. The test-set sections skip themselves until the run
reaches the test stage.

## Style

Follows `261019_immunopeptidome_overview.ipynb`: Arial 10, 600 dpi, Tableau-10
colors, top/right spines hidden, `#f0f0f0` grid, Title Case titles, left-aligned
frameless legends outside the axes. Each figure stands alone — no subplots.

The shared style block (`new_fig`, `save_fig`, `left_legend`, the color roles)
is duplicated at the top of both notebooks rather than imported, so each one
runs standalone.

## Expanded 2026-10-27 analysis

Run `python scripts/analyze_attention_cohort.py` from the repository root first, then execute `05_results_20261027.ipynb`. This notebook saves 300 dpi PNG figures beside the audited results. Read [the analysis report](../results/20261027/attention_analysis.md) for cohort definitions, biological interpretation, precision differences, and limits.
