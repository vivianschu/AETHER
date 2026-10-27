# `workflow` — scoring the PMLB immunopeptidome

Scores the sample-specific immunopeptidome from `pmlb-neoantigens` against each
sample's own class-I HLA, using the checkpoint wrapped by [`pepmhc`](../pepmhc).
**Read [`pepmhc/README.md`](../pepmhc/README.md#what-the-scores-are-worth)
first** — the checkpoint is weak, and the figures are built to show that rather
than hide it.

Run in order (an env with torch, pandas, pyarrow, scikit-learn and matplotlib —
`conda activate organoids` works):

```bash
python workflow/validate_model.py    # picks the config, measures what it is worth
python workflow/diagnose_model.py    # peptide- vs allele-driven score variance
PYTHONPATH=workflow python workflow/score_peptides.py
python workflow/make_figures.py
```

## Inputs

| what | where |
|---|---|
| database-search peptides | `data/Fragpipe_SampleSpecificResults/<alias>.peptide.tsv` |
| de novo peptides | `data/Casanovo_DeNovoResults/<alias>.casanovo.denovo.mztab` |
| HLA calls | `results/hla/hla_summary.csv` |
| alias → organoid barcode | `workflow/sample_specific_search/analysis_outputs/261019_immunopeptidome/sample_table.csv` |
| background proteome | `workflow/mutation_reference/data/2025-04-01-reviewed-UP000005640.fas` |

all under `/Users/vivianschu/repos/pmlb-neoantigens`.

Three sample namespaces have to be bridged: the MS file stem (`PROT109_He_LX108`),
the PMLB organoid barcode (`LX0108.TXO`) and the HLA table's `sample`. `alias`
→ `pmlb_organoid_id` comes from `sample_table.csv`; the barcode is then matched
to `hla_summary.sample` after stripping `_pmlb`/`_novo`/`.rN` suffixes, reusing
`sample_pairing.normalize_barcode`. That pairs **33 of 45** FragPipe and **73 of
171** Casanovo samples; the rest have no `pmlb_organoid_id` or no HLA call and
are dropped rather than guessed at.

## Filters

* class-I ligand lengths, 8–11 aa, plain amino acids only;
* FragPipe `Probability >= 0.9`, Casanovo `search_engine_score[1] >= 0.9`;
* Casanovo ProForma modifications (`C[Carbamidomethyl]`, `[Acetyl]-`, …)
  stripped to bare residues, matching the repo's own
  `str.replace(r"[^A-Z]", "")`;
* peptides deduplicated per sample, then each distinct (peptide, allele)
  combination scored once.

Each peptide is scored against all six class-I alleles the sample carries
(A/B/C × 2) and summarised by the allele that ranks it best.

## Outputs

`results/`
: `peptide_allele_scores.parquet` (every peptide × allele), `peptide_scores.parquet`
  (best allele per sample × peptide, with `binder`/`strong_binder`),
  `background_scores.parquet`, `validation.csv`, `validation_metrics.csv`,
  `config_sweep.csv`, `diagnostic_grid.csv`.

`figures/`
: `fig1` %rank distribution by source · `fig2` binder rate per sample ·
  `fig3` binders per sample · `fig4` binder rate by peptide length ·
  `fig5` best-ranking allele · `fig6` raw score as an allele offset ·
  `fig7` held-out ROC and the saturated score band ·
  `fig8` what moves the score.

600 dpi PNGs, Arial, the palette and `box_strip`/`bare_x` helpers of
`261020_poster.ipynb` (see `figures_style.py`).

## Why %rank and not the score

The raw output sits above 0.5 for every peptide and spans 0.0102 across the
entire held-out split, so the model's own threshold calls everything a binder.
Worse, 95% of the score variance is between alleles and ~5% between peptides —
the score is close to a per-allele constant. `background.py` therefore ranks
each peptide against 5,000 random natural peptides on the *same* allele, which
removes that offset, and binder calls use netMHCpan's conventional `%rank <= 2`
(strong `<= 0.5`). This makes the numbers comparable across alleles; it does
not make the underlying model good.
