# CEDAR — phase-2 training data

Peptide–MHC and TCR data from the **Cancer Epitope Database and Analysis
Resource**, pulled for a second training stage on top of the IEDB-derived
phase-1 set in `data/mhcattnnet/`.

## How it was obtained

CEDAR runs the same PostgREST stack as IEDB's IQ-API, so the web UI's
"export results" tables are all reachable as URLs — no manual download, no
timestamped filenames:

```bash
python download_cedar.py mhc   -o raw/mhc_search.csv   # ~1.03M assay rows, 129 MB
python download_cedar.py tcr   -o raw/tcr_search.csv   # 12,066 receptor groups
python download_cedar.py hla   -o ../HLA/hla_prot.fasta
python download_cedar.py build -o build
```

Base URL `https://cedar-api.iedb.org`, schema at `/docs/swagger/`.
Filters follow `data/CEDAR/README.md` (human host, non-viral, non-germline,
class I, HLA alleles, qualitative measurement present) but are applied as
column predicates — `viral_antigen_bool=eq.0` rather than antigen-name
matching. Downloaded 2026-10-20 against the CEDAR build of 2026-10-14.

`data/CEDAR/` holds the superseded route: a notebook reading six UI exports
that are not in the repo.

## The overlap with phase 1 is the headline

CEDAR curates largely the same literature as IEDB, so phase-1 rows reappear
here. `build` drops any `(peptide, 2-field allele)` already present anywhere
in `data/mhcattnnet/` — matching on 2-field allele so that
`HLA-A*02:01:01:02N` and `HLA-A*02:01` count as the same restriction.

Of 1,029,503 assay rows, **537,020 (52%) are phase-1 pairs.** Broken down by
label, the asymmetry is what matters:

| | already in phase 1 | new |
|---|---|---|
| negative | 45,444 (98%) | **1,053** |
| positive | 491,576 | 473,517 |

**CEDAR contributes almost no new negatives.** Spot-checked: `DLERDLRKI`,
`KIKKKIKKL`, `HLSKEEEEEL` and `SVPESPFSRT` on HLA-A\*02:01 are all present in
`classI/train.csv` with `bind=0`. The 47k negatives that the raw counts
promise are, nearly all of them, rows the model has already trained on.

## Outputs

| file | rows | alleles | contents |
|---|---|---|---|
| `build/cedar_train_pool.csv` | 2,409 | 64 | measured negatives + neoantigens; 946 neg / 1,463 pos |
| `build/cedar_ms_ligands.csv` | 321,185 | 151 | eluted ligands, **positive only** |
| `build/cedar_tcr.csv` | 11,577 | 53 | peptide / MHC / CDR3α / CDR3β |

Columns match `data/mhcattnnet/`: `peptide`, `mhc_amino_acid`, `bind`,
`mhc_allele`, plus `bind_graded` (Positive-Low/Intermediate/High → 0.25/0.5/0.75),
`neoantigen`, `n_assays` and `label_conflict`.

`cedar_train_pool.csv` is small enough to be a fine-tuning or evaluation set,
not a training stage on its own. `cedar_ms_ligands.csv` is positive-only, so
it needs decoys before it can be trained on and must never be used as a
validation set. 229,301 of its peptides are new to the project.

### MHC sequences

`mhc_amino_acid` reuses the exact string `data/mhcattnnet/` stores for that
allele (210 alleles), so precomputed MHC embeddings stay valid across both
phases. New alleles fall back to `hla_prot.fasta` with the signal peptide
removed.

The leader length is **learned, not assumed**: for alleles present in both
sources, `calibrate_trim` finds the offset at which the stored sequence sits
inside the fasta entry. Every class-I locus agrees on **25** (not the
textbook 24) across 121 alleles, and HLA-DRB on 34. Hardcoding 24 would shift
every novel allele's sequence by one residue.

Caveat inherited from phase 1: mhcattnnet's own sequences are not uniform —
33 of 161 class-I alleles store only α1+α2 (181 aa) while most store the full
mature chain (337–341 aa). Novel alleles get the full mature chain.

### Assay collapsing

CEDAR records one row per assay, so a pair can appear dozens of times; left
unmerged, a random split would put the same pair on both sides. Rows are
collapsed to one per `(peptide, allele)`, keeping `n_assays`. 72 pairs had
both positive and negative assays; those resolve to positive and are marked
with `label_conflict=1`.

### TCR set

Not deduplicated against phase 1 — the label is receptor recognition, not
peptide–MHC binding, so a shared `(peptide, allele)` is a different example.
`pmhc_in_phase1` marks the 6,785 rows whose pMHC the model has seen, for
evaluations that need to exclude them. 10,840 of 11,577 are positive and
1,442 are neoantigen-derived.

## Known gaps

- The pull retrieved 1,029,503 of the 1,029,528 rows the count endpoint
  reported. Offset pagination against a live database; re-run to confirm.
- 17,913 rows were dropped for allele resolution too coarse to attach a
  sequence (`HLA class I`, `HLA-A2`), and 5,705 for an allele absent from
  `hla_prot.fasta`.
- Class II was not pulled. `download_cedar.py mhc --mhc-class II` will.
