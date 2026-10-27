"""Score the sample-specific immunopeptidome against each sample's own class-I HLA.

Inputs (pmlb-neoantigens):
  data/Fragpipe_SampleSpecificResults/<alias>.peptide.tsv   database search
  data/Casanovo_DeNovoResults/<alias>.casanovo.denovo.mztab de novo
  results/hla/hla_summary.csv                               per-organoid HLA calls
  workflow/sample_specific_search/.../sample_table.csv      alias -> pmlb_organoid_id

A peptide is scored against every class-I allele the sample carries (A/B/C, two
each) and summarised by its best-scoring allele, which is how a peptide is
actually presented. Writes workflow/results/peptide_scores.parquet.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pepmhc.predictor import Predictor, normalize_allele
import background as bg

PM = Path("/Users/vivianschu/repos/pmlb-neoantigens")
FRAGPIPE = PM / "data/Fragpipe_SampleSpecificResults"
CASANOVO = PM / "data/Casanovo_DeNovoResults"
HLA = PM / "results/hla/hla_summary.csv"
SAMPLE_TABLE = (PM / "workflow/sample_specific_search/analysis_outputs"
                   / "261019_immunopeptidome/sample_table.csv")
OUT = Path(__file__).resolve().parent / "results"

CLASS_I_COLS = ["a_1", "a_2", "b_1", "b_2", "c_1", "c_2"]
MIN_LEN, MAX_LEN = 8, 11          # class-I ligand length range
FRAGPIPE_MIN_PROB = 0.9           # PeptideProphet
CASANOVO_MIN_SCORE = 0.9          # Casanovo confidence


def normalize_barcode(s):
    """sample_pairing.normalize_barcode: drop batch/replicate suffixes."""
    return re.sub(r"(_pmlb\d*|_novo\d*|\.r\d+|_r\d+)+$", "", str(s))


def sample_alleles():
    """alias (MS file stem) -> sorted list of supported class-I alleles."""
    st = pd.read_csv(SAMPLE_TABLE).dropna(subset=["pmlb_organoid_id"])
    hla = pd.read_csv(HLA)
    hla = hla[hla.status == "success"]
    by_barcode = {}
    for _, r in hla.iterrows():
        by_barcode.setdefault(normalize_barcode(r["sample"]), r)

    out = {}
    for _, r in st.iterrows():
        row = by_barcode.get(normalize_barcode(r["pmlb_organoid_id"]))
        if row is None:
            continue
        alleles = {normalize_allele(row[c]) for c in CLASS_I_COLS}
        alleles.discard(None)
        if alleles:
            out[r["alias"]] = sorted(alleles)
    return out


def read_fragpipe(alias):
    df = pd.read_csv(FRAGPIPE / f"{alias}.peptide.tsv", sep="\t",
                     usecols=["Peptide", "Probability", "Spectral Count"])
    df = df[df.Probability >= FRAGPIPE_MIN_PROB]
    return pd.DataFrame({"peptide": df.Peptide.str.upper(),
                         "confidence": df.Probability,
                         "spectral_count": df["Spectral Count"]})


def read_casanovo(alias):
    """mzTab PSM rows; the ProForma mods in `sequence` are stripped to bare AAs."""
    rows = []
    with open(CASANOVO / f"{alias}.casanovo.denovo.mztab") as f:
        header = None
        for line in f:
            if line.startswith("PSH\t"):
                header = line.rstrip("\n").split("\t")
            elif line.startswith("PSM\t") and header:
                rows.append(line.rstrip("\n").split("\t"))
    if not rows:
        return pd.DataFrame(columns=["peptide", "confidence", "spectral_count"])
    df = pd.DataFrame(rows, columns=header)
    score = pd.to_numeric(df["search_engine_score[1]"], errors="coerce")
    df = df.assign(score=score)
    df = df[df.score >= CASANOVO_MIN_SCORE]
    pep = df.sequence.str.replace(r"[^A-Z]", "", regex=True)
    # one PSM == one spectrum; collapse repeats of the same peptide
    g = pd.DataFrame({"peptide": pep, "confidence": df.score}).groupby("peptide")
    return pd.DataFrame({"peptide": g.size().index,
                         "confidence": g.confidence.max().values,
                         "spectral_count": g.size().values})


def collect(alias_alleles):
    frames = []
    for source, directory, suffix, reader in (
            ("FragPipe", FRAGPIPE, ".peptide.tsv", read_fragpipe),
            ("Casanovo", CASANOVO, ".casanovo.denovo.mztab", read_casanovo)):
        for path in sorted(directory.glob(f"*{suffix}")):
            alias = path.name[: -len(suffix)]
            if alias not in alias_alleles:
                continue
            df = reader(alias)
            df = df[df.peptide.str.len().between(MIN_LEN, MAX_LEN)]
            df = df[~df.peptide.str.contains(r"[^ACDEFGHIKLMNPQRSTVWYX]", regex=True)]
            if df.empty:
                continue
            df = df.drop_duplicates("peptide")
            df.insert(0, "alias", alias)
            df.insert(0, "source", source)
            frames.append(df)
            print(f"  {source:9s} {alias:45s} {len(df):7,d} peptides", flush=True)
    return pd.concat(frames, ignore_index=True)


def chosen_config():
    """The config validate_model.py selected on the held-out split."""
    path = OUT / "validation_metrics.csv"
    if not path.exists():
        raise SystemExit("run workflow/validate_model.py first "
                         "(it chooses the config and records what it is worth)")
    s = pd.read_csv(path, index_col=0).iloc[:, 0]
    return dict(seq_first=str(s["seq_first"]) == "True",
                nhead=int(s["nhead"]),
                activation=str(s["activation"]),
                relu_on_output=str(s["relu_on_output"]) == "True")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-samples", type=int, default=None)
    ap.add_argument("--background-n", type=int, default=5000)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    alias_alleles = sample_alleles()
    if args.limit_samples:
        alias_alleles = dict(sorted(alias_alleles.items())[: args.limit_samples])
    print(f"{len(alias_alleles)} MS samples paired to class-I HLA")

    peptides = collect(alias_alleles)
    print(f"\n{len(peptides):,} sample x peptide rows "
          f"({peptides.peptide.nunique():,} unique peptides)")

    # explode to one row per (peptide, allele)
    peptides["allele"] = peptides.alias.map(alias_alleles)
    pairs = peptides.explode("allele", ignore_index=True)
    print(f"{len(pairs):,} peptide x allele rows")

    # score each distinct (peptide, allele) once, in a fixed order
    uniq = (pairs[["peptide", "allele"]].drop_duplicates()
            .sort_values(["allele", "peptide"], ignore_index=True))
    print(f"{len(uniq):,} distinct peptide x allele combinations to score")

    cfg = chosen_config()
    print(f"model config (from held-out selection): {cfg}")
    p = Predictor(**cfg)
    uniq["score"] = p.predict_pairs(uniq.peptide, uniq.allele)
    print(f"scored: {uniq.score.notna().sum():,} / {len(uniq):,}")

    # per-allele background of random natural peptides -> %rank, because the
    # raw score is saturated and only orders peptides within an allele
    lengths = sorted(peptides.peptide.str.len().unique())
    wts = [(peptides.peptide.str.len() == L).sum() for L in lengths]
    backdrop = bg.sample_background(n=args.background_n, lengths=tuple(lengths),
                                    weights=wts)
    print(f"\nscoring {len(backdrop):,} background peptides against "
          f"{uniq.allele.nunique()} alleles")
    back = bg.build(p, uniq.allele.unique(), backdrop,
                    cache=OUT / "background_scores.parquet")
    uniq["pct_rank"] = bg.pct_rank(uniq.score, uniq.allele, back)

    pairs = pairs.merge(uniq, on=["peptide", "allele"], how="left")
    pairs.to_parquet(OUT / "peptide_allele_scores.parquet", index=False)

    # best allele per sample x peptide
    # a peptide is presented by whichever of the sample's alleles ranks it best
    best = (pairs.sort_values("pct_rank")
                 .drop_duplicates(["source", "alias", "peptide"])
                 .rename(columns={"allele": "best_allele", "score": "best_score",
                                  "pct_rank": "best_rank"})
                 .sort_values(["source", "alias", "peptide"], ignore_index=True))
    best["binder"] = best.best_rank <= bg.RANK_BINDER
    best["strong_binder"] = best.best_rank <= bg.RANK_STRONG
    best.to_parquet(OUT / "peptide_scores.parquet", index=False)
    print(f"\nwrote {OUT/'peptide_scores.parquet'}  ({len(best):,} rows)")
    print(best.groupby("source").agg(
        peptides=("peptide", "size"), samples=("alias", "nunique"),
        binders=("binder", "sum"), binder_pct=("binder", lambda x: 100 * x.mean()),
        strong=("strong_binder", "sum")))


if __name__ == "__main__":
    main()
