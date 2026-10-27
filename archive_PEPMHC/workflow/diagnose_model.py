"""How much of the score comes from the peptide, and how much from the allele?

A usable binder should move a lot when the peptide changes while the allele is
held fixed. This splits the variance of the checkpoint's output between the two
inputs over the held-out split.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pepmhc.predictor import Predictor

OUT = Path(__file__).resolve().parent / "results"


def main(n_pep=200, n_allele=20):
    cfg = {}
    mpath = OUT / "validation_metrics.csv"
    if mpath.exists():
        s = pd.read_csv(mpath, index_col=0).iloc[:, 0]
        cfg = dict(seq_first=s["seq_first"] == "True", nhead=int(s["nhead"]),
                   activation=s["activation"],
                   relu_on_output=s["relu_on_output"] == "True")
    print("config:", cfg or "defaults")
    p = Predictor(**cfg)

    df = pd.read_csv(ROOT / "MHCAttnNet_dataset/classI/test.csv")
    peps = df.peptide.drop_duplicates().sample(n_pep, random_state=0).tolist()
    alleles = [a for a in p.supported_alleles()][:n_allele]

    grid = pd.DataFrame([(pep, a) for a in alleles for pep in peps],
                        columns=["peptide", "allele"])
    grid["score"] = p.predict_pairs(grid.peptide, grid.allele)
    grid = grid.dropna(subset=["score"])
    grid.to_csv(OUT / "diagnostic_grid.csv", index=False)

    total = grid.score.var()
    within_allele = grid.groupby("allele").score.var().mean()   # peptide-driven
    within_pep = grid.groupby("peptide").score.var().mean()     # allele-driven
    print(f"\n{len(peps)} peptides x {len(alleles)} alleles = {len(grid):,} scores")
    print(f"  overall score range      {grid.score.min():.4f} .. {grid.score.max():.4f}")
    print(f"  total variance           {total:.3e}")
    print(f"  mean variance across peptides, allele fixed  {within_allele:.3e}"
          f"  ({within_allele/total:>6.1%} of total)")
    print(f"  mean variance across alleles, peptide fixed  {within_pep:.3e}"
          f"  ({within_pep/total:>6.1%} of total)")

    spread = grid.groupby("peptide").score.agg(lambda s: s.max() - s.min())
    print(f"\n  per-peptide spread across alleles: median {spread.median():.5f}, "
          f"max {spread.max():.5f}")
    spread_a = grid.groupby("allele").score.agg(lambda s: s.max() - s.min())
    print(f"  per-allele spread across peptides: median {spread_a.median():.5f}, "
          f"max {spread_a.max():.5f}")
    print(f"\nwrote {OUT/'diagnostic_grid.csv'}")


if __name__ == "__main__":
    main()
