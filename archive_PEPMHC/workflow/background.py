"""Percentile ranks for the checkpoint's scores.

The raw output is compressed into a band a few thousandths wide and sits above
0.5 for essentially every peptide, so the model's own 0.5 boundary calls
everything a binder and carries no information. What ordering the score does
carry is only usable *relative to other peptides on the same allele*.

So, as netMHCpan does, each allele gets a background of random natural peptides
drawn from the reviewed human proteome, and an observed peptide is reported as
a %rank against that background: %rank 1.0 means it scores above 99% of random
peptides on that allele. Binder calls then use the conventional cutoffs
(<=2% binder, <=0.5% strong binder) instead of the model's saturated 0.5.

This fixes the per-allele offset -- the dominant signal in the raw output -- but
it cannot manufacture discrimination the weights do not have; see
workflow/validate_model.py for what that is worth.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROTEOME = Path("/Users/vivianschu/repos/pmlb-neoantigens/workflow/"
                "mutation_reference/data/2025-04-01-reviewed-UP000005640.fas")
AA = set("ACDEFGHIKLMNPQRSTVWY")
RANK_BINDER, RANK_STRONG = 2.0, 0.5


def read_proteome(path=PROTEOME):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def sample_background(n=5000, lengths=(8, 9, 10, 11), weights=None, seed=0):
    """Random natural k-mers, length-matched to the observed repertoire."""
    rng = np.random.default_rng(seed)
    seqs = [s for s in read_proteome() if len(s) >= max(lengths)]
    if weights is None:
        weights = np.ones(len(lengths))
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    out = []
    while len(out) < n:
        s = seqs[rng.integers(len(seqs))]
        L = int(rng.choice(lengths, p=weights))
        if len(s) < L:
            continue
        i = int(rng.integers(0, len(s) - L + 1))
        kmer = s[i:i + L]
        if set(kmer) <= AA:
            out.append(kmer)
    return out


def build(predictor, alleles, background, cache=None):
    """allele -> sorted background scores (ascending)."""
    if cache is not None and Path(cache).exists():
        d = pd.read_parquet(cache)
        return {a: g.score.values for a, g in d.groupby("allele")}
    rows = []
    for i, a in enumerate(sorted(alleles), 1):
        s = predictor.predict(background, a)
        s = s[~np.isnan(s)]
        if s.size:
            rows.append(pd.DataFrame({"allele": a, "score": np.sort(s)}))
        print(f"  background {i}/{len(alleles)} {a}  n={s.size}", flush=True)
    d = pd.concat(rows, ignore_index=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(cache, index=False)
    return {a: g.score.values for a, g in d.groupby("allele")}


def pct_rank(scores, alleles, background):
    """%rank: percentage of background peptides on that allele scoring higher.
    Lower is a better binder. NaN where the allele has no background."""
    out = np.full(len(scores), np.nan, dtype=np.float32)
    s = np.asarray(scores, dtype=float)
    a = np.asarray(alleles, dtype=object)
    for allele, bg in background.items():
        m = (a == allele) & ~np.isnan(s)
        if not m.any():
            continue
        # fraction of background strictly above the observed score
        above = bg.size - np.searchsorted(bg, s[m], side="right")
        out[m] = 100.0 * above / bg.size
    return out
