"""Ranking metrics shared by training, evaluation and benchmarking.

PPV@100 uses min(100, n_scored) candidates. Ties crossing the cutoff receive
fractional credit equal to their expected positives under random tie breaking.
Within-allele AUROC is support weighted over alleles with both labels; excluded
rows and eligible allele counts are reported explicitly.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(y, scores, alleles=None, min_allele_rows=1):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.shape != scores.shape or y.ndim != 1:
        raise ValueError("Labels and scores must be aligned one-dimensional arrays")
    ok = np.isfinite(scores)
    a = np.asarray(alleles)[ok] if alleles is not None else None
    y, scores = y[ok], scores[ok]
    n = len(y)
    out = dict(n_total=len(ok), n_scored=n,
               coverage=float(ok.mean()) if len(ok) else 0.0)
    if not n:
        return out
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Ranking metrics require binary labels")
    k = min(100, n)
    cutoff = np.sort(scores)[-k]
    above, tied = scores > cutoff, scores == cutoff
    out.update(positive_rate=float(y.mean()), ppv_at_100_n=k,
               ppv_at_100=float((y[above].sum() +
                   (k - above.sum()) * y[tied].mean()) / k),
               auprc=float(average_precision_score(y, scores)) if y.any() else 0.0)
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, scores))
    if a is not None:
        weighted, support, count = 0.0, 0, 0
        for allele in pd.unique(a):
            mask = a == allele
            if mask.sum() < min_allele_rows or len(np.unique(y[mask])) < 2:
                continue
            weighted += mask.sum() * roc_auc_score(y[mask], scores[mask])
            support += int(mask.sum())
            count += 1
        out.update(within_allele_n=support, within_allele_n_alleles=count,
                   within_allele_excluded_n=n-support)
        if support:
            out["within_allele_auroc"] = float(weighted / support)
    return out


def per_allele_metrics(y, scores, alleles):
    y, scores, alleles = np.asarray(y), np.asarray(scores), np.asarray(alleles)
    return pd.DataFrame([
        dict(mhc_allele=a, **ranking_metrics(y[alleles == a], scores[alleles == a]))
        for a in sorted(pd.unique(alleles))
    ])
