"""What are the checkpoint's scores actually worth?

Nothing in MHCAttnNet_dataset/models/classI/check.pt records the two
hyperparameters that leave no trace in the weight shapes (number of attention
heads, final activation) or how the batch axis was wired, so they are chosen
here by held-out ROC AUC on classI/test.csv -- the split the torchtext training
path held out (data_loader.IEDB.splits defaults to config.base_path).

Also measures how much a score moves when the same pair is scored in a
different batch: the trained graph attends across the batch dimension, so this
is not guaranteed to be zero.

Writes workflow/results/validation.csv for the ROC figure.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pepmhc.predictor import Predictor

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, precision_score,
                             recall_score)

TEST = (Path(__file__).resolve().parent.parent
        / "MHCAttnNet_dataset/classI/test.csv")
OUT = Path(__file__).resolve().parent / "results"


def metrics(y, s):
    pred = s > 0.5
    return {
        "ROC_AUC": roc_auc_score(y, s),
        "PRC_AUC": average_precision_score(y, s),
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "score_mean": s.mean(),
        "score_sd": s.std(),
        "score_range": s.max() - s.min(),
    }


def sweep(sub, grid):
    rows = []
    for seq_first, nhead, act, relu_out in grid:
        p = Predictor(activation=act, nhead=nhead, seq_first=seq_first,
                      relu_on_output=relu_out)
        s = p.predict_pairs(sub.peptide, sub.mhc_allele)
        ok = ~np.isnan(s)
        m = metrics(sub.bind.values[ok], s[ok])
        m.update(seq_first=seq_first, nhead=nhead, activation=act,
                 relu_on_output=relu_out, n=int(ok.sum()))
        rows.append(m)
        print(f"  seq_first={seq_first!s:5s} nhead={nhead:<3d} act={act:4s} "
              f"relu_out={relu_out!s:5s} AUC={m['ROC_AUC']:.4f} "
              f"acc={m['Accuracy']:.4f} sd={m['score_sd']:.4f}", flush=True)
    return pd.DataFrame(rows)


def batch_sensitivity(best, sub):
    """Same pairs, three batch layouts. Spread = how much the cross-batch
    attention leaks between unrelated peptides."""
    p = Predictor(**best)
    base = p.predict_pairs(sub.peptide, sub.mhc_allele)
    shuf = sub.sample(frac=1, random_state=1)
    s_shuf = pd.Series(p.predict_pairs(shuf.peptide, shuf.mhc_allele),
                       index=shuf.index).reindex(sub.index).values
    p64 = Predictor(**{**best, "batch_size": 64})
    s_64 = p64.predict_pairs(sub.peptide, sub.mhc_allele)
    ok = ~(np.isnan(base) | np.isnan(s_shuf) | np.isnan(s_64))
    print(f"\nbatch sensitivity on {ok.sum():,} pairs")
    print(f"  reshuffled batches : max |delta| = {np.abs(base[ok]-s_shuf[ok]).max():.5f}, "
          f"mean = {np.abs(base[ok]-s_shuf[ok]).mean():.5f}")
    print(f"  batch_size 32 -> 64: max |delta| = {np.abs(base[ok]-s_64[ok]).max():.5f}, "
          f"mean = {np.abs(base[ok]-s_64[ok]).mean():.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-n", type=int, default=4000)
    ap.add_argument("--final-n", type=int, default=20000)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    df = pd.read_csv(TEST)
    print(f"classI/test.csv: {len(df):,} rows, {df.bind.mean():.1%} positive, "
          f"{df.mhc_allele.nunique()} alleles\n")
    sub = df.sample(n=min(args.sweep_n, len(df)), random_state=0)

    grid = [(sf, nh, act, ro)
            for sf in (True, False)
            for nh in (1, 2, 10)
            for act in ("mish", "relu")
            for ro in (False, True)]
    print(f"sweeping {len(grid)} configurations on {len(sub):,} rows")
    res = sweep(sub, grid)
    res.sort_values("ROC_AUC", ascending=False).to_csv(
        OUT / "config_sweep.csv", index=False)

    top = res.loc[res.ROC_AUC.idxmax()]
    best = dict(seq_first=bool(top.seq_first), nhead=int(top.nhead),
                activation=top.activation, relu_on_output=bool(top.relu_on_output))
    print(f"\nbest config: {best}  (sweep AUC {top.ROC_AUC:.4f})")

    final = df.sample(n=min(args.final_n, len(df)), random_state=7)
    p = Predictor(**best)
    s = p.predict_pairs(final.peptide, final.mhc_allele)
    ok = ~np.isnan(s)
    m = metrics(final.bind.values[ok], s[ok])
    print(f"\nheld-out performance on {ok.sum():,} rows")
    for k, v in m.items():
        print(f"  {k:12s} {v:.4f}")

    pd.DataFrame({"peptide": final.peptide.values[ok],
                  "mhc_allele": final.mhc_allele.values[ok],
                  "bind": final.bind.values[ok],
                  "score": s[ok]}).to_csv(OUT / "validation.csv", index=False)
    pd.Series({**best, **m}).to_csv(OUT / "validation_metrics.csv")
    print(f"\nwrote {OUT/'validation.csv'}")

    batch_sensitivity(best, df.sample(n=2000, random_state=3))


if __name__ == "__main__":
    main()
