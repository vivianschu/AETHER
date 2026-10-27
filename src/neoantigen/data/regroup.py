# src/neoantigen/data/regroup.py
"""
Rebuild the MHCAttnNet splits without train/test leakage.

Why
---
The shipped class-I split is partitioned by *row*, so the same peptide — and
often the same (peptide, allele) pair — appears on both sides:

    test peptides also in train            34,116 / 54,104   63.1%
    test (peptide, allele) pairs in train  24,543 / 68,643   35.8%
    exact duplicate rows within train      74,979

A model is therefore scored partly on pairs it memorised, and the resulting
AUROC is optimistic. This script produces two honest alternatives:

``peptide`` (default)
    Partition on unique peptides, so a peptide and all of its allele
    measurements land in exactly one split. Measures generalization to
    **unseen peptides** — the usual neoantigen-screening question.

``allele``
    Hold out whole alleles, keeping identical encoded MHC sequences together. Measures generalization to **unseen MHC alleles**,
    which is what a pan-allele model claims to do. (The shipped
    ``classI/new_mhc_split`` is already of this kind.)

Exact duplicate rows are dropped first in both cases. Rows that disagree on the
label for the same (peptide, allele) are reported and dropped. This binary
benchmark does not model assay context, so retaining both gives the same input
incompatible targets; exclusion does not establish which measurement is right.

Usage
-----
    python -m neoantigen.data.regroup \
        --in_dir data/mhcattnnet/classI \
        --out_dir data/mhcattnnet/classI_peptide_split
"""
from __future__ import annotations

import argparse
import os
import json
import hashlib
from neoantigen.train.dataloader import prepare_df, audit_splits

import numpy as np
import pandas as pd

SPLIT_FILES = {"train": "train.csv", "valid": "val.csv", "test": "test.csv"}


def load_pool(in_dir: str) -> pd.DataFrame:
    """Concatenate whatever splits exist in ``in_dir`` back into one pool."""
    frames = []
    for fname in SPLIT_FILES.values():
        path = os.path.join(in_dir, fname)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No train/val/test CSVs found in {in_dir}")
    return pd.concat(frames, ignore_index=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates, then drop label-contradicting (peptide, allele) pairs."""
    n0 = len(df)
    df = df.drop_duplicates()
    print(f"  dropped {n0 - len(df):,} exact duplicate rows")

    key = ["peptide", "mhc_allele"]
    nunique = df.groupby(key)["bind"].transform("nunique")
    conflicted = nunique > 1
    if conflicted.any():
        n_pairs = df.loc[conflicted, key].drop_duplicates().shape[0]
        print(f"  dropped {int(conflicted.sum()):,} rows across {n_pairs:,} "
              f"(peptide, allele) pairs with contradictory labels")
        df = df[~conflicted]

    # One row per (peptide, allele) — repeated identical measurements would
    # otherwise weight some pairs more heavily than others.
    n1 = len(df)
    df = df.drop_duplicates(subset=key)
    print(f"  collapsed {n1 - len(df):,} repeated (peptide, allele) measurements")
    return df.reset_index(drop=True)


def grouped_split(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    valid_frac: float,
    test_frac: float,
) -> dict[str, pd.DataFrame]:
    """
    Assign whole groups (all rows sharing ``group_col``) to one split each.

    Groups are shuffled and then walked in order, filling test and validation to
    their target *row* counts before the rest becomes train. Assigning by rows
    rather than by group count keeps the split sizes close to the requested
    fractions even though group sizes vary by orders of magnitude here.
    """
    if valid_frac <= 0 or test_frac <= 0 or valid_frac + test_frac >= 1:
        raise ValueError("valid/test fractions must be positive and sum to less than one")
    if df[group_col].nunique() < 3:
        raise ValueError("At least three groups are required")
    rng = np.random.default_rng(seed)
    sizes = df.groupby(group_col).size()
    groups = sizes.index.to_numpy()
    rng.shuffle(groups)

    n_total = len(df)
    targets = {"test": test_frac * n_total, "valid": valid_frac * n_total}
    assignment: dict[str, list] = {"train": [], "valid": [], "test": []}
    filled = {"test": 0, "valid": 0}

    for g in groups:
        if filled["test"] < targets["test"]:
            dest = "test"
        elif filled["valid"] < targets["valid"]:
            dest = "valid"
        else:
            dest = "train"
        assignment[dest].append(g)
        if dest in filled:
            filled[dest] += int(sizes[g])

    out = {}
    for split, gs in assignment.items():
        out[split] = df[df[group_col].isin(set(gs))].reset_index(drop=True)
    return out


def restore_assignments(pool, path):
    """Restore a versioned split and its row order, independent of RNG changes."""
    required = ["peptide", "mhc_allele", "split", "split_row"]
    assignments = pd.read_csv(path)
    if not set(required) <= set(assignments):
        raise ValueError(f"Assignments require {required}")
    if set(assignments["split"]) != set(SPLIT_FILES):
        raise ValueError("Assignments must contain train, valid and test")
    merged = pool.merge(assignments[required], on=["peptide", "mhc_allele"],
                        how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Assignments do not cover exactly the cleaned input corpus")
    result = {}
    for split in SPLIT_FILES:
        frame = merged.loc[merged["split"].eq(split)].sort_values("split_row")
        if not np.array_equal(frame.split_row.to_numpy(), np.arange(len(frame))):
            raise ValueError(f"{split}: split_row must be unique and contiguous from zero")
        result[split] = frame[pool.columns].reset_index(drop=True)
    return result


def report(splits: dict[str, pd.DataFrame], group_col: str) -> None:
    train = splits["train"]
    print("\n  split summary")
    for name in ("train", "valid", "test"):
        d = splits[name]
        print(f"    {name:5s} {len(d):>7,} rows | positive {d['bind'].mean():.3f} "
              f"| {d['peptide'].nunique():>6,} peptides "
              f"| {d['mhc_allele'].nunique():>3} alleles")

    # The check the original split fails
    tr_pairs = set(zip(train["peptide"], train["mhc_allele"]))
    te = splits["test"]
    te_pairs = set(zip(te["peptide"], te["mhc_allele"]))
    overlap_pairs = len(te_pairs & tr_pairs)
    overlap_group = len(set(te[group_col]) & set(train[group_col]))
    print(f"\n  leakage check")
    print(f"    test (peptide, allele) pairs also in train : {overlap_pairs:,}")
    print(f"    shared {group_col} between train and test   : {overlap_group:,}")
    if overlap_pairs or overlap_group:
        print("    WARNING: leakage present — this should be zero")
    else:
        print("    clean")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", required=True, help="Folder holding the original splits")
    p.add_argument("--out_dir", required=True, help="Where to write the new splits")
    p.add_argument("--group_by", choices=["peptide", "allele"], default="peptide",
                   help="peptide = unseen-peptide split; allele = unseen-allele split")
    p.add_argument("--assignments_file", default=None,
                   help="Restore saved peptide/allele split assignments and row order instead of resampling")
    p.add_argument("--valid_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    group_col = "peptide" if args.group_by == "peptide" else "mhc_amino_acid"

    print(f"Pooling splits from {args.in_dir}")
    pool = load_pool(args.in_dir)
    print(f"  {len(pool):,} rows")

    # Apply precisely the same sequence normalization as training before grouping.
    pool = prepare_df(pool).rename(columns={"epitope": "peptide",
        "mhc_sequence": "mhc_amino_acid", "y": "bind"})
    print("Deduplicating")
    pool = deduplicate(pool)
    print(f"  {len(pool):,} rows remain")

    if args.group_by == "allele" and pool.groupby("mhc_allele")["mhc_amino_acid"].nunique().max() > 1:
        raise ValueError("An allele maps to multiple sequences; resolve mappings before allele splitting")
    print(f"Splitting, grouped by {group_col}")
    splits = (restore_assignments(pool, args.assignments_file) if args.assignments_file else
              grouped_split(pool, group_col, args.seed, args.valid_frac, args.test_frac))
    report(splits, group_col)

    audit = audit_splits({k: prepare_df(v) for k,v in splits.items()}, args.group_by)
    os.makedirs(args.out_dir, exist_ok=False)
    for split, fname in SPLIT_FILES.items():
        path = os.path.join(args.out_dir, fname)
        splits[split].to_csv(path, index=False)
        print(f"\n  wrote {path}")

    manifest = dict(config=vars(args), audit=audit,
        assignments_sha256=hashlib.sha256(open(args.assignments_file, "rb").read()).hexdigest() if args.assignments_file else None,
        files={
        fname: hashlib.sha256(open(os.path.join(args.out_dir, fname), "rb").read()).hexdigest()
        for fname in SPLIT_FILES.values()})
    with open(os.path.join(args.out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
