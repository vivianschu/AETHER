# src/neoantigen/train/tcr_dataloader.py
"""
Data loading for phase 2: does a given TCR recognize a peptide-MHC complex?

This is a different objective from phase 1, not a richer version of it. The
label is receptor recognition; the rows come from tetramer and activation
assays, not binding assays. Phase 1's loader is left untouched.

Two things about this data decide whether the result means anything.

**Split by epitope, never by row.** ``data/cedar_phase2/build/cedar_tcr.csv``
holds ~52k rows over ~1.1k distinct epitopes, and the ten commonest account
for most of the rows. Split at random and the same epitope lands on both
sides: the model memorizes which epitopes are immunogenic, reports a fine
AUROC, and collapses on an epitope it has not seen. ``make_tcr_splits``
partitions on the epitope, so every test epitope is new.

**Negatives have to be constructed, and how matters.** Curated data is ~94%
positive -- nobody publishes receptors that failed to bind. The tempting fix,
pairing receptors with random peptides, teaches the model to tell a real
assayed epitope from a random string, which it can do without ever looking at
the receptor. ``sample_negatives`` instead shuffles receptors against *other
epitopes that are themselves real positives elsewhere in the same split*, so
the epitope marginal is identical in both classes and the only thing that
distinguishes a positive from a negative is which receptor is attached.

Shuffled negatives are an assumption, not an observation: a receptor paired
with a different epitope is *probably* a non-binder, and at a ~1% true
cross-reactivity rate the label noise is tolerable. The ``real_negative``
column marks the few hundred measured negatives, so they can be weighted
differently or held out as a cleaner test.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .dataloader import clean_sequences

CDR3_ALIASES = ["cdr3b", "cdr3_b", "cdr3beta", "chain2_cdr3_seq", "tcell_sequence"]


class TCRDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.epitopes = self.df["epitope"].tolist()
        self.mhc_seqs = self.df["mhc_sequence"].tolist()
        self.cdr3b = self.df["cdr3b"].tolist()
        self.y = self.df["y"].astype(np.float32).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "epitope": self.epitopes[idx],
            "mhc_sequence": self.mhc_seqs[idx],
            "cdr3b": self.cdr3b[idx],
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def tcr_collate_fn(batch):
    """Returns (peptides, mhcs, cdr3b, y) -- one more field than phase 1."""
    return (
        [b["epitope"] for b in batch],
        [b["mhc_sequence"] for b in batch],
        [b["cdr3b"] for b in batch],
        torch.stack([b["y"] for b in batch], dim=0),
    )


# ── Preparation ───────────────────────────────────────────────────────────────

def prepare_tcr_df(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize columns and clean all three sequences."""
    rename = {}
    for canonical, aliases in (
        ("epitope", ["epitope", "peptide"]),
        ("mhc_sequence", ["mhc_sequence", "mhc_amino_acid"]),
        ("y", ["y", "bind", "label"]),
        ("cdr3b", CDR3_ALIASES),
    ):
        hit = next((a for a in aliases if a in df.columns), None)
        if hit is None:
            raise ValueError(
                f"No column found for {canonical!r}. Tried {aliases}; "
                f"the file has {list(df.columns)}."
            )
        rename[hit] = canonical

    keep = list(rename) + [c for c in ("mhc_allele", "neoantigen", "viral")
                           if c in df.columns]
    out = df[keep].rename(columns=rename).copy()
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["epitope", "mhc_sequence", "cdr3b", "y"])

    out = clean_sequences(out)                     # epitope + mhc_sequence
    out = clean_sequences(out.rename(columns={"cdr3b": "epitope",
                                              "epitope": "_pep"})) \
        .rename(columns={"epitope": "cdr3b", "_pep": "epitope"})
    out = out[out["cdr3b"].str.len() >= 5]         # a CDR3 shorter than this is junk
    return out.reset_index(drop=True)


def make_tcr_splits(
    df: pd.DataFrame,
    seed: int = 1337,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    mode: str = "epitope",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition on the epitope, so no test epitope was seen in training.

    Epitopes are shuffled and assigned whole. Because a handful of epitopes
    carry most of the rows, the resulting row counts will not match the
    requested fractions closely -- that is the honest cost of the split, not a
    bug. The row counts are printed so the imbalance is visible.
    """
    rng = np.random.default_rng(seed)

    if mode == "random":
        # Diagnostic only: epitopes appear on both sides, so a model can score
        # well by recognizing the epitope. Use it to separate "the head cannot
        # learn" from "the head cannot generalize to unseen epitopes" -- never
        # to report a result.
        shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        a, b = int(round(train_frac * n)), int(round((train_frac + valid_frac) * n))
        return shuffled.iloc[:a], shuffled.iloc[a:b], shuffled.iloc[b:]

    # Permute indices rather than the array itself: these are Python strings in
    # an object array, which numpy declines to shuffle in place.
    uniq = df["epitope"].unique()
    epitopes = uniq[rng.permutation(len(uniq))]

    n = len(epitopes)
    n_train = int(round(train_frac * n))
    n_valid = int(round(valid_frac * n))
    groups = {
        "train": set(epitopes[:n_train]),
        "valid": set(epitopes[n_train : n_train + n_valid]),
        "test": set(epitopes[n_train + n_valid :]),
    }
    return tuple(
        df[df["epitope"].isin(groups[k])].reset_index(drop=True)
        for k in ("train", "valid", "test")
    )


def cap_positives_per_epitope(df: pd.DataFrame, cap: int, seed: int = 1337) -> pd.DataFrame:
    """
    Limit how many positives any one epitope contributes, and drop epitopes
    that have none.

    Two reasons, both about keeping the label balanced within each epitope.

    A negative needs a CDR3b observed against a *different* epitope, and each
    (epitope, receptor) pair is used once. KLGGALQAK has 13,466 positives in
    the train split but only 24,318 foreign receptors exist, so it tops out at
    ratio 1.8 while every other epitope reaches 5 -- leaving it at a 35%
    positive rate against their 17%, which is the leak again in miniature.
    A cap of C needs C*(ratio+1) <= distinct receptors to stay balanced.

    Epitopes with no positives at all (11 here, carrying only measured
    negatives) get no generated negatives either, so they sit at a 0% positive
    rate and leak the same way in the other direction.
    """
    has_pos = df.groupby("epitope")["y"].transform("max") > 0
    dropped_eps = df.loc[~has_pos, "epitope"].nunique()
    df = df[has_pos]

    if cap and cap > 0:
        # cumcount over a shuffled frame, not groupby.apply: on pandas >= 2.2
        # apply drops the grouping column from the group it passes, which
        # silently returns rows with a NaN epitope.
        pos = df[df["y"] > 0].sample(frac=1, random_state=seed)
        keep = pos[pos.groupby("epitope").cumcount() < cap]
        df = pd.concat([keep, df[df["y"] <= 0]], ignore_index=True)
    if dropped_eps:
        print(f"  dropped {dropped_eps} epitopes with no positives", flush=True)
    return df.reset_index(drop=True)


def sample_negatives(
    df: pd.DataFrame,
    ratio: float = 5.0,
    seed: int = 1337,
) -> pd.DataFrame:
    """
    Add shuffled negatives *within this split*, preserving the epitope marginal.

    For each synthetic negative we keep a real (epitope, MHC) pair and attach a
    CDR3b observed against a different epitope in the same split. Sampling
    inside the split matters: draw from the whole table and a receptor from the
    test set would appear in training negatives.

    Pairs that exist as observed positives are never emitted as negatives.
    """
    if ratio <= 0:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    pos = df[df["y"] > 0]
    if pos.empty:
        return df.reset_index(drop=True)

    observed = set(zip(df["epitope"], df["cdr3b"]))
    receptors = pos[["cdr3b", "epitope"]].to_numpy()

    # Negatives are generated PER POSITIVE ROW, against that row's own epitope.
    #
    # Sampling epitopes uniformly instead would give every epitope a similar
    # negative count while the positives stay concentrated -- KLGGALQAK has
    # 13,639 positives, hundreds of epitopes have one. The positive rate then
    # ranges from 0.99 down to 0.00 across epitopes, and the epitope's base
    # rate alone scores AUROC 0.976 on train: the model memorizes which
    # epitopes are immunogenic and never reads the receptor. Because the
    # splits are epitope-disjoint, none of that transfers, so training loss
    # collapses immediately and validation never improves.
    #
    # One negative count per positive keeps the positive rate identical for
    # every epitope, so epitope identity carries no information about the
    # label and the only thing separating the classes is which receptor is
    # attached -- which is the task.
    # A positive that cannot be given its full complement of negatives is
    # dropped along with its partial ones, so the ratio holds exactly for
    # every epitope and every split.
    #
    # Without this, the smaller splits silently drift: valid holds only 1,488
    # distinct receptors against train's 37,784, so the sampler runs dry and
    # valid lands at a 27% positive rate with per-epitope rates up to 56%
    # while train sits at a uniform 17%. That is the same base-rate leak in
    # the evaluation set, and it also means val and test scores are measured
    # on a different class balance than the model was trained for.
    rows, kept_pos = [], []
    n_neg = int(ratio)
    for pos_i, (ep, mhc) in enumerate(zip(pos["epitope"].to_numpy(),
                                          pos["mhc_sequence"].to_numpy())):
        made, attempts, staged = 0, 0, []
        while made < n_neg and attempts < n_neg * 20:
            attempts += 1
            cdr, from_ep = receptors[rng.integers(len(receptors))]
            if from_ep == ep or (ep, cdr) in observed:
                continue
            observed.add((ep, cdr))
            staged.append({"epitope": ep, "mhc_sequence": mhc,
                           "cdr3b": cdr, "y": 0.0})
            made += 1
        if made == n_neg:
            rows.extend(staged)
            kept_pos.append(pos_i)
        else:
            for r in staged:            # release the pairs we are not using
                observed.discard((ep, r["cdr3b"]))

    if len(kept_pos) < len(pos):
        print(f"  dropped {len(pos) - len(kept_pos):,} of {len(pos):,} positives "
              f"that could not be matched with {n_neg} distinct negatives",
              flush=True)
    df = pd.concat([pos.iloc[kept_pos], df[df["y"] <= 0]], ignore_index=True)

    neg = pd.DataFrame(rows)
    out = df.copy()
    out["real_negative"] = (out["y"] == 0).astype(int)
    if not neg.empty:
        neg["real_negative"] = 0
        out = pd.concat([out, neg], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Config and entry point ────────────────────────────────────────────────────

@dataclass
class TCRDataConfig:
    tcr_csv: str
    seed: int = 1337
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    neg_ratio: float = 5.0
    split_mode: str = "epitope"   # "random" is a diagnostic, not a result
    max_pos_per_epitope: int = 1000  # keeps the per-epitope balance reachable
    drop_viral: bool = False       # keep viral rows: they are 95% of the corpus
    neoantigen_only: bool = False  # cancer-only evaluation set
    max_train_rows: int = 0
    max_eval_rows: int = 0
    rank: int = 0
    world_size: int = 1


def get_tcr_dataloaders(cfg: TCRDataConfig):
    """Returns (train_loader, valid_loader, test_loader, splits)."""
    print(f"Loading TCR dataset from: {cfg.tcr_csv}", flush=True)
    df = prepare_tcr_df(pd.read_csv(cfg.tcr_csv))

    if cfg.drop_viral and "viral" in df.columns:
        before = len(df)
        df = df[df["viral"] == 0].reset_index(drop=True)
        print(f"  dropped {before - len(df):,} viral-antigen rows", flush=True)
    if cfg.neoantigen_only and "neoantigen" in df.columns:
        before = len(df)
        df = df[df["neoantigen"] == 1].reset_index(drop=True)
        print(f"  kept neoantigen rows only ({len(df):,} of {before:,})", flush=True)

    print(f"  {len(df):,} rows | {df['epitope'].nunique():,} distinct epitopes | "
          f"{df['cdr3b'].nunique():,} distinct CDR3b | "
          f"observed positive {df['y'].mean():.3f}", flush=True)

    train_df, valid_df, test_df = make_tcr_splits(df, seed=cfg.seed,
                                                  mode=cfg.split_mode)

    # Negatives are drawn per split, after the epitope partition.
    splits = {}
    for name, part, seed_off in (("train", train_df, 0),
                                 ("valid", valid_df, 1),
                                 ("test", test_df, 2)):
        part = cap_positives_per_epitope(part, cfg.max_pos_per_epitope,
                                         seed=cfg.seed + seed_off)
        part = sample_negatives(part, ratio=cfg.neg_ratio, seed=cfg.seed + seed_off)
        n_cap = cfg.max_train_rows if name == "train" else cfg.max_eval_rows
        if n_cap and len(part) > n_cap:
            part = part.sample(n=n_cap, random_state=cfg.seed).reset_index(drop=True)
        splits[name] = part
        print(f"  {name:5s} {len(part):>7,} rows | "
              f"{part['epitope'].nunique():>5,} epitopes | "
              f"positive {part['y'].mean():.3f} | "
              f"real negatives {int(part.get('real_negative', pd.Series(dtype=int)).sum()):,}",
              flush=True)

    overlap = set(splits["train"]["epitope"]) & set(splits["test"]["epitope"])
    if cfg.split_mode == "random":
        print(f"  WARNING: --tcr_split random -- {len(overlap)} epitopes appear in "
              f"both train and test. Diagnostic only; not a reportable number.",
              flush=True)
    else:
        assert not overlap, f"epitope leaked across splits: {sorted(overlap)[:5]}"

    datasets = {k: TCRDataset(v) for k, v in splits.items()}

    train_sampler = None
    if cfg.world_size > 1:
        from torch.utils.data import DistributedSampler

        train_sampler = DistributedSampler(
            datasets["train"], num_replicas=cfg.world_size, rank=cfg.rank,
            shuffle=True, drop_last=True,
        )

    common = dict(
        collate_fn=tcr_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(
        datasets["train"], batch_size=cfg.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        drop_last=False, **common,
    )
    valid_loader = DataLoader(
        datasets["valid"], batch_size=cfg.batch_size * 2, shuffle=False, **common
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=cfg.batch_size * 2, shuffle=False, **common
    )
    return train_loader, valid_loader, test_loader, splits
