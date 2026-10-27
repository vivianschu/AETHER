# src/neoantigen/train/dataloader.py
"""
Data loading for peptide-MHC binding.

Two layouts are supported:

**Pre-split directory** (``--data_dir``) — a folder holding ``train.csv``,
``val.csv`` and ``test.csv``, as shipped with the MHCAttnNet dataset. This is
preferred when the files are already deduplicated and group-disjoint. The
original shipped row split must be regrouped before training.

**Single CSV** (``--csv_url``, a local path or a URL) — split here by grouping
on peptide by default, so a peptide never straddles two splits.

Column names are resolved by alias, so both the MHCAttnNet schema
(``peptide`` / ``mhc_amino_acid`` / ``bind``) and the AETHER schema
(``epitope`` / ``mhc_sequence`` / ``mhc_binding``) load without edits.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Accepted spellings for each required column, in priority order.
COLUMN_ALIASES = {
    "epitope": ["epitope", "peptide", "pep", "peptide_seq"],
    "mhc_sequence": ["mhc_sequence", "mhc_amino_acid", "mhc_seq", "mhc"],
    "y": ["y", "bind", "mhc_binding", "label", "binder", "target"],
}
# Optional; kept when present so results can be broken down per allele.
ALLELE_ALIASES = ["mhc_allele", "allele", "hla"]


class NeoantigenDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.epitopes = self.df["epitope"].tolist()
        self.mhc_seqs = self.df["mhc_sequence"].tolist()
        self.y = self.df["y"].astype(np.float32).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "epitope": self.epitopes[idx],
            "mhc_sequence": self.mhc_seqs[idx],
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def collate_fn(batch):
    peptides = [b["epitope"] for b in batch]
    mhcs = [b["mhc_sequence"] for b in batch]
    y = torch.stack([b["y"] for b in batch], dim=0)
    return peptides, mhcs, y


# ── Column resolution and cleanup ─────────────────────────────────────────────

def _resolve_column(df: pd.DataFrame, canonical: str) -> str:
    for alias in COLUMN_ALIASES[canonical]:
        if alias in df.columns:
            return alias
    raise ValueError(
        f"No column found for {canonical!r}. Tried {COLUMN_ALIASES[canonical]}; "
        f"the file has {list(df.columns)}."
    )


def clean_sequences(out: pd.DataFrame) -> pd.DataFrame:
    """Uppercase, strip whitespace, and drop rows left with an empty sequence."""
    for col in ("epitope", "mhc_sequence"):
        out[col] = out[col].astype(str).str.upper().str.strip().str.replace(" ", "")
    # Non-standard residue codes (B, J, O, U, Z, ...) have no meaningful ESM2
    # embedding; map them to the unknown token X rather than letting the
    # tokenizer handle them inconsistently.
    valid = set("ACDEFGHIKLMNPQRSTVWYX")
    for col in ("epitope", "mhc_sequence"):
        out[col] = out[col].map(lambda s: "".join(c if c in valid else "X" for c in s))
    return out[(out["epitope"].str.len() > 0) & (out["mhc_sequence"].str.len() > 0)]


def resolve_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename just the peptide and MHC columns to the canonical names and clean
    them. Used at inference time, where a label column need not be present.
    """
    rename = {_resolve_column(df, c): c for c in ("epitope", "mhc_sequence")}
    return clean_sequences(df.rename(columns=rename).copy()).reset_index(drop=True)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical schema, clean sequences, drop bad rows."""
    if "label_conflict" in df:
        conflicted = df["label_conflict"].astype(str).str.lower().isin(["true", "1", "1.0"])
        if conflicted.any():
            print(f"  dropping {int(conflicted.sum())} rows explicitly flagged label_conflict", flush=True)
            df = df.loc[~conflicted]
    rename = {_resolve_column(df, c): c for c in COLUMN_ALIASES}
    allele_col = next((a for a in ALLELE_ALIASES if a in df.columns), None)

    keep = list(rename) + ([allele_col] if allele_col else [])
    out = df[keep].rename(columns=rename).copy()
    if allele_col and allele_col != "mhc_allele":
        out = out.rename(columns={allele_col: "mhc_allele"})

    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["epitope", "mhc_sequence", "y"])
    return clean_sequences(out).reset_index(drop=True)


def _read_csv(path: str) -> pd.DataFrame:
    """Read a local path or a URL. HF-gated URLs pick up HF_TOKEN if set."""
    if path.startswith(("http://", "https://")):
        token = os.environ.get("HF_TOKEN")
        if token and "huggingface.co" in path:
            import urllib.request

            req = urllib.request.Request(
                path, headers={"Authorization": f"Bearer {token}"}
            )
            with urllib.request.urlopen(req) as fh:
                return pd.read_csv(fh)
    return pd.read_csv(path)


# ── Splits ────────────────────────────────────────────────────────────────────

def make_splits(
    df: pd.DataFrame,
    seed: int = 1337,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    split_group: str = "peptide",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Remove repeated/conflicting pairs and split whole peptides (default),
    alleles, or pairs. Pre-split inputs are audited separately.
    """
    if not np.isclose(train_frac + valid_frac + test_frac, 1.0):
        raise ValueError("train_frac + valid_frac + test_frac must sum to 1.0")

    if split_group == "allele" and df.groupby("mhc_allele")["mhc_sequence"].nunique().max() > 1:
        raise ValueError("An allele maps to multiple MHC sequences; resolve mappings first")
    group_cols = {"peptide": ["epitope"], "allele": ["mhc_sequence"],
                  "pair": ["epitope", "mhc_sequence"]}[split_group]
    key = ["epitope", "mhc_allele" if "mhc_allele" in df else "mhc_sequence"]
    conflicts = df.groupby(key)["y"].transform("nunique") > 1
    df = df.loc[~conflicts].drop_duplicates(key).reset_index(drop=True)
    groups = df[group_cols].drop_duplicates().sample(frac=1, random_state=seed)
    n = len(groups)
    n_train, n_valid = int(round(train_frac*n)), int(round(valid_frac*n))
    parts = [groups.iloc[:n_train], groups.iloc[n_train:n_train+n_valid],
             groups.iloc[n_train+n_valid:]]
    return tuple(df.merge(part, on=group_cols) for part in parts)


def audit_splits(splits, split_group="peptide"):
    """Fail closed on leakage after canonical sequence cleaning, before subsampling."""
    import itertools
    report = {"split_group": split_group, "splits": {}, "overlaps": {}}
    cols = {"peptide": ["epitope"], "allele": ["mhc_allele"],
            "pair": ["epitope", "mhc_sequence"]}[split_group]
    for name, df in splits.items():
        if df.empty:
            raise ValueError(f"{name} split is empty")
        duplicates = int(df.duplicated(["epitope", "mhc_allele" if "mhc_allele" in df else "mhc_sequence"]).sum())
        if duplicates:
            raise ValueError(f"{name} has {duplicates} repeated peptide-MHC pairs; regroup first")
        report["splits"][name] = dict(n=len(df), positive_rate=float(df.y.mean()),
            peptides=int(df.epitope.nunique()),
            alleles=int(df.mhc_allele.nunique()) if "mhc_allele" in df else None)
    for left, right in itertools.combinations(splits, 2):
        a, b = splits[left], splits[right]
        overlap = len(set(a[cols].itertuples(index=False, name=None)) &
                      set(b[cols].itertuples(index=False, name=None)))
        report["overlaps"][f"{left}_{right}"] = overlap
        if split_group == "allele":
            sequence_overlap = len(set(a.mhc_sequence) & set(b.mhc_sequence))
            report["overlaps"][f"{left}_{right}_mhc_sequence"] = sequence_overlap
            if sequence_overlap:
                raise ValueError(f"{left}/{right} share {sequence_overlap} MHC sequences despite held-out allele names; regroup by MHC sequence")
        if overlap:
            raise ValueError(f"{left}/{right} share {overlap} {split_group} groups; regroup first")
    return report


@dataclass
class DataConfig:
    csv_url: Optional[str] = None
    data_dir: Optional[str] = None     # folder with train.csv / val.csv / test.csv
    seed: int = 1337
    split_group: str = "peptide"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    max_train_rows: int = 0            # >0 subsamples the train split (quick runs)
    max_eval_rows: int = 0             # >0 subsamples val/test (quick runs)
    rank: int = 0                      # DDP rank  (0 for single-GPU)
    world_size: int = 1                # DDP world size (1 for single-GPU)


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n and len(df) > n:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


def load_splits(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, valid_df, test_df) in the canonical schema."""
    if cfg.data_dir:
        def _load(fname: str, required: bool = True):
            path = os.path.join(cfg.data_dir, fname)
            if not os.path.exists(path):
                if required:
                    raise FileNotFoundError(
                        f"Expected {path} in --data_dir {cfg.data_dir}"
                    )
                return None
            return prepare_df(_read_csv(path))

        train_df = _load("train.csv")
        test_df = _load("test.csv")
        valid_df = _load("val.csv", required=False)

        if valid_df is None:
            # Some splits (e.g. classI/new_mhc_split) ship train/test only. Hold
            # out a slice of train by (epitope, mhc) pair, so a pair measured
            # twice cannot land on both sides of the split.
            valid_frac = 0.1
            train_df, valid_df, _ = make_splits(
                train_df,
                seed=cfg.seed,
                split_group=cfg.split_group,
                train_frac=1.0 - valid_frac,
                valid_frac=valid_frac,
                test_frac=0.0,
            )
            print(
                f"  no val.csv in {cfg.data_dir}; held out "
                f"{valid_frac:.0%} of train for validation",
                flush=True,
            )
    elif cfg.csv_url:
        full = prepare_df(_read_csv(cfg.csv_url))
        train_df, valid_df, test_df = make_splits(full, seed=cfg.seed, split_group=cfg.split_group)
    else:
        raise ValueError("Set either --data_dir or --csv_url.")

    audit_splits(dict(train=train_df, valid=valid_df, test=test_df), cfg.split_group)
    train_df = _subsample(train_df, cfg.max_train_rows, cfg.seed)
    valid_df = _subsample(valid_df, cfg.max_eval_rows, cfg.seed)
    test_df = _subsample(test_df, cfg.max_eval_rows, cfg.seed)
    return train_df, valid_df, test_df


def get_dataloaders(cfg: DataConfig):
    """
    Returns (train_loader, valid_loader, test_loader, splits) where ``splits`` is
    the dict of DataFrames, so callers can inspect label balance and collect the
    unique MHC sequences to precompute.
    """
    source = cfg.data_dir or cfg.csv_url
    print(f"Loading dataset from: {source}", flush=True)

    train_df, valid_df, test_df = load_splits(cfg)
    for name, d in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        print(
            f"  {name:5s} {len(d):>7,} rows | "
            f"positive {d['y'].mean():.3f} | "
            f"{d['mhc_sequence'].nunique()} unique MHC seqs",
            flush=True,
        )

    datasets = {
        "train": NeoantigenDataset(train_df),
        "valid": NeoantigenDataset(valid_df),
        "test": NeoantigenDataset(test_df),
    }

    train_sampler = None
    if cfg.world_size > 1:
        from torch.utils.data import DistributedSampler

        train_sampler = DistributedSampler(
            datasets["train"],
            num_replicas=cfg.world_size,
            rank=cfg.rank,
            shuffle=True,
            drop_last=True,
        )

    common = dict(
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(
        datasets["train"],
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=False,
        **common,
    )
    # Evaluation has no backward pass, so a wider batch costs nothing extra.
    valid_loader = DataLoader(
        datasets["valid"], batch_size=cfg.batch_size * 2, shuffle=False, **common
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=cfg.batch_size * 2, shuffle=False, **common
    )

    splits = {"train": train_df, "valid": valid_df, "test": test_df}
    return train_loader, valid_loader, test_loader, splits


def unique_mhc_sequences(splits: dict) -> list:
    """Every distinct MHC sequence across all splits — the precompute worklist."""
    seqs = pd.concat([d["mhc_sequence"] for d in splits.values()])
    return sorted(seqs.unique().tolist())
