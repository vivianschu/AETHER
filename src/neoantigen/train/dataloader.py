# dataloader.py
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset


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


def _maybe_login_hint():
    # purely informational; no side effects
    if os.environ.get("HF_TOKEN"):
        return "HF_TOKEN is set."
    return (
        "HF_TOKEN not set. If the file is private/gated, do one of:\n"
        "  • export HF_TOKEN=... (recommended)\n"
        "  • huggingface-cli login\n"
    )


def load_mhc_dataframe_from_hf(
    csv_url_or_path: str,
    hf_token_env: str = "HF_TOKEN",
) -> pd.DataFrame:
    """
    Downloads CSV via Hugging Face Datasets library.
    Works for:
      - public URLs
      - private/gated URLs if you provide a token (HF_TOKEN) or are logged in
    """
    token = os.environ.get(hf_token_env, None)

    # datasets.load_dataset supports token=... in newer versions; some installs use use_auth_token=...
    # We'll try token first, then fallback.
    try:
        ds = load_dataset(
            "csv",
            data_files=csv_url_or_path,
            token=token,
        )
    except TypeError:
        # older datasets
        ds = load_dataset(
            "csv",
            data_files=csv_url_or_path,
            use_auth_token=token,
        )

    # Single split "train" produced by csv loader
    df = ds["train"].to_pandas()
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["epitope", "mhc_sequence", "mhc_binding"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}. Found: {list(df.columns)}")

    df = df[required].copy()
    df["y"] = pd.to_numeric(df["mhc_binding"], errors="coerce")

    # Drop NaNs (labels or sequences)
    df = df.dropna(subset=["epitope", "mhc_sequence", "y"]).reset_index(drop=True)

    # Light cleanup
    df["epitope"] = df["epitope"].astype(str).str.upper().str.replace(" ", "")
    df["mhc_sequence"] = df["mhc_sequence"].astype(str).str.upper().str.replace(" ", "")

    return df


def make_splits(
    df: pd.DataFrame,
    seed: int = 1337,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    stratify: bool = True,
    stratify_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_frac + valid_frac + test_frac, 1.0):
        raise ValueError("train_frac + valid_frac + test_frac must sum to 1.0")

    n = len(df)
    rng = np.random.default_rng(seed)

    if stratify:
        # Bin labels for stratification only (does NOT change regression targets)
        y_bin = (df["y"].to_numpy() >= stratify_threshold).astype(np.int32)

        idx_train, idx_valid, idx_test = [], [], []
        for cls in [0, 1]:
            cls_idx = np.where(y_bin == cls)[0]
            rng.shuffle(cls_idx)

            n_cls = len(cls_idx)
            n_train = int(round(train_frac * n_cls))
            n_valid = int(round(valid_frac * n_cls))
            # remainder goes to test
            n_test = n_cls - n_train - n_valid

            idx_train.append(cls_idx[:n_train])
            idx_valid.append(cls_idx[n_train:n_train + n_valid])
            idx_test.append(cls_idx[n_train + n_valid:])

        idx_train = np.concatenate(idx_train)
        idx_valid = np.concatenate(idx_valid)
        idx_test = np.concatenate(idx_test)

        rng.shuffle(idx_train)
        rng.shuffle(idx_valid)
        rng.shuffle(idx_test)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_train = int(round(train_frac * n))
        n_valid = int(round(valid_frac * n))
        idx_train = idx[:n_train]
        idx_valid = idx[n_train:n_train + n_valid]
        idx_test = idx[n_train + n_valid:]

    return df.iloc[idx_train].copy(), df.iloc[idx_valid].copy(), df.iloc[idx_test].copy()


@dataclass
class DataConfig:
    csv_url: str
    seed: int = 1337
    batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True
    stratify: bool = True
    stratify_threshold: float = 0.5


def get_dataloaders(cfg: DataConfig):
    print(f"Loading dataset from: {cfg.csv_url}")
    print(_maybe_login_hint())

    raw_df = load_mhc_dataframe_from_hf(cfg.csv_url)
    df = prepare_df(raw_df)

    train_df, valid_df, test_df = make_splits(
        df,
        seed=cfg.seed,
        stratify=cfg.stratify,
        stratify_threshold=cfg.stratify_threshold,
    )

    train_ds = NeoantigenDataset(train_df)
    valid_ds = NeoantigenDataset(valid_df)
    test_ds = NeoantigenDataset(test_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_loader, valid_loader, test_loader, df