# src/neoantigen/benchmark/baselines.py
"""
Reference models to compare AETHER against.

These exist to answer three different questions, and it is worth being explicit
about which is which:

**Is the model learning anything about peptides at all?**
    ``majority``, ``allele_prior``, ``length_prior``, ``allele_length_prior``
    predict from label frequencies alone and never look at the peptide
    sequence. On a 77%-positive set with long-tailed allele support these are
    not as weak as they sound — if AETHER only narrowly beats
    ``allele_length_prior``, it has mostly learned which alleles are
    well-represented, not binding chemistry.

**Does it beat the classical approach?**
    ``pssm`` is a per-allele position-specific scoring matrix, essentially what
    the field used before neural networks. ``blosum_lr`` is a logistic
    regression on BLOSUM-encoded peptides. A 650M-parameter transformer should
    clear both by a wide margin; if it does not, the extra machinery is not
    paying for itself.

**Does the architecture earn its keep?**
    ``esm2_probe`` is the important one. It takes the *same frozen ESM2
    embeddings* AETHER uses, mean-pools them, concatenates peptide and MHC, and
    fits a linear model. This compares the cross-attention, learned attention
    pooling and MLP head against a simpler representation. The default probe
    training cap is 100,000 rows, so its gap to AETHER also reflects training-set
    size unless the fitting populations are explicitly matched.

Every baseline implements ``fit(train_df)`` / ``predict(df)`` over the canonical
schema (``epitope``, ``mhc_sequence``, ``y``, optionally ``mhc_allele``) and
returns a score per row. Scores only need to be monotone in confidence, since
the metrics that matter here (AUROC, AUPRC) are rank-based; baselines that emit
genuine probabilities set ``outputs_probability = True``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# Background amino acid frequencies (UniProt-wide, rounded). Used as the null
# model a PSSM scores against.
BACKGROUND = np.array([
    0.0825, 0.0137, 0.0545, 0.0675, 0.0386, 0.0707, 0.0227, 0.0596, 0.0584,
    0.0966, 0.0242, 0.0406, 0.0470, 0.0393, 0.0553, 0.0656, 0.0534, 0.0687,
    0.0108, 0.0292,
])
BACKGROUND = BACKGROUND / BACKGROUND.sum()


def canonical9(peptide: str) -> str:
    """
    Align a peptide of any length onto 9 slots, anchors preserved.

    Class-I binding is anchored at roughly P2 and P-omega, so the standard trick
    is to keep the first four and last five residues and absorb any extra length
    in the middle, which is where the peptide bulges out of the groove. Shorter
    peptides get a gap there instead. ``-`` means "no residue", scored as
    background.
    """
    L = len(peptide)
    if L == 9:
        return peptide
    if L > 9:
        return peptide[:4] + peptide[-5:]
    return (peptide[:4] + "-" * (9 - L) + peptide[4:]).ljust(9, "-")[:9]


class Baseline:
    """Interface shared by every reference model."""

    name = "baseline"
    outputs_probability = False

    def fit(self, train_df: pd.DataFrame) -> "Baseline":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ── Label-frequency baselines (never see the peptide sequence) ────────────────

class MajorityBaseline(Baseline):
    """Constant prediction: the training positive rate. AUROC is 0.5 by construction."""

    name = "majority"
    outputs_probability = True

    def fit(self, train_df):
        self.rate = float(train_df["y"].mean())
        return self

    def predict(self, df):
        return np.full(len(df), self.rate, dtype=np.float32)


class _GroupPrior(Baseline):
    """Predict the training positive rate of whatever group a row belongs to."""

    group_cols: list[str] = []
    outputs_probability = True

    def fit(self, train_df):
        df = train_df.copy()
        self._prepare(df)
        self.global_rate = float(df["y"].mean())
        self.rates = df.groupby(self.group_cols)["y"].mean()
        return self

    def _prepare(self, df):
        pass

    def predict(self, df):
        df = df.copy()
        self._prepare(df)
        idx = pd.MultiIndex.from_frame(df[self.group_cols]) if len(self.group_cols) > 1 \
            else pd.Index(df[self.group_cols[0]])
        out = self.rates.reindex(idx).to_numpy(dtype=np.float64)
        # Groups unseen in training fall back to the global rate.
        return np.nan_to_num(out, nan=self.global_rate).astype(np.float32)


class AllelePrior(_GroupPrior):
    """Per-allele training positive rate."""
    name = "allele_prior"
    group_cols = ["mhc_allele"]


class LengthPrior(_GroupPrior):
    """Per-peptide-length training positive rate."""
    name = "length_prior"
    group_cols = ["_len"]

    def _prepare(self, df):
        df["_len"] = df["epitope"].str.len()


class AlleleLengthPrior(_GroupPrior):
    """Per (allele, peptide length) training positive rate — the strongest
    baseline that still never looks at the peptide sequence."""
    name = "allele_length_prior"
    group_cols = ["mhc_allele", "_len"]

    def _prepare(self, df):
        df["_len"] = df["epitope"].str.len()


# ── Classical sequence baselines ──────────────────────────────────────────────

class PSSMBaseline(Baseline):
    """
    Per-allele position-specific scoring matrix over the 9-slot alignment.

    Built from training *binders* only: count residues per position, add a
    pseudocount, and score as a log-odds against background frequency. An allele
    with too few binders to estimate a matrix falls back to the pooled matrix.

    This is roughly what MHC binding prediction looked like before neural
    networks, and it is a genuinely strong baseline for well-represented alleles.
    """

    name = "pssm"

    def __init__(self, min_binders: int = 20, pseudocount: float = 0.5):
        self.min_binders = min_binders
        self.pseudocount = pseudocount

    def _matrix(self, peptides) -> np.ndarray:
        counts = np.full((9, 20), self.pseudocount)
        for pep in peptides:
            for pos, aa in enumerate(canonical9(pep)):
                j = AA_INDEX.get(aa)
                if j is not None:
                    counts[pos, j] += 1.0
        freq = counts / counts.sum(axis=1, keepdims=True)
        return np.log(freq / BACKGROUND)

    def fit(self, train_df):
        binders = train_df[train_df["y"] >= 0.5]
        self.pooled = self._matrix(binders["epitope"])
        self.per_allele = {}
        if "mhc_allele" in binders.columns:
            for allele, g in binders.groupby("mhc_allele"):
                if len(g) >= self.min_binders:
                    self.per_allele[allele] = self._matrix(g["epitope"])
        return self

    def predict(self, df):
        has_allele = "mhc_allele" in df.columns
        alleles = df["mhc_allele"].to_numpy() if has_allele else np.full(len(df), None)
        out = np.zeros(len(df), dtype=np.float32)
        for i, (pep, allele) in enumerate(zip(df["epitope"].to_numpy(), alleles)):
            M = self.per_allele.get(allele, self.pooled)
            s = 0.0
            for pos, aa in enumerate(canonical9(pep)):
                j = AA_INDEX.get(aa)
                if j is not None:
                    s += M[pos, j]
            out[i] = s
        return out


def _blosum62() -> np.ndarray:
    """BLOSUM62 rows for the 20 standard residues, as a 20x20 float array."""
    from Bio.Align import substitution_matrices

    m = substitution_matrices.load("BLOSUM62")
    alphabet = list(m.alphabet)
    idx = [alphabet.index(a) for a in AA]
    return np.array(m)[np.ix_(idx, idx)].astype(np.float32)


class BlosumLogistic(Baseline):
    """
    Logistic regression on a BLOSUM62-encoded 9-slot peptide plus a one-hot
    allele indicator.

    Note the allele is a *lookup*, not a sequence, so unlike AETHER this cannot
    say anything about an allele it never saw in training. On an unseen-allele
    split it degrades to the peptide term alone, which is the honest comparison
    to draw there.
    """

    name = "blosum_lr"
    outputs_probability = True

    def __init__(self, max_rows: int = 0, seed: int = 1337):  # 0 = no cap
        self.max_rows = max_rows
        self.seed = seed

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        X = np.zeros((n, 9 * 20 + len(self.alleles_)), dtype=np.float32)
        for i, pep in enumerate(df["epitope"].to_numpy()):
            for pos, aa in enumerate(canonical9(pep)):
                j = AA_INDEX.get(aa)
                if j is not None:
                    X[i, pos * 20:(pos + 1) * 20] = self.blosum_[j]
        if "mhc_allele" in df.columns:
            offset = 9 * 20
            for i, allele in enumerate(df["mhc_allele"].to_numpy()):
                k = self.allele_index_.get(allele)
                if k is not None:
                    X[i, offset + k] = 1.0
        return X

    def fit(self, train_df):
        from sklearn.linear_model import LogisticRegression

        self.blosum_ = _blosum62()
        self.alleles_ = (sorted(train_df["mhc_allele"].unique())
                         if "mhc_allele" in train_df.columns else [])
        self.allele_index_ = {a: i for i, a in enumerate(self.alleles_)}

        df = train_df
        if self.max_rows and len(df) > self.max_rows:
            # Subsampling here makes the esm2_probe-vs-AETHER gap a mix of
            # architecture and training-set size, so say so out loud.
            print(f"    {self.name}: fitting on {self.max_rows:,} of "
                  f"{len(df):,} train rows -- the gap to AETHER is then "
                  f"architecture AND training-set size", flush=True)
            df = df.sample(n=self.max_rows, random_state=self.seed)
        self.n_fit_rows = len(df)
        X = self._features(df)
        y = (df["y"].to_numpy() >= 0.5).astype(int)
        self.clf_ = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
        return self

    def predict(self, df):
        return self.clf_.predict_proba(self._features(df))[:, 1].astype(np.float32)


# ── Architecture ablation ─────────────────────────────────────────────────────

class ESM2LinearProbe(Baseline):
    """
    The ablation that matters: same frozen ESM2 embeddings AETHER uses, mean
    pooled, peptide and MHC concatenated, linear classifier on top.

    What this removes relative to AETHER is exactly the architecture under test —
    bidirectional cross-attention, learned attention pooling, and the MLP head.
    Whatever AETHER scores above this probe is what those components contribute;
    whatever it scores below is a sign they are getting in the way.

    Embeddings are computed once per unique sequence, so the ~150 MHC sequences
    cost almost nothing and the peptides dominate.
    """

    name = "esm2_probe"
    outputs_probability = True

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        max_rows: int = 0,   # 0 = fit on everything, as AETHER does
        batch_size: int = 256,
        device: str | None = None,
        seed: int = 1337,
    ):
        self.model_name = model_name
        self.max_rows = max_rows
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self._cache: dict[str, np.ndarray] = {}

    def _embed_unique(self, seqs) -> None:
        import torch
        from transformers import AutoTokenizer, EsmModel

        todo = sorted({s for s in seqs if s not in self._cache})
        if not todo:
            return

        if not hasattr(self, "_model"):
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = torch.device(dev)
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = EsmModel.from_pretrained(
                self.model_name, add_pooling_layer=False
            ).to(self._device).eval()

        print(f"    embedding {len(todo):,} unique sequences", flush=True)
        with torch.no_grad():
            for i in range(0, len(todo), self.batch_size):
                chunk = todo[i:i + self.batch_size]
                enc = self._tok(chunk, padding=True, return_tensors="pt").to(self._device)
                out = self._model(**enc).last_hidden_state
                # Mean over real tokens only; padding would drag the mean around.
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = pooled.float().cpu().numpy()
                for j, s in enumerate(chunk):
                    self._cache[s] = pooled[j]

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        peps = df["epitope"].tolist()
        mhcs = df["mhc_sequence"].tolist()
        self._embed_unique(peps)
        self._embed_unique(mhcs)
        return np.stack([
            np.concatenate([self._cache[p], self._cache[m]]) for p, m in zip(peps, mhcs)
        ]).astype(np.float32)

    def fit(self, train_df):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        df = train_df
        if self.max_rows and len(df) > self.max_rows:
            # Subsampling here makes the esm2_probe-vs-AETHER gap a mix of
            # architecture and training-set size, so say so out loud.
            print(f"    {self.name}: fitting on {self.max_rows:,} of "
                  f"{len(df):,} train rows -- the gap to AETHER is then "
                  f"architecture AND training-set size", flush=True)
            df = df.sample(n=self.max_rows, random_state=self.seed)
        self.n_fit_rows = len(df)
        X = self._features(df)
        y = (df["y"].to_numpy() >= 0.5).astype(int)

        self.scaler_ = StandardScaler().fit(X)
        self.clf_ = LogisticRegression(max_iter=3000, C=1.0).fit(
            self.scaler_.transform(X), y
        )
        return self

    def predict(self, df):
        X = self.scaler_.transform(self._features(df))
        return self.clf_.predict_proba(X)[:, 1].astype(np.float32)


REGISTRY = {
    c.name: c for c in (
        MajorityBaseline, AllelePrior, LengthPrior, AlleleLengthPrior,
        PSSMBaseline, BlosumLogistic, ESM2LinearProbe,
    )
}
