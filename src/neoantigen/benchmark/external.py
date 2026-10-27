# src/neoantigen/benchmark/external.py
"""
Adapters for peptide-MHC predictors that live outside this repo.

Each one is optional and reports clearly when it cannot run, so the benchmark
suite degrades to whatever is actually installed rather than failing outright.

``pepmhc_archive``
    The retired Siamese n-gram transformer in ``archive_PEPMHC/``, scored from
    its own checkpoint. This is the most direct "did AETHER actually improve on
    what we had?" comparison, since it was trained on this very dataset.

``mhcflurry``
    Open-source, pip-installable, and a reasonable stand-in for the
    state of the art. ``pip install mhcflurry && mhcflurry-downloads fetch``.

``netmhcpan``
    The field standard, but license-gated: you must download and install the
    binary yourself from DTU. The adapter shells out to it if it is on PATH.

The adapters return NaN for rows they cannot score (an allele outside their
supported set, a peptide of unusable length). The runner reports coverage
alongside the metrics and computes metrics only over scored rows, so a
predictor is never silently credited for rows it declined.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import Baseline


def affinity_nm_to_score(nm: np.ndarray) -> np.ndarray:
    """
    Standard 1 - log50k(nM) transform, clipped to [0, 1].

    Affinity predictors emit IC50 in nanomolar where *lower* means a stronger
    binder, so it has to be inverted before it can be compared to a probability.
    50,000 nM is the conventional ceiling for "does not bind".
    """
    nm = np.asarray(nm, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = 1.0 - np.log(np.clip(nm, 1e-3, None)) / np.log(50000.0)
    return np.clip(score, 0.0, 1.0)


# ── The model AETHER replaces ─────────────────────────────────────────────────

class PepMHCArchive(Baseline):
    """
    The archived class-I Siamese transformer from ``archive_PEPMHC/``.

    Two quirks of the original training code carry over and are reproduced
    faithfully by its ``Predictor`` (see ``archive_PEPMHC/pepmhc/README.md``):
    the model attends across the *batch* dimension, and its positional encoding
    is indexed by position in the batch. A score therefore depends on the batch
    a row was scored in. We mitigate that the way the predictor intends — score
    at the training batch size (32) and feed rows allele-major, so batches are
    uniform — but the numbers still carry more noise than a normal model's.

    It is already trained, so ``fit`` only records the fallback rate.
    """

    name = "pepmhc_archive"
    outputs_probability = True

    def __init__(self, repo_dir: str | None = None, device: str = "cpu",
                 batch_size: int = 32):
        self.repo_dir = repo_dir
        self.device = device
        self.batch_size = batch_size

    def _load(self):
        import sys

        repo = self.repo_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "archive_PEPMHC"
        )
        repo = os.path.abspath(repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from pepmhc.predictor import Predictor  # noqa: E402

        return Predictor(device=self.device, batch_size=self.batch_size)

    def fit(self, train_df):
        return self

    def predict(self, df):
        if "mhc_allele" not in df.columns:
            raise ValueError(
                "pepmhc_archive scores by allele name and needs an mhc_allele column."
            )
        predictor = self._load()

        # Allele-major ordering keeps each batch to a single allele, which is
        # what the predictor's MHC-arm cache (and the batch quirk) expect.
        order = np.argsort(df["mhc_allele"].to_numpy(), kind="stable")
        peps = df["epitope"].to_numpy()[order]
        alleles = df["mhc_allele"].to_numpy()[order]

        scored = predictor.predict_pairs(peps, alleles, batch_size=self.batch_size)

        out = np.full(len(df), np.nan, dtype=np.float32)
        out[order] = scored
        return out


# ── MHCflurry ─────────────────────────────────────────────────────────────────

class MHCflurryBaseline(Baseline):
    """
    MHCflurry 2.x. Uses the presentation predictor when its downloads are
    present, falling back to the affinity predictor alone.

    Install with::

        pip install mhcflurry
        mhcflurry-downloads fetch
    """

    name = "mhcflurry"
    outputs_probability = True

    def __init__(self, use_presentation: bool = True):
        self.use_presentation = use_presentation

    def fit(self, train_df):
        return self

    def predict(self, df):
        try:
            import mhcflurry  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "mhcflurry is not installed. "
                "`pip install mhcflurry && mhcflurry-downloads fetch`"
            ) from e

        if "mhc_allele" not in df.columns:
            raise ValueError("mhcflurry scores by allele name and needs mhc_allele.")

        peptides = df["epitope"].astype(str).tolist()
        alleles = df["mhc_allele"].astype(str).tolist()

        predictor, kind = self._load()
        supported = set(predictor.supported_alleles)

        # MHCflurry only handles 8-15mers over the standard alphabet; anything
        # else is out of scope. The alphabet test matters because
        # clean_sequences maps B/J/O/U/Z to X, and MHCflurry *raises* on X
        # rather than returning NaN -- one such peptide in 35k would otherwise
        # take the whole baseline out of the comparison.
        STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
        ok = np.array([
            (8 <= len(p) <= 15) and (a in supported) and set(p) <= STANDARD_AA
            for p, a in zip(peptides, alleles)
        ])
        out = np.full(len(df), np.nan, dtype=np.float32)
        if not ok.any():
            print("    mhcflurry: no rows in scope", flush=True)
            return out

        if kind == "presentation":
            # Class1PresentationPredictor.predict() reads `alleles` as the
            # allele set of ONE sample and returns the strongest score across
            # the set -- it does not pair element-wise. Passing the whole
            # column turns every score into a max over ~100 alleles. Group by
            # allele, as run_mhcflurry.py already does.
            by_allele: dict[str, list[int]] = {}
            for i in np.flatnonzero(ok):
                by_allele.setdefault(alleles[i], []).append(i)
            for allele, idxs in sorted(by_allele.items()):
                res = predictor.predict(
                    peptides=[peptides[i] for i in idxs],
                    alleles=[allele], verbose=0,
                )
                out[idxs] = res["presentation_score"].to_numpy().astype(np.float32)
        else:
            # The affinity predictor does broadcast element-wise.
            sub_pep = [p for p, k in zip(peptides, ok) if k]
            sub_all = [a for a, k in zip(alleles, ok) if k]
            res = predictor.predict(peptides=sub_pep, alleles=sub_all)
            out[ok] = affinity_nm_to_score(np.asarray(res)).astype(np.float32)
        print(f"    mhcflurry ({kind}): scored {int(ok.sum()):,}/{len(df):,} rows",
              flush=True)
        return out

    def _load(self):
        if self.use_presentation:
            try:
                from mhcflurry import Class1PresentationPredictor

                return Class1PresentationPredictor.load(), "presentation"
            except Exception as e:
                print(f"    mhcflurry presentation predictor unavailable ({e}); "
                      f"falling back to affinity", flush=True)
        from mhcflurry import Class1AffinityPredictor

        return Class1AffinityPredictor.load(), "affinity"


# ── NetMHCpan ─────────────────────────────────────────────────────────────────

class NetMHCpanBaseline(Baseline):
    """
    NetMHCpan 4.x, via its command-line binary.

    The binary is license-gated and must be installed separately from
    https://services.healthtech.dtu.dk/services/NetMHCpan-4.2/ — this adapter
    only drives it. Set ``--netmhcpan_bin`` or put ``netMHCpan`` on PATH.

    Scores are ``1 - rank/100`` from the %EL (or %Rank) column, so that higher
    means a stronger binder, matching every other predictor here.
    """

    name = "netmhcpan"
    outputs_probability = False

    def __init__(self, binary: str = "netMHCpan", length_range=(8, 15)):
        self.binary = binary
        self.length_range = length_range

    def fit(self, train_df):
        return self

    def _format_allele(self, allele: str) -> str:
        # HLA-A*02:01 -> HLA-A02:01, which is what the binary expects.
        return str(allele).replace("*", "")

    def predict(self, df):
        if shutil.which(self.binary) is None:
            raise RuntimeError(
                f"{self.binary!r} not found on PATH. NetMHCpan is license-gated; "
                "install it from DTU and pass --netmhcpan_bin."
            )
        if "mhc_allele" not in df.columns:
            raise ValueError("netmhcpan scores by allele name and needs mhc_allele.")

        lo, hi = self.length_range
        out = np.full(len(df), np.nan, dtype=np.float32)
        peptides = df["epitope"].astype(str).to_numpy()
        alleles = df["mhc_allele"].astype(str).to_numpy()

        # Peptide input accepts mixed lengths; one call per allele avoids repeated model loads.
        groups: dict[tuple, list[int]] = {}
        for i, (p, a) in enumerate(zip(peptides, alleles)):
            if lo <= len(p) <= hi and set(p) <= set("ACDEFGHIKLMNPQRSTVWY"):
                groups.setdefault((a, 0), []).append(i)

        for (allele, length), idxs in groups.items():
            scores = self._run_group(peptides[idxs], allele, length)
            if scores is not None:
                out[idxs] = scores
        return out

    def _run_group(self, peptides, allele, length):
        with tempfile.TemporaryDirectory() as tmp:
            pep_file = os.path.join(tmp, "peptides.txt")
            with open(pep_file, "w") as f:
                f.write("\n".join(peptides) + "\n")
            cmd = [self.binary, "-p", pep_file, "-a", self._format_allele(allele),
                   "-BA"]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            except (OSError, subprocess.TimeoutExpired) as e:
                print(f"    netmhcpan failed for {allele}/{length}: {e}", flush=True)
                return None
            if res.returncode != 0:
                print(f"    netmhcpan error for {allele}/{length}: "
                      f"{res.stderr.strip()[:200]}", flush=True)
                return None
            # Preserve row identity, including peptides with tied predictions.
            parsed = self._parse(res.stdout, len(peptides), expected_peptides=peptides)
            return parsed

    @staticmethod
    def _parse(stdout: str, n: int, expected_peptides=None):
        """
        Pull the %Rank column out of the table, in input order.

        Located by header name, not by position from the end of the row. The
        last field is Aff(nM) on a non-binder row and the BindLevel flag
        ("<= SB") on a binder, so counting backwards reads the affinity as a
        rank on one and raises on the other.
        """
        scores, seen, rank_col = [], 0, None
        peptide_col = None
        parsed_peptides = []
        for line in stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            # Header: "Pos MHC Peptide ... Score_EL %Rank_EL Score_BA %Rank_BA ..."
            if rank_col is None and parts[0] == "Pos":
                peptide_col = parts.index("Peptide")
                for want in ("%Rank_EL", "%Rank_BA", "%Rank"):
                    if want in parts:
                        rank_col = parts.index(want)
                        break
                continue
            # Data rows start with the position index and carry numeric columns.
            if len(parts) < 8 or not parts[0].isdigit():
                continue
            if rank_col is None:
                print("    netmhcpan: no %Rank column in the header", flush=True)
                return None
            try:
                rank = float(parts[rank_col])
            except (ValueError, IndexError):
                continue
            parsed_peptides.append(parts[peptide_col])
            scores.append(1.0 - rank / 100.0)
            seen += 1
            if seen >= n:
                break
        if len(scores) != n:
            print(f"    netmhcpan: parsed {len(scores)} of {n} rows", flush=True)
            return None
        if expected_peptides is not None and list(expected_peptides) != parsed_peptides:
            raise ValueError("NetMHCpan output peptide order does not match input")
        return np.asarray(scores, dtype=np.float32)


class BigMHCBaseline(Baseline):
    """BigMHC EL ensemble; exact supported alleles only, without fuzzy remapping.

    EL is presentation rather than binding, and the scores are ranking scores.
    The downloaded ensemble is executed in its own process to isolate imports.
    """
    name = "bigmhc"
    outputs_probability = True

    def __init__(self, repo_dir="tools/bigmhc", python=None, device="cpu"):
        self.repo_dir = Path(repo_dir).resolve()
        self.python = python or sys.executable
        self.device = device

    def fit(self, train_df):
        return self

    def predict(self, df):
        import re
        normalize = lambda a: re.sub(r"[^A-Z0-9]", "", str(a).upper())
        supported = {normalize(a): a for a in pd.read_csv(
            self.repo_dir / "data/pseudoseqs.csv", usecols=[0]).iloc[:, 0]}
        peptides = df.epitope.astype(str).to_numpy()
        alleles = df.mhc_allele.astype(str).map(lambda a: supported.get(normalize(a)))
        ok = np.array([pd.notna(a) and 8 <= len(p) <= 15 and
                       set(p) <= set("ACDEFGHIKLMNPQRSTVWY")
                       for p,a in zip(peptides, alleles)])
        out = np.full(len(df), np.nan)
        if not ok.any():
            return out
        with tempfile.TemporaryDirectory() as tmp:
            inp, output = Path(tmp)/"input.csv", Path(tmp)/"output.csv"
            pd.DataFrame(dict(mhc=alleles[ok].to_numpy(), pep=peptides[ok])).to_csv(inp, index=False)
            env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
            cmd = [self.python, str(self.repo_dir/"src/predict.py"),
                   "-i", str(inp), "-o", str(output), "-m", "el", "-a", "0",
                   "-p", "1", "-c", "1", "-d", self.device, "-j", "1",
                   "-b", "1024", "-v", "0"]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
            if result.returncode:
                raise RuntimeError(f"BigMHC failed: {result.stderr[-3000:]}")
            pred = pd.read_csv(output)
            # Upstream sorts by peptide-MHC UID; restore the target order by key.
            keys = list(zip(pred.mhc.map(normalize), pred.pep))
            lookup = {}
            for key, score in zip(keys, pred.BigMHC_EL):
                if key in lookup and not np.isclose(lookup[key], score):
                    raise ValueError("BigMHC returned conflicting scores for a pair")
                lookup[key] = score
            expected = list(zip(alleles[ok].map(normalize), peptides[ok]))
            if set(keys) != set(expected) or len(pred) != ok.sum():
                raise ValueError("BigMHC output keys do not match input")
            out[ok] = [lookup[key] for key in expected]
        return out


REGISTRY = {
    c.name: c for c in (PepMHCArchive, MHCflurryBaseline, NetMHCpanBaseline, BigMHCBaseline)
}
