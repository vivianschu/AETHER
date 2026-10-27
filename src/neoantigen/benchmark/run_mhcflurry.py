# src/neoantigen/benchmark/run_mhcflurry.py
"""
Score a split with MHCflurry 2.x, from a separate Python environment.

MHCflurry can use its own environment and downloaded model release. This
script is standalone: it needs ``pandas``, ``numpy`` and ``mhcflurry`` with
its supported backend, and never imports anything from ``neoantigen``.

Set-up (once, in its own venv):

    python -m venv ~/venv/mhctools
    ~/venv/mhctools/bin/pip install mhcflurry
    ~/venv/mhctools/bin/mhcflurry-downloads fetch models_class1_presentation

Then:

    # 1. the main runner writes the prepared rows
    python -m neoantigen.benchmark.run --data_dir ... --out_dir outputs/benchmarks ...

    # 2. score them with MHCflurry, in its own environment
    ~/venv/mhctools/bin/python src/neoantigen/benchmark/run_mhcflurry.py \
        --target_rows outputs/benchmarks/_target_rows.csv \
        --out_csv outputs/benchmarks/mhcflurry_predictions.csv

    # 3. fold the result back in
    python -m neoantigen.benchmark.run --data_dir ... --out_dir outputs/benchmarks ... \
        --include mhcflurry=outputs/benchmarks/mhcflurry_predictions.csv

Reading ``_target_rows.csv`` rather than the raw split is what guarantees the
rows line up: they have already been through the same cleaning and are in the
same order the other methods scored.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def affinity_nm_to_score(nm) -> np.ndarray:
    """1 - log50k(nM), clipped to [0, 1]; inverts affinity so higher = stronger."""
    nm = np.asarray(nm, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = 1.0 - np.log(np.clip(nm, 1e-3, None)) / np.log(50000.0)
    return np.clip(score, 0.0, 1.0)


def load_predictor(prefer_presentation: bool):
    if prefer_presentation:
        try:
            from mhcflurry import Class1PresentationPredictor

            return Class1PresentationPredictor.load(), "presentation"
        except Exception as e:
            print(f"presentation predictor unavailable ({e}); using affinity")
    from mhcflurry import Class1AffinityPredictor

    return Class1AffinityPredictor.load(), "affinity"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target_rows", required=True,
                   help="_target_rows.csv written by neoantigen.benchmark.run")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--affinity_only", action="store_true",
                   help="Use the affinity predictor even if presentation is available.")
    p.add_argument("--batch_size", type=int, default=50_000)
    args = p.parse_args()

    rows = pd.read_csv(args.target_rows)
    for col in ("epitope", "y"):
        if col not in rows.columns:
            raise SystemExit(f"{args.target_rows} has no {col!r} column")
    if "mhc_allele" not in rows.columns:
        raise SystemExit("MHCflurry scores by allele name; _target_rows.csv has no "
                         "mhc_allele column.")

    peptides = rows["epitope"].astype(str).tolist()
    alleles = rows["mhc_allele"].astype(str).tolist()

    predictor, kind = load_predictor(not args.affinity_only)
    supported = set(predictor.supported_alleles)
    print(f"MHCflurry {kind} predictor | {len(supported)} supported alleles")

    # Rows MHCflurry cannot take are left as NaN, and the main runner reports
    # them as reduced coverage rather than counting them as scored. Three
    # reasons it declines a row:
    #   * peptide outside 8-15 residues,
    #   * allele outside its supported set,
    #   * a non-standard residue. Our cleaning maps unknown codes to X, which
    #     MHCflurry rejects outright (it raises rather than returning NaN), so
    #     these have to be filtered here or they abort the whole batch.
    STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
    lengths_ok = np.array([8 <= len(p) <= 15 for p in peptides])
    allele_ok = np.array([a in supported for a in alleles])
    residues_ok = np.array([set(p) <= STANDARD_AA for p in peptides])
    in_scope = lengths_ok & allele_ok & residues_ok

    missing = sorted({a for a in alleles if a not in supported})
    if missing:
        print(f"{len(missing)} alleles unsupported, e.g. {missing[:5]}")
    print(f"declined: {int((~lengths_ok).sum()):,} on length, "
          f"{int((~allele_ok).sum()):,} on allele, "
          f"{int((~residues_ok).sum()):,} on non-standard residues")
    print(f"scoring {int(in_scope.sum()):,} / {len(rows):,} rows")

    scores = np.full(len(rows), np.nan, dtype=np.float64)

    # Group by allele and score one allele at a time.
    #
    # Class1PresentationPredictor.predict() reads `alleles` as the allele set of
    # a single sample (at most 6) and returns the strongest prediction across
    # them — it does NOT pair alleles with peptides element-wise the way the
    # affinity predictor does. Passing one allele per call is what makes each
    # score a genuine (peptide, allele) pair rather than a max over a set.
    by_allele: dict[str, list[int]] = {}
    for i in np.flatnonzero(in_scope):
        by_allele.setdefault(alleles[i], []).append(i)

    for n, (allele, idxs) in enumerate(sorted(by_allele.items()), start=1):
        sub_pep = [peptides[i] for i in idxs]
        if kind == "presentation":
            res = predictor.predict(peptides=sub_pep, alleles=[allele], verbose=0)
            vals = res["presentation_score"].to_numpy()
        else:
            vals = np.asarray(predictor.predict(peptides=sub_pep,
                                                alleles=[allele] * len(sub_pep)))
            vals = affinity_nm_to_score(vals)
        if len(vals) != len(idxs):
            raise SystemExit(f"{allele}: got {len(vals)} scores for {len(idxs)} peptides")
        scores[idxs] = vals
        if n % 25 == 0 or n == len(by_allele):
            print(f"  {n} / {len(by_allele)} alleles", flush=True)

    out = pd.DataFrame({
        "y_true": (rows["y"].to_numpy() >= 0.5).astype(int),
        "y_pred_prob": scores,
        "peptide": peptides,
        "mhc_allele": alleles,
    })
    out.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}  ({np.isfinite(scores).sum():,} scored)")


if __name__ == "__main__":
    main()
