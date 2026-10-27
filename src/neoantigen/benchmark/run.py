# src/neoantigen/benchmark/run.py
"""
Run reference models on a split and compare them to AETHER.

Every method writes ``<out_dir>/<name>_predictions.csv`` in exactly the schema
``train.py`` uses for its own test predictions, so one notebook can plot them
all together and an AETHER run can be folded in with ``--include``.

Fairness note
-------------
Methods differ in what they can score: MHCflurry handles 8-15mers for its
supported alleles, NetMHCpan has its own range, and the archived model only
knows the 161 alleles in its vocabulary. Scoring each one only on the rows it
accepted would flatter whichever method declines the hardest rows.

So the summary reports two things: ``auroc`` over each method's own scored rows
(with ``coverage``), and ``auroc_common`` over the subset **every** method
scored. ``within_allele_auroc_common`` is the primary comparison; pooled AUROC can
credit allele separation. AUPRC and tie-aware PPV@100 are also reported, with
per-allele tables and explicit eligible-row counts.

Usage
-----
    python -m neoantigen.benchmark.run \
        --data_dir data/mhcattnnet/classI_peptide_split \
        --split test \
        --out_dir outputs/benchmarks \
        --methods majority allele_length_prior pssm blosum_lr esm2_probe pepmhc_archive \
        --include aether=outputs/classI_peptide_split/test_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

from neoantigen.train.dataloader import DataConfig, load_splits

from neoantigen.metrics import ranking_metrics, per_allele_metrics
from neoantigen.provenance import file_sha256
from . import baselines as B
from . import external as E

ALL_METHODS = {**B.REGISTRY, **E.REGISTRY}

DEFAULT_METHODS = [
    "majority",
    "allele_prior",
    "allele_length_prior",
    "pssm",
    "blosum_lr",
    "esm2_probe",
    "mhcflurry",
    "netmhcpan",
    "bigmhc",
]


def score_frame(y_true, scores, alleles=None, min_allele_rows=50):
    return ranking_metrics(y_true, scores, alleles, min_allele_rows)


def load_included(spec: str) -> tuple[str, pd.DataFrame]:
    """``name=path`` -> (name, predictions frame)."""
    if "=" not in spec:
        raise ValueError(f"--include expects name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    df = pd.read_csv(path)
    for col in ("y_true", "y_pred_prob"):
        if col not in df.columns:
            raise ValueError(f"{path} has no {col!r} column")
    return name, df


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data_dir", default=None,
                   help="Folder with train.csv / val.csv / test.csv")
    p.add_argument("--csv_url", default=None, help="Single CSV instead of --data_dir")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                   help=f"Any of: {', '.join(sorted(ALL_METHODS))}")
    p.add_argument("--include", nargs="*", default=[], metavar="NAME=PATH",
                   help="Fold in an existing predictions CSV, e.g. AETHER's.")
    p.add_argument("--max_train_rows", type=int, default=0,
                   help=">0 subsamples the data the baselines are fit on.")
    p.add_argument("--max_eval_rows", type=int, default=0,
                   help=">0 subsamples the split being scored.")
    p.add_argument("--probe_model", default="facebook/esm2_t33_650M_UR50D",
                   help="Backbone for esm2_probe; use the same one AETHER used.")
    p.add_argument("--probe_max_rows", type=int, default=100_000)
    p.add_argument("--netmhcpan_bin", default="scripts/netmhcpan42")
    p.add_argument("--bigmhc_dir", default="tools/bigmhc")
    p.add_argument("--bigmhc_python", default=None)
    p.add_argument("--bigmhc_device", default="cpu")
    p.add_argument("--split_group", choices=["peptide", "allele", "pair"], default="peptide")
    p.add_argument("--min_allele_rows", type=int, default=50,
                   help="Minimum scored rows per allele for within-allele AUROC")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    unknown = set(args.methods) - set(ALL_METHODS)
    if unknown:
        raise SystemExit(f"Unknown methods: {sorted(unknown)}. "
                         f"Available: {sorted(ALL_METHODS)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    cfg = DataConfig(
        csv_url=args.csv_url, data_dir=args.data_dir, seed=args.seed, split_group=args.split_group,
        max_train_rows=args.max_train_rows, max_eval_rows=args.max_eval_rows,
    )
    train_df, valid_df, test_df = load_splits(cfg)
    target = {"train": train_df, "valid": valid_df, "test": test_df}[args.split]
    alleles = target["mhc_allele"].to_numpy() if "mhc_allele" in target else None
    y_true = (target["y"].to_numpy() >= 0.5).astype(int)

    print(f"\nFitting on {len(train_df):,} train rows, "
          f"scoring {len(target):,} {args.split} rows\n", flush=True)

    # Dump the prepared rows so a tool living in another environment (MHCflurry
    # needs TensorFlow, NetMHCpan is a separate binary) can score exactly these
    # rows in exactly this order, then be folded back in with --include. Without
    # this, each such script would have to re-implement the cleaning in
    # dataloader.prepare_df and hope it matched.
    target_path = os.path.join(args.out_dir, "_target_rows.csv")
    target.to_csv(target_path, index=False)

    # ── Run each method ───────────────────────────────────────────────────────
    predictions: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    for name in args.methods:
        print(f"[{name}]", flush=True)
        try:
            cls = ALL_METHODS[name]
            if name == "esm2_probe":
                model = cls(model_name=args.probe_model,
                            max_rows=args.probe_max_rows,
                            device=args.device, seed=args.seed)
            elif name == "netmhcpan":
                model = cls(binary=args.netmhcpan_bin)
            elif name == "bigmhc":
                model = cls(repo_dir=args.bigmhc_dir, python=args.bigmhc_python,
                            device=args.bigmhc_device)
            elif name == "pepmhc_archive":
                model = cls(device=args.device or "cpu")
            else:
                model = cls()

            model.fit(train_df)
            scores = np.asarray(model.predict(target), dtype=np.float64)
        except Exception as e:
            print(f"    SKIPPED: {e}\n", flush=True)
            continue

        predictions[name] = scores
        summary[name] = score_frame(y_true, scores, alleles, args.min_allele_rows)

        out = pd.DataFrame({"y_true": y_true, "y_pred_prob": scores,
                            "peptide": target["epitope"].to_numpy()})
        if "mhc_allele" in target.columns:
            out["mhc_allele"] = target["mhc_allele"].to_numpy()
        out.to_csv(os.path.join(args.out_dir, f"{name}_predictions.csv"), index=False)

        m = summary[name]
        print(f"    AUROC {m.get('auroc', float('nan')):.4f}  "
              f"AUPRC {m.get('auprc', float('nan')):.4f}  "
              f"coverage {m['coverage']:.1%}\n", flush=True)

    # ── Fold in externally-produced predictions (e.g. AETHER) ─────────────────
    for spec in args.include:
        name, df = load_included(spec)
        required = {"peptide", "mhc_allele", "y_true", "y_pred_prob"}
        if not required <= set(df):
            raise ValueError(f"{name}: inclusion requires row identity columns {sorted(required)}")
        if not np.array_equal((df.y_true.to_numpy() >= 0.5).astype(int), y_true):
            raise ValueError(f"{name}: included labels do not match target")
        if len(df) != len(target):
            print(f"[{name}] SKIPPED: {len(df):,} rows but the {args.split} split "
                  f"has {len(target):,}. It must be the same split.", flush=True)
            continue

        # Equal row counts are not alignment. The file is joined by position,
        # so a CSV produced from a different seed, a re-sorted CSV, or a
        # sibling split of the same size would scramble every row silently --
        # and asymmetrically, since this method's own AUROC is computed from
        # its own y_true column and would still look fine while auroc_common
        # collapsed. Both frames carry the keys, so check them.
        for csv_col, tgt_col in (("peptide", "epitope"), ("mhc_allele", "mhc_allele")):
            if csv_col in df.columns and tgt_col in target.columns:
                mism = int((df[csv_col].to_numpy() != target[tgt_col].to_numpy()).sum())
                if mism:
                    print(f"[{name}] SKIPPED: {mism:,} of {len(df):,} rows have a "
                          f"different {csv_col} than the {args.split} split at the "
                          f"same position. The file is not row-aligned.", flush=True)
                    break
        else:
            scores = df["y_pred_prob"].to_numpy(dtype=np.float64)
            predictions[name] = scores
            summary[name] = score_frame(y_true, scores, alleles, args.min_allele_rows)

            # Copy it into out_dir too, so the benchmark directory is self-contained
            # and the comparison notebook finds every method the same way.
            out = pd.DataFrame({"y_true": df["y_true"].to_numpy(), "y_pred_prob": scores})
            for col, fallback in (("peptide", target["epitope"]),
                                  ("mhc_allele", target.get("mhc_allele"))):
                if col in df.columns:
                    out[col] = df[col].to_numpy()
                elif fallback is not None:
                    out[col] = fallback.to_numpy()
            out.to_csv(os.path.join(args.out_dir, f"{name}_predictions.csv"), index=False)

            print(f"[{name}] included — AUROC {summary[name].get('auroc', float('nan')):.4f}",
                  flush=True)

    if not summary:
        raise SystemExit("No method produced predictions.")

    # ── Common-subset comparison ──────────────────────────────────────────────
    common = np.ones(len(y_true), dtype=bool)
    per_method_cov = {}
    for name, s in predictions.items():
        finite = np.isfinite(s)
        per_method_cov[name] = float(finite.mean())
        common &= finite
    print(f"\nCommon scored subset: {int(common.sum()):,} / {len(y_true):,} rows "
          f"({common.mean():.1%})", flush=True)
    if common.sum() < len(y_true):
        # Name whoever shrank it: one all-NaN method collapses the common
        # subset to nothing and silently drops the whole fair comparison.
        worst = sorted(per_method_cov.items(), key=lambda kv: kv[1])[:3]
        print("  limited by: "
              + ", ".join(f"{n} {c:.1%}" for n, c in worst), flush=True)

    if common.sum() > 0 and len(np.unique(y_true[common])) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score
        for name, s in predictions.items():
            common_metrics = score_frame(y_true[common], s[common],
                alleles[common] if alleles is not None else None, args.min_allele_rows)
            summary[name].update({f"{k}_common": v for k,v in common_metrics.items()})
        summary_meta = {"n_common": int(common.sum())}
    else:
        print("\n" + "!" * 72, flush=True)
        print("WARNING: the common subset is empty or single-class, so "
              "auroc_common was NOT computed.\n"
              "The table below falls back to `auroc`, which scores each method "
              "on\nits OWN subset -- that flatters whichever method declined "
              "the hardest\nrows, and the columns are not comparable to each "
              "other.", flush=True)
        print("!" * 72 + "\n", flush=True)
        summary_meta = {"n_common": int(common.sum())}

    # n_common belongs in the table, not only in benchmark_meta.json: the
    # notebook and the poster read the CSV.
    for name in summary:
        summary[name]["n_common"] = int(common.sum())
        summary[name]["coverage"] = per_method_cov.get(
            name, summary[name].get("coverage", float("nan")))

    for name, scores in predictions.items():
        if alleles is not None:
            per_allele_metrics(y_true, scores, alleles).to_csv(
                os.path.join(args.out_dir, f"{name}_per_allele.csv"), index=False)
            per_allele_metrics(y_true[common], scores[common], alleles[common]).to_csv(
                os.path.join(args.out_dir, f"{name}_per_allele_common.csv"), index=False)

    # ── Write + print ─────────────────────────────────────────────────────────
    table = pd.DataFrame(summary).T
    sort_col = next((c for c in ["within_allele_auroc_common", "auroc_common", "auroc"]
                     if c in table), None)
    if sort_col:
        table = table.sort_values(sort_col, ascending=False)
    table.index.name = "method"
    table.to_csv(os.path.join(args.out_dir, "benchmark_summary.csv"))
    target.loc[common].to_csv(os.path.join(args.out_dir, "_common_rows.csv"), index=False)
    prediction_hashes = {name: file_sha256(os.path.join(args.out_dir, f"{name}_predictions.csv"))
                         for name in predictions}
    with open(os.path.join(args.out_dir, "benchmark_meta.json"), "w") as f:
        json.dump({**summary_meta, "split": args.split,
                   "target_sha256": file_sha256(target_path),
                   "prediction_sha256": prediction_hashes,
                   "min_allele_rows": args.min_allele_rows,
                   "data_dir": args.data_dir, "methods": list(summary),
                   "requested_methods": args.methods, "config": vars(args),
                   "missing_methods": sorted(set(args.methods)-set(summary)),
                   "score_semantics": {"netmhcpan": "NetMHCpan 4.2 default EL rank, 1-rank/100; no -neo/-pathogen", "bigmhc": "BigMHC EL ensemble, exact supported alleles, 8-15mers"},
                   "ppv_at_100": "min(100,n_scored); fractional random tie-breaking at cutoff"}, f, indent=2)

    # n_common is printed, not just stored in benchmark_meta.json: without it
    # a reader cannot tell whether auroc_common covers the whole split or a
    # small, systematically easier intersection.
    cols = [c for c in ("within_allele_auroc_common", "auroc_common", "auprc_common", "ppv_at_100_common", "auroc", "auprc",
                        "coverage", "n_common") if c in table.columns]
    print("\n" + "=" * 72)
    print(f"BENCHMARK — {args.split} split")
    print("=" * 72)
    print(table[cols].to_string(float_format=lambda v: f"{v:.4f}"))
    print("=" * 72)
    if "auroc_common" in table.columns and int(common.sum()) < len(y_true):
        # The intersection keeps the rows every tool accepted -- well-covered
        # alleles and canonical lengths -- so it is an easier subpopulation
        # than the split, and is not any method's test-set AUROC.
        print(f"auroc_common covers {int(common.sum()):,} of {len(y_true):,} rows "
              f"({common.mean():.1%}); that subset is the intersection of what "
              f"every method\nwould score; it does not represent the full test distribution.")
    print(f"\nWrote {args.out_dir}/benchmark_summary.csv")
    print("Plot with figures/03_benchmark_comparison.ipynb")


if __name__ == "__main__":
    main()
