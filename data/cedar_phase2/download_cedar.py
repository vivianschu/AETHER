#!/usr/bin/env python3
"""Download CEDAR peptide-MHC and TCR data in the `data/mhcattnnet/` schema.

CEDAR is served by the same PostgREST stack as IEDB's IQ-API, so everything the
web UI's "export results" button produces is reachable programmatically:

    https://cedar-api.iedb.org/<table>?<filters>&limit=&offset=
    https://cedar-api.iedb.org/docs/swagger/

Two tables matter here:

  mhc_search  — one row per MHC ligand / binding assay (5.4M rows total).
                Carries `linear_sequence`, `mhc_allele_name`,
                `qualitative_measure`, and the antigen-class booleans the
                README's filters refer to.
  tcr_search  — one row per receptor group (85k rows), already joined to the
                epitope and the restricting allele, with CDR3α/β curated.

Usage
-----
    python download_cedar.py mhc   -o raw/mhc_search.csv
    python download_cedar.py tcr   -o raw/tcr_search.csv
    python download_cedar.py hla   -o ../HLA/hla_prot.fasta
    python download_cedar.py build -o build        # raw/ -> train-ready CSVs
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://cedar-api.iedb.org"
PAGE = 10_000  # API's maximum page size
HLA_FASTA_URL = "https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/hla_prot.fasta"

# Filters from data/CEDAR/README.md.
#
# CAUTION: `viral_antigen_bool` does NOT mean "the antigen is from a virus".
# It is CEDAR's classification of the antigen's role in a *cancer* context, so
# influenza M1 GILGFVFTL -- used as a control in tumour studies -- is filed as
# viral_antigen_bool=0, other_antigen_bool=1. Excluding viral organisms needs
# the source-organism taxonomy closure instead: NCBITaxon:10239 is Viruses, and
# `source_organism_iri_search` carries every ancestor of the source organism.
VIRUSES = "NCBITaxon:10239"

COMMON_FILTERS = {
    "host_organism_name": "like.Homo sapiens*",
    "germline_antigen_bool": "eq.0",
}

MHC_FIELDS = [
    "elution_id",
    "linear_sequence",
    "linear_sequence_length",
    "mhc_allele_name",
    "mhc_allele_resolution",
    "mhc_class",
    "qualitative_measure",
    "quantitative_measure",
    "assay_names",
    "neoantigen_bool",
    "mutation",
    "pubmed_id",
    "reference_id",
]

TCR_FIELDS = [
    "receptor_group_id",
    "receptor_type",
    "linear_sequences",
    "mhc_allele_names",
    "mhc_classes",
    "chain1_cdr3_seq",
    "chain2_cdr3_seq",
    "receptor_chain1_full_seqs",
    "receptor_chain2_full_seqs",
    "qualitative_measures",
    "assay_names",
    "neoantigen_bool",
    "source_organism_names",
    "source_organism_iri_search",
    "disease_names",
    "pubmed_ids",
    "reference_ids",
]

# qualitative_measure -> label, matching data/CEDAR/preprocess.ipynb
LABELS = {
    "Positive": 1.0,
    "Positive-High": 0.75,
    "Positive-Intermediate": 0.5,
    "Positive-Low": 0.25,
    "Negative": 0.0,
}


def fetch(table: str, params: dict[str, str], order: str, out: Path) -> int:
    """Page through `table` with `params`, writing CSV to `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out.open("w", newline="") as fh:
        writer = None
        for offset in range(0, 10_000_000, PAGE):
            q = dict(params, order=order, limit=str(PAGE), offset=str(offset))
            url = f"{API}/{table}?" + urllib.parse.urlencode(q, safe=".*{}:,")
            page = request_json(url)
            if not page:
                break
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(page[0]))
                writer.writeheader()
            for r in page:
                writer.writerow({k: serialize(v) for k, v in r.items()})
            rows += len(page)
            print(f"  {table}: {rows} rows", file=sys.stderr)
            if len(page) < PAGE:
                break
    return rows


def request_json(url: str, attempts: int = 4):
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"accept": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if i == attempts - 1:
                raise
            print(f"  retry after {exc}", file=sys.stderr)
            time.sleep(5 * (i + 1))


def serialize(v):
    return json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v


def cmd_mhc(args):
    params = dict(COMMON_FILTERS)
    params.update(
        {
            "mhc_class": f"eq.{args.mhc_class}",
            "mhc_allele_name": "like.HLA-*",
            "qualitative_measure": "not.is.null",
            "linear_sequence": "not.is.null",
            "source_organism_iri_search": f"not.cs.{{{VIRUSES}}}",
            "select": ",".join(MHC_FIELDS),
        }
    )
    n = fetch("mhc_search", params, "elution_id", Path(args.out))
    print(f"wrote {n} MHC assay rows -> {args.out}")


def cmd_tcr(args):
    params = dict(COMMON_FILTERS)
    params.update(
        {
            "host_organism_names": "cs.{\"Homo sapiens (human)\"}",
            "chain2_cdr3_seq": "not.is.null",
            "select": ",".join(TCR_FIELDS),
        }
    )
    params.pop("host_organism_name")  # tcr_search pluralizes it (array column)
    n = fetch("tcr_search", params, "receptor_group_id", Path(args.out))
    print(f"wrote {n} receptor rows -> {args.out}")


def cmd_hla(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(HLA_FASTA_URL, timeout=300) as resp, out.open("wb") as fh:
        fh.write(resp.read())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------- build step


def two_field(allele: str) -> str | None:
    """'HLA-A*02:01:01:02N' / 'HLA-A*02:01' -> 'HLA-A*02:01'."""
    allele = str(allele).strip().replace(" ", "")
    if "*" not in allele or ":" not in allele:
        return None  # 'HLA-A2', 'HLA class I' etc. -- no sequence to attach
    locus, _, fields = allele.partition("*")
    parts = fields.split(":")
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    return f"{locus}*{parts[0]}:{parts[1].rstrip('NLSCAQ')}"


def mhcattnnet_sequences(root: Path) -> dict[str, str]:
    """allele -> mhc_amino_acid, as used by the existing IEDB-derived splits."""
    seqs: dict[str, str] = {}
    for csv_path in sorted(root.glob("**/*.csv")):
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                key = two_field(row["mhc_allele"])
                if key:
                    seqs.setdefault(key, row["mhc_amino_acid"])
    return seqs


def fasta_sequences(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name, chunks = None, []
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    seqs.setdefault(name, "".join(chunks))
                # >HLA:HLA00001 A*01:01:01:01 365 bp
                field = line.split()[1] if len(line.split()) > 1 else ""
                name = two_field("HLA-" + field)
                chunks = []
            else:
                chunks.append(line.strip())
    if name:
        seqs.setdefault(name, "".join(chunks))
    return seqs


def calibrate_trim(known: dict[str, str], fasta: dict[str, str]) -> dict[str, int]:
    """Per-locus leader length, learned from alleles present in both sources.

    hla_prot.fasta carries the signal peptide; the mhcattnnet column does not,
    so a raw fasta lookup yields a sequence the model has never seen for an
    allele it has. Derive the offset empirically instead of hardcoding 24.
    """
    offsets: dict[str, list[int]] = {}
    for allele, seq in known.items():
        full = fasta.get(allele)
        if full and seq in full:
            offsets.setdefault(allele.split("*")[0], []).append(full.index(seq))
    trims = {}
    for locus, vals in offsets.items():
        trims[locus] = max(set(vals), key=vals.count)
        print(f"  {locus}: leader={trims[locus]} (n={len(vals)}, agreement="
              f"{vals.count(trims[locus]) / len(vals):.0%})", file=sys.stderr)
    return trims


def cmd_build(args):
    repo = Path(__file__).resolve().parents[2]
    raw = Path(args.raw)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    known = mhcattnnet_sequences(repo / "data" / "mhcattnnet")
    fasta = fasta_sequences(repo / "data" / "HLA" / "hla_prot.fasta")
    print(f"{len(known)} alleles from mhcattnnet, {len(fasta)} from hla_prot.fasta",
          file=sys.stderr)
    trims = calibrate_trim(known, fasta)

    def sequence_for(allele: str) -> str | None:
        key = two_field(allele)
        if not key:
            return None
        if key in known:            # reuse the exact string the model trains on
            return known[key]
        full = fasta.get(key)
        if full is None:
            return None
        return full[trims.get(key.split("*")[0], 25):]

    # --- peptide-MHC: collapse assays to one row per (peptide, allele) ---
    #
    # Phase 1 (data/mhcattnnet) already trained on IEDB. CEDAR is curated from
    # the same literature, so every pair phase 1 has seen is dropped outright
    # rather than flagged -- a phase-2 set must not re-present phase-1 rows.
    phase1 = mhcattnnet_pairs(repo / "data" / "mhcattnnet")

    pairs: dict[tuple[str, str], dict] = {}
    stats = collections.Counter()
    with (raw / "mhc_search.csv").open() as fh:
        for row in csv.DictReader(fh):
            stats["read"] += 1
            label = LABELS.get(row["qualitative_measure"])
            allele = row["mhc_allele_name"]
            key2 = two_field(allele)
            pep = row["linear_sequence"].strip().upper()
            if label is None or not pep.isalpha():
                stats["unusable_assay"] += 1
                continue
            if key2 is None:
                stats["allele_too_coarse"] += 1
                continue
            seq = sequence_for(allele)
            if seq is None:
                stats["no_sequence"] += 1
                continue
            if (pep, key2) in phase1:
                stats["overlaps_phase1"] += 1
                continue
            rec = pairs.get((pep, key2))
            if rec is None:
                pairs[(pep, key2)] = {
                    "peptide": pep, "mhc_amino_acid": seq, "mhc_allele": allele,
                    "label": label, "neoantigen": int(row["neoantigen_bool"] == "1"),
                    "n_assays": 1, "conflict": 0,
                }
            else:
                stats["merged_assay"] += 1
                if (rec["label"] > 0) != (label > 0):
                    rec["conflict"] = 1
                rec["label"] = max(rec["label"], label)
                rec["neoantigen"] |= int(row["neoantigen_bool"] == "1")
                rec["n_assays"] += 1

    # --- route each pair to its pool ------------------------------------
    #
    # train pool: measured negatives + neoantigens -- the rows that carry
    #   discriminative signal. MS ligands would otherwise swamp it.
    # ligand pool: everything else, i.e. eluted positives. Positive-only, so
    #   it is pretraining / PU signal, never a validation set.
    cols = ["peptide", "mhc_amino_acid", "bind", "mhc_allele", "bind_graded",
            "neoantigen", "n_assays", "label_conflict"]
    counts = collections.Counter()
    with (outdir / "cedar_train_pool.csv").open("w", newline="") as f_tr, \
         (outdir / "cedar_ms_ligands.csv").open("w", newline="") as f_ms:
        w_tr, w_ms = csv.writer(f_tr), csv.writer(f_ms)
        w_tr.writerow(cols)
        w_ms.writerow(cols)
        for rec in pairs.values():
            line = [rec["peptide"], rec["mhc_amino_acid"], int(rec["label"] > 0),
                    rec["mhc_allele"], rec["label"], rec["neoantigen"],
                    rec["n_assays"], rec["conflict"]]
            if rec["label"] == 0 or rec["neoantigen"]:
                w_tr.writerow(line)
                counts["train"] += 1
                counts["train_pos" if rec["label"] > 0 else "train_neg"] += 1
            else:
                w_ms.writerow(line)
                counts["ms"] += 1
            counts["conflict"] += rec["conflict"]

    print(f"\nread {stats['read']} assay rows")
    for k in ("unusable_assay", "allele_too_coarse", "no_sequence",
              "overlaps_phase1", "merged_assay"):
        print(f"  {k:18s} {stats[k]:>8}")
    print(f"\ncedar_train_pool.csv  {counts['train']:>8}  "
          f"({counts['train_neg']} negative, {counts['train_pos']} neoantigen-positive)")
    print(f"cedar_ms_ligands.csv  {counts['ms']:>8}  (positive only)")
    print(f"label conflicts resolved to positive: {counts['conflict']}")

    # --- peptide-MHC-TCR set -------------------------------------------
    #
    # Not deduplicated against phase 1: the label here is receptor recognition,
    # not peptide-MHC binding, so a shared (peptide, allele) is a different
    # example rather than a repeat. `pmhc_in_phase1` marks them so a joint
    # evaluation can still exclude them.
    tcr_counts = collections.Counter()
    with (raw / "tcr_search.csv").open() as fh, \
         (outdir / "cedar_tcr.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["peptide", "mhc_amino_acid", "bind", "mhc_allele",
                         "cdr3a", "cdr3b", "receptor_group_id", "neoantigen",
                         "viral", "source_organism", "pmhc_in_phase1"])
        seen_receptors = set()
        for row in csv.DictReader(fh):
            peps = json.loads(row["linear_sequences"] or "[]")
            alleles = json.loads(row["mhc_allele_names"] or "[]")
            measures = json.loads(row["qualitative_measures"] or "[]")
            if len(peps) != 1 or len(alleles) != 1:
                # ambiguous: the group was assayed against several epitopes or
                # the restriction was not resolved to one allele
                tcr_counts["ambiguous"] += 1
                continue
            seq = sequence_for(alleles[0])
            label = LABELS.get(measures[0] if measures else None)
            if seq is None or label is None:
                tcr_counts["unusable"] += 1
                continue
            pep = peps[0].strip().upper()
            cdr3b = row["chain2_cdr3_seq"] or ""
            dedupe_key = (pep, two_field(alleles[0]), row["chain1_cdr3_seq"] or "", cdr3b)
            if dedupe_key in seen_receptors:
                tcr_counts["duplicate"] += 1
                continue
            seen_receptors.add(dedupe_key)
            writer.writerow([pep, seq, int(label > 0), alleles[0],
                             row["chain1_cdr3_seq"] or "", cdr3b,
                             row["receptor_group_id"],
                             int(row["neoantigen_bool"] == "1"),
                             int(VIRUSES in (row.get("source_organism_iri_search") or "")),
                             "|".join(json.loads(row.get("source_organism_names") or "[]")),
                             int((pep, two_field(alleles[0])) in phase1)])
            tcr_counts["kept"] += 1
    print(f"\ncedar_tcr.csv         {tcr_counts['kept']:>8}  "
          f"({tcr_counts['ambiguous']} ambiguous, {tcr_counts['unusable']} unusable, "
          f"{tcr_counts['duplicate']} duplicate dropped)")


def mhcattnnet_pairs(root: Path) -> set[tuple[str, str | None]]:
    pairs = set()
    for csv_path in sorted(root.glob("**/*.csv")):
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                pairs.add((row["peptide"].strip().upper(), two_field(row["mhc_allele"])))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mhc", help="download mhc_search assays")
    m.add_argument("-o", "--out", default="raw/mhc_search.csv")
    m.add_argument("--mhc-class", dest="mhc_class", default="I", choices=["I", "II"])
    m.set_defaults(func=cmd_mhc)

    t = sub.add_parser("tcr", help="download tcr_search receptor groups")
    t.add_argument("-o", "--out", default="raw/tcr_search.csv")
    t.set_defaults(func=cmd_tcr)

    h = sub.add_parser("hla", help="download hla_prot.fasta from IPD-IMGT/HLA")
    h.add_argument("-o", "--out", default="../HLA/hla_prot.fasta")
    h.set_defaults(func=cmd_hla)

    b = sub.add_parser("build", help="turn raw exports into train-ready CSVs")
    b.add_argument("--raw", default="raw")
    b.add_argument("-o", "--out", default="build")
    b.set_defaults(func=cmd_build)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
