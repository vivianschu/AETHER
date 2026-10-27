"""Rebuild the two torchtext vocabularies the checkpoint was trained with, and
the allele -> MHC amino-acid sequence table.

Nothing about the vocabularies was saved alongside check.pt, and they cannot be
read back out of the embedding matrices (those were fine-tuned, so they no
longer match the n-gram vector files they were initialised from). They are
instead reconstructed from the training CSVs, which is exact: torchtext.legacy
builds a vocab by listing the counter alphabetically and then *stably* sorting
it by frequency, descending, behind the specials.

The reconstruction is self-checking -- it must land on 23 peptide and 365 MHC
tokens, the sizes baked into the checkpoint, and the peptide order must match
the one recorded in MHCAttnNet_ft/src/data_loader.py ("L (2), V (3), A (4)
....X (22)").
"""
import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "MHCAttnNet_dataset" / "classI"
ASSETS = Path(__file__).resolve().parent / "assets"

PEP_N, MHC_N = 1, 3
SPECIALS = ("<unk>", "<pad>")
EXPECTED = {"peptide": 23, "mhc": 365}
# the order data_loader.py documents, used as an independent check
EXPECTED_PEPTIDE_ITOS = list("LVASIFRETKPYGDQNHMWCX")

csv.field_size_limit(10 ** 9)


def ngrams(seq, n):
    """Non-overlapping n-grams; the trailing remainder is dropped, as in
    data_loader.tokenize_pep / tokenize_mhc."""
    return list(map("".join, zip(*[iter(seq)] * n)))


def _order(counter):
    items = sorted(counter.items(), key=lambda t: t[0])
    items.sort(key=lambda t: t[1], reverse=True)
    return list(SPECIALS) + [w for w, _ in items]


def build(splits=("train.csv", "val.csv")):
    """build_vocab() was called on the train and val splits only."""
    pep_c, mhc_c, mhc_seq_c = Counter(), Counter(), Counter()
    allele_seq = {}
    for fn in splits:
        with open(DATASET / fn, newline="") as f:
            r = csv.reader(f)
            next(r)
            for peptide, mhc_seq, _bind, allele in r:
                pep_c.update(ngrams(peptide, PEP_N))
                mhc_seq_c[mhc_seq] += 1           # few unique sequences, many rows
                allele_seq.setdefault(allele, mhc_seq)
    for seq, k in mhc_seq_c.items():
        for t in ngrams(seq, MHC_N):
            mhc_c[t] += k

    pep_itos, mhc_itos = _order(pep_c), _order(mhc_c)
    if len(pep_itos) != EXPECTED["peptide"] or len(mhc_itos) != EXPECTED["mhc"]:
        raise RuntimeError(
            f"vocab sizes {len(pep_itos)}/{len(mhc_itos)} do not match the "
            f"checkpoint's {EXPECTED['peptide']}/{EXPECTED['mhc']}")
    if pep_itos[len(SPECIALS):] != EXPECTED_PEPTIDE_ITOS:
        raise RuntimeError(f"peptide vocab order changed: {pep_itos}")
    return pep_itos, mhc_itos, allele_seq


def load(rebuild=False):
    """Cached accessor. Returns (pep_itos, mhc_itos, allele_seq)."""
    cache = ASSETS / "vocab.json"
    if cache.exists() and not rebuild:
        d = json.loads(cache.read_text())
        return d["peptide_itos"], d["mhc_itos"], d["allele_sequences"]
    pep_itos, mhc_itos, allele_seq = build()
    ASSETS.mkdir(exist_ok=True)
    cache.write_text(json.dumps(
        {"peptide_itos": pep_itos, "mhc_itos": mhc_itos,
         "allele_sequences": allele_seq}))
    return pep_itos, mhc_itos, allele_seq


if __name__ == "__main__":
    pep, mhc, alleles = load(rebuild=True)
    print(f"peptide vocab {len(pep)}, mhc vocab {len(mhc)}, "
          f"{len(alleles)} alleles with sequences")
    print("peptide itos:", pep)
    print("example alleles:", sorted(alleles)[:8])
