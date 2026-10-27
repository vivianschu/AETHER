"""Inference for the trained class-I peptide:MHC binder.

    from pepmhc.predictor import Predictor
    p = Predictor()
    p.predict(["KRVTEILGF", "AANPHSFVF"], "A*02:01:01")      # -> scores
    p.predict_pairs(df.peptide, df.allele)

The model scores a peptide against the *full* MHC alpha-chain amino-acid
sequence, not an allele name, so an allele has to be resolvable to the sequence
the model saw in training; `allele_sequences` holds the 161 class-I alleles
present in the training CSVs. Alleles outside that set cannot be scored and are
reported as NaN rather than silently mapped to something else.
"""
import re
import numpy as np
import torch

from .model import PepMHCTransformer, PEPTIDE_LENGTH, MHC_LENGTH
from . import vocab

CHECKPOINT = vocab.REPO / "MHCAttnNet_dataset" / "models" / "classI" / "check.pt"
UNK, PAD = 0, 1


def normalize_allele(allele):
    """'A*01:01:01' / 'HLA-A*01:01:01:02' / 'HLA-A0101' -> 'HLA-A*01:01'.

    The HLA caller writes 3-field, prefix-less names; the model's table is
    2-field with an 'HLA-' prefix. Returns None for NA / null alleles."""
    if allele is None:
        return None
    a = str(allele).strip()
    if not a or a.upper() in {"NA", "NAN", "NONE", "-"}:
        return None
    a = re.sub(r"^HLA-", "", a, flags=re.I)
    if a.endswith("N"):          # null allele: not expressed, not presentable
        return None
    m = re.match(r"^([A-Z]+\d*)\*?(\d{2,3}):?(\d{2,3})", a)
    if not m:
        return None
    return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"


class Predictor:
    # The trained graph attends across the batch dimension (see model.py), so a
    # score is only reproducible at the batch size the model was trained with.
    TRAIN_BATCH_SIZE = 32          # config.batch_size

    def __init__(self, checkpoint=CHECKPOINT, activation="mish", nhead=10, seq_first=False,
                 relu_on_output=False,
                 batch_size=TRAIN_BATCH_SIZE, device="cpu"):
        self.pep_itos, self.mhc_itos, self.allele_sequences = vocab.load()
        self.pep_stoi = {t: i for i, t in enumerate(self.pep_itos)}
        self.mhc_stoi = {t: i for i, t in enumerate(self.mhc_itos)}
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.model = PepMHCTransformer(len(self.pep_itos), len(self.mhc_itos),
                                       activation=activation, nhead=nhead, seq_first=seq_first,
                                       relu_on_output=relu_on_output)
        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        # MHC sequences repeat across rows; encode each one once
        self._mhc_cache = {}

    # -- encoding ---------------------------------------------------------
    def _encode(self, tokens, stoi, length):
        """torchtext Field.pad: truncate to `length`, then right-pad."""
        ids = [stoi.get(t, UNK) for t in tokens[:length]]
        return ids + [PAD] * (length - len(ids))

    def encode_peptide(self, peptide):
        return self._encode(vocab.ngrams(peptide, vocab.PEP_N),
                            self.pep_stoi, PEPTIDE_LENGTH)

    def encode_allele(self, allele):
        """allele (any notation) -> MHC token ids, or None if unsupported."""
        key = normalize_allele(allele)
        if key not in self.allele_sequences:
            return None
        if key not in self._mhc_cache:
            self._mhc_cache[key] = self._encode(
                vocab.ngrams(self.allele_sequences[key], vocab.MHC_N),
                self.mhc_stoi, MHC_LENGTH)
        return self._mhc_cache[key]

    def supported_alleles(self):
        return sorted(self.allele_sequences)

    # -- scoring ----------------------------------------------------------
    @torch.inference_mode()
    def predict_pairs(self, peptides, alleles, batch_size=None):
        """Score peptide[i] against allele[i]. Returns float32 array of
        P(bind) in [0,1]; NaN where the peptide or allele is unusable."""
        peptides = list(peptides)
        alleles = list(alleles)
        if len(peptides) != len(alleles):
            raise ValueError("peptides and alleles must be the same length")

        out = np.full(len(peptides), np.nan, dtype=np.float32)
        keep, pep_ids, mhc_ids = [], [], []
        valid_aa = set(self.pep_itos[2:])
        for i, (pep, allele) in enumerate(zip(peptides, alleles)):
            pep = "" if pep is None else str(pep).strip().upper()
            mhc = self.encode_allele(allele)
            if not pep or mhc is None or not set(pep) <= valid_aa:
                continue
            keep.append(i)
            pep_ids.append(self.encode_peptide(pep))
            mhc_ids.append(mhc)
        if not keep:
            return out

        batch_size = batch_size or self.batch_size
        keys = [normalize_allele(alleles[i]) for i in keep]
        scores = []
        arm_cache = {}
        for s in range(0, len(keep), batch_size):
            p = torch.tensor(pep_ids[s:s + batch_size], device=self.device)
            m_rows = mhc_ids[s:s + batch_size]
            batch_keys = keys[s:s + batch_size]
            # A full batch of one allele gives an MHC arm that depends only on
            # that allele and the batch length, so it can be reused. Pairs are
            # normally fed allele-major, which makes almost every batch uniform.
            ck = (batch_keys[0], len(m_rows))
            if len(set(batch_keys)) == 1 and ck in arm_cache:
                mhc_out = arm_cache[ck]
            else:
                mhc_out = self.model.mhc_arm(
                    torch.tensor(m_rows, device=self.device))
                if len(set(batch_keys)) == 1:
                    arm_cache[ck] = mhc_out
            logits = self.model.head(self.model.peptide_arm(p), mhc_out)
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        out[np.asarray(keep)] = np.concatenate(scores)
        return out

    def predict(self, peptides, allele, **kw):
        """Score many peptides against one allele."""
        peptides = list(peptides)
        return self.predict_pairs(peptides, [allele] * len(peptides), **kw)
