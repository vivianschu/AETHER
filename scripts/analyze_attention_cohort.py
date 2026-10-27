#!/usr/bin/env python
"""Export 9-mer attention cohorts and exhaustive single-substitution model sensitivity.

Runs the existing checkpoint in float32; never trains or changes model weights.
Plotting is in figures/05_results_20261027.ipynb.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neoantigen.infer.explain import AttentionCapture
from neoantigen.models.config import ModelConfig
from neoantigen.models.transformer import NeoantigenModel
from neoantigen.provenance import file_sha256
from neoantigen.train.dataloader import prepare_df

ROOT = Path(__file__).resolve().parents[1]
AA = 'ACDEFGHIKLMNPQRSTVWY'


def select_cohort(data, seed=1337, n_random=2048, n_alleles=16, per_label=32):
    """Population-random cohort plus a separate balanced allele/label comparison."""
    nine = data.loc[data.epitope.str.len().eq(9)].copy()
    random = nine.sample(min(n_random, len(nine)), random_state=seed)
    counts = nine.groupby(['mhc_allele', 'y']).size().unstack(fill_value=0)
    eligible = counts.loc[counts.min(axis=1) >= per_label]
    alleles = eligible.sum(axis=1).sort_values(ascending=False, kind='stable').head(n_alleles).index.tolist()
    balanced = pd.concat([g.sample(per_label, random_state=seed)
        for _, g in nine.loc[nine.mhc_allele.isin(alleles)].groupby(['mhc_allele', 'y'])])
    cohort = pd.concat([random, balanced]).drop_duplicates('input_row').sort_values('input_row')
    cohort['population_sample'] = cohort.input_row.isin(random.input_row)
    cohort['balanced_sample'] = cohort.input_row.isin(balanced.input_row)
    return cohort.reset_index(drop=True), alleles


@torch.inference_mode()
def run(args):
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = ROOT / args.input_csv
    data = prepare_df(pd.read_csv(source))
    data['input_row'] = np.arange(len(data))
    pred_path = ROOT / 'outputs/clean_peptide_20261027/test_predictions.csv'
    pred = pd.read_csv(pred_path)
    assert len(pred) == len(data) and np.array_equal(pred.peptide, data.epitope)
    assert np.array_equal(pred.mhc_allele, data.mhc_allele) and np.array_equal(pred.y_true, data.y)
    data['benchmark_probability'] = pred.y_pred_prob
    cohort, alleles = select_cohort(data, args.seed, args.n_random)
    cohort.to_csv(out / 'cohort.csv', index=False)
    ckpt_path = ROOT / args.ckpt
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ModelConfig.from_dict(ckpt['model_cfg'])
    cfg.freeze_backbone = True
    model = NeoantigenModel(cfg).to(args.device).eval()
    model.load_state_dict(ckpt['model_state'])
    del ckpt
    # Match uncached float32 inference, rather than quantizing cached embeddings.
    model.encoder._cache_dtype = torch.float32
    model.precompute_mhc(cohort.mhc_sequence.tolist(), batch_size=4)
    for sequence in cohort.epitope:
        tokens = model.encoder.tokenizer.convert_ids_to_tokens(model.encoder.tokenizer(sequence)['input_ids'])
        assert tokens == ['<cls>', *sequence, '<eos>'], tokens
    capture = AttentionCapture(model)
    layer_count = len(model.layers)
    rows, arrays = [], {}
    example_ids = []
    for allele in alleles[:6]:
        for label in [1, 0]:
            g = cohort.loc[cohort.balanced_sample & cohort.mhc_allele.eq(allele) & cohort.y.eq(label)]
            example_ids.append(int(g.sort_values('input_row').iloc[0].input_row))
    for start in range(0, len(cohort), args.batch_size):
        batch = cohort.iloc[start:start+args.batch_size]
        capture.weights.clear()
        logits = model(batch.epitope.tolist(), batch.mhc_sequence.tolist()).cpu().numpy()
        w = capture.weights
        # Softmax normalization checks include special tokens and exclude padding.
        assert np.allclose(w['pep_pool'].sum(-1), 1, atol=2e-5)
        for name, value in w.items():
            if name.endswith('.attn'):
                assert np.allclose(value.sum(-1), 1, atol=2e-5), name
        for j, (_, row) in enumerate(batch.iterrows()):
            m = len(row.mhc_sequence)
            record = row.to_dict()
            record.update(logit=float(logits[j]), probability=float(torch.sigmoid(torch.tensor(logits[j]))))
            rows.append(record)
            metrics = {'pool_raw': w['pep_pool'][j], 'mhc_pool_special': w['mhc_pool'][j, [0, m+1]]}
            for layer in range(layer_count):
                # Peptide is the key axis here: incoming attention can differ by P1-P9.
                metrics[f'incoming_l{layer+1}_raw'] = w[f'layers.{layer}.mhc_to_pep.attn'][j, 1:m+1, :].mean(0)
                metrics[f'self_l{layer+1}_raw'] = w[f'layers.{layer}.pep_self.attn'][j, 1:10, :].mean(0)
            for key, value in metrics.items():
                arrays.setdefault(key, []).append(value)
            if row.input_row in example_ids:
                raw = {}
                for key, value in w.items():
                    if value.ndim == 2:
                        raw[key] = value[j, :11 if key == 'pep_pool' else m+2]
                    else:
                        q = 11 if '.pep_' in key else m+2
                        k = m+2 if '.pep_to_mhc.' in key or '.mhc_self.' in key else 11
                        raw[key] = value[j, :q, :k]
                np.savez_compressed(out / f'example_{int(row.input_row)}.npz', **raw)
        print(f'Attention {min(start+args.batch_size,len(cohort))}/{len(cohort)}', flush=True)
    capture.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'cohort.csv', index=False)
    np.savez_compressed(out / 'position_weights.npz', **{k: np.stack(v) for k,v in arrays.items()})
    # Random fixed backgrounds, independent of model score; 4 per label for 8 alleles.
    backgrounds = pd.concat([g.sample(4, random_state=args.seed)
        for _, g in frame.loc[frame.balanced_sample & frame.mhc_allele.isin(alleles[:8])].groupby(['mhc_allele','y'])])
    mutation_rows = []
    for i, row in enumerate(backgrounds.itertuples()):
        variants = [(p, aa, row.epitope[:p]+aa+row.epitope[p+1:])
                    for p in range(9) for aa in AA if aa != row.epitope[p]]
        values = []
        for start in range(0, len(variants), args.batch_size):
            peptides = [v[2] for v in variants[start:start+args.batch_size]]
            values.extend(model(peptides, [row.mhc_sequence]*len(peptides)).cpu().tolist())
        for (pos, aa, peptide), logit in zip(variants, values):
            prob = float(torch.sigmoid(torch.tensor(logit)))
            mutation_rows.append(dict(input_row=row.input_row, mhc_allele=row.mhc_allele,
                peptide=row.epitope, y=row.y, position=pos+1, original_aa=row.epitope[pos],
                mutant_aa=aa, mutant_peptide=peptide, original_logit=row.logit,
                mutant_logit=logit, delta_logit=logit-row.logit,
                original_probability=row.probability, mutant_probability=prob,
                delta_probability=prob-row.probability))
        print(f'Substitution scans {i+1}/{len(backgrounds)}', flush=True)
    mutations = pd.DataFrame(mutation_rows)
    assert len(mutations) == 64*9*19
    assert np.isfinite(mutations.select_dtypes('number')).all().all()
    mutations.to_csv(out / 'substitutions.csv.gz', index=False)
    # One uncached direct comparison validates the float32 embedding cache.
    first = frame.iloc[0]
    cached = model([first.epitope], [first.mhc_sequence]).item()
    model.encoder.invalidate_cache()
    direct = model([first.epitope], [first.mhc_sequence]).item()
    assert abs(cached-direct) < 2e-4, (cached, direct)
    manifest = dict(seed=args.seed, checkpoint=str(ckpt_path.relative_to(ROOT)),
        checkpoint_sha256=file_sha256(ckpt_path), input_csv=str(source.relative_to(ROOT)),
        input_sha256=file_sha256(source), benchmark_prediction_sha256=file_sha256(pred_path),
        source_script_sha256=file_sha256(Path(__file__)), device=args.device,
        torch_version=torch.__version__, numpy_version=np.__version__,
        precision='float32, no AMP or TF32; MHC embedding cache float32',
        test_rows=len(data), nine_mer_rows=int(data.epitope.str.len().eq(9).sum()),
        cohort_rows=len(frame), unique_peptides=int(frame.epitope.nunique()),
        population_sample=int(frame.population_sample.sum()), balanced_sample=int(frame.balanced_sample.sum()),
        balanced_alleles=alleles, example_input_rows=example_ids,
        mutation_backgrounds=len(backgrounds), mutation_rows=len(mutations),
        cached_direct_logit_difference=abs(cached-direct),
        benchmark_probability_difference_max=float((frame.probability-frame.benchmark_probability).abs().max()),
        benchmark_probability_difference_median=float((frame.probability-frame.benchmark_probability).abs().median()),
        definitions={'pool_raw':'Peptide pooling over CLS, P1-P9, EOS',
          'incoming_raw':'MHC-to-peptide attention averaged over real MHC residue queries, head averaged',
          'self_raw':'Peptide self-attention averaged over P1-P9 queries, head averaged',
          'residue_normalized':'P1-P9 weights divided by their sum, excluding CLS/EOS',
          'mutation':'All 19 non-native amino acids at each position; delta = mutant - original logit'},
        limitations=['Descriptive head-averaged attention; no physical contact or causal biological attribution.',
          'Balanced cohort uses high-support alleles and 32 examples per label; not population prevalence.',
          'Mutations are computational perturbations, not experimental binding or TCR measurements.'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', default='outputs/clean_peptide_20261027/best.pt')
    p.add_argument('--input_csv', default='data/mhcattnnet/classI_peptide_split/test.csv')
    p.add_argument('--out_dir', default='results/20261027/peptide/attention_cohort')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_random', type=int, default=2048)
    p.add_argument('--seed', type=int, default=1337)
    run(p.parse_args())
