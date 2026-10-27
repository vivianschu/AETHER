#!/usr/bin/env python
"""Collect complete clean runs, compare aligned predictions, and persist small records.

Run after the training and external scoring commands finish. --wait optionally
waits for their artifacts, verifying successful complete outputs rather than
using any historical AETHER predictions.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from neoantigen.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {
    'peptide': ('clean_peptide_20261027', 'benchmark_external_20261027',
                'probe_peptide_20261027', 'classI_peptide_split'),
    'heldout_allele': ('heldout_allele_sequence_20261027', 'heldout_sequence_external_20261027',
                       'probe_heldout_sequence_20261027', 'classI_allele_sequence_split'),
}
METHODS = ['majority', 'allele_prior', 'allele_length_prior', 'pssm', 'blosum_lr',
           'netmhcpan', 'bigmhc', 'mhcflurry']


def finalize(protocol, result_root, wait):
    run_name, external_name, probe_name, data_name = PROTOCOLS[protocol]
    run, external, probe = [ROOT/'outputs'/s for s in [run_name, external_name, probe_name]]
    required = [run/'artifact_manifest.json', external/'benchmark_meta.json',
                probe/'esm2_probe_predictions.csv', external/'mhcflurry_predictions.csv']
    deadline = time.monotonic() + wait
    while any(not p.exists() for p in required):
        if time.monotonic() >= deadline:
            raise RuntimeError(f'{protocol}: missing {[str(p) for p in required if not p.exists()]}')
        time.sleep(10)
    # The final artifact manifest is written only after the test pass has completed.
    for name, sha in json.loads((run/'artifact_manifest.json').read_text()).items():
        if file_sha256(run/name) != sha:
            raise RuntimeError(f'{run/name}: artifact hash mismatch')
    manifest = json.loads((run/'run_manifest.json').read_text())
    assert all(v == 0 for v in manifest['split_audit']['overlaps'].values())
    for p in ['best.pt', 'test_metrics.json', 'test_predictions.csv']:
        assert (run/p).exists(), p
    inputs = {method: external/f'{method}_predictions.csv' for method in METHODS}
    inputs.update(aether=run/'test_predictions.csv', esm2_probe=probe/'esm2_probe_predictions.csv')
    for path in inputs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    benchmark = ROOT/'outputs'/f'final_{protocol}_20261027'
    cmd = [sys.executable, '-m', 'neoantigen.benchmark.run', '--data_dir',
           str(ROOT/'data/mhcattnnet'/data_name), '--split_group',
           'allele' if protocol == 'heldout_allele' else 'peptide',
           '--out_dir', str(benchmark), '--methods', 'majority', '--include']
    cmd += [f'{name}={path}' for name,path in inputs.items()]
    subprocess.run(cmd, check=True, cwd=ROOT)
    summary = pd.read_csv(benchmark/'benchmark_summary.csv')
    if set(summary.method) != set(inputs):
        raise RuntimeError('Final benchmark has missing methods')
    dest = result_root/protocol
    dest.mkdir(parents=True, exist_ok=True)
    for p in ['config.json', 'run_manifest.json', 'test_metrics.json',
              'metrics.jsonl', 'artifact_manifest.json', 'test_per_allele.csv']:
        shutil.copyfile(run/p, dest/p)
    for p in benchmark.glob('*'):
        if p.name.endswith('_per_allele.csv') or p.name.endswith('_per_allele_common.csv') or p.name in ['benchmark_summary.csv','benchmark_meta.json']:
            shutil.copyfile(p, dest/p.name)
    comparisons = {}
    a = pd.read_csv(benchmark/'aether_per_allele_common.csv')
    for method in ['mhcflurry','bigmhc','netmhcpan','pssm']:
        b = pd.read_csv(benchmark/f'{method}_per_allele_common.csv')
        pair = a[['mhc_allele','n_scored','auroc']].merge(
            b[['mhc_allele','auroc']], on='mhc_allele', suffixes=('_aether',f'_{method}'))
        pair = pair.loc[(pair.n_scored >= 50) & pair.auroc_aether.notna() & pair[f'auroc_{method}'].notna()].copy()
        pair['delta'] = pair.auroc_aether - pair[f'auroc_{method}']
        pair.to_csv(dest/f'aether_vs_{method}_per_allele.csv', index=False)
        comparisons[method] = dict(n_alleles=len(pair), aether_wins=int((pair.delta>0).sum()),
            ties=int((pair.delta==0).sum()), median_delta=float(pair.delta.median()),
            weighted_delta=float(np.average(pair.delta, weights=pair.n_scored)),
            support_on_aether_winning_alleles=float(pair.loc[pair.delta>0,'n_scored'].sum()/pair.n_scored.sum()))
    metadata = dict(checkpoint=str((run/'best.pt').relative_to(ROOT)),
        checkpoint_sha256=file_sha256(run/'best.pt'), run_dir=str(run.relative_to(ROOT)),
        benchmark_dir=str(benchmark.relative_to(ROOT)), comparisons=comparisons,
        external_prediction_sha256={n:file_sha256(p) for n,p in inputs.items()},
        external_run_metadata=json.loads((external/'benchmark_meta.json').read_text()),
        probe_run_metadata=json.loads((probe/'benchmark_meta.json').read_text()),
        external_tools=dict(mhcflurry='2.3.16 presentation',netmhcpan='4.2d standard EL',bigmhc='EL ensemble'),
        notes=['Benchmark within-allele aggregates require >=50 common scored rows and both labels.',
               'Training validation/checkpoint selection uses all alleles with both labels.',
               'ESM2 probe uses 100,000 training rows; AETHER and classical baselines use the full training split.',
               'External pretraining overlap with IEDB/CEDAR has not been excluded.'])
    (dest/'record.json').write_text(json.dumps(metadata,indent=2))
    return summary, metadata


def plot(tables, result_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    for ax,(protocol,table) in zip(axes,tables.items()):
        table = table.sort_values('within_allele_auroc_common')
        for i,row in enumerate(table.itertuples()):
            ax.plot([row.within_allele_auroc_common,row.auroc_common],[i,i], color='#b9c1cc',lw=2)
        ax.scatter(table.auroc_common,np.arange(len(table)),label='Pooled AUROC',color='#8a99ad',s=30)
        ax.scatter(table.within_allele_auroc_common,np.arange(len(table)),label='Within-allele AUROC',color='#2257a2',s=35)
        ax.set_yticks(np.arange(len(table)),table.method)
        ax.set_xlim(.45,1.0)
        ax.set_xlabel('AUROC on rows scored by every method')
        title = 'Peptide-disjoint split' if protocol=='peptide' else 'Allele / MHC-sequence-disjoint split'
        ax.set_title(f'{title}\nCommon rows: {int(table.n_common.iloc[0]):,}; eligible within-allele rows: {int(table.within_allele_n_common.iloc[0]):,}', fontsize=10)
        ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='upper left',fontsize=8)
    fig.savefig(result_root/'benchmark_comparison.png',dpi=180)
    plt.close(fig)


def write_report(tables, records, result_root):
    lines=['# Clean AETHER benchmarks — 2026-10-27','',
        'Clean 10-epoch runs use the frozen ESM2-650M backbone and the existing two-layer ',
        'interaction head. Each run uses four GPUs with total batch size 64 and seed 1337.',
        'Checkpoints are selected by within-allele validation AUROC, independently of the test set.',
        'Source snapshots, checkpoints and predictions remain together in the recorded run directories.',
        'Small metrics and provenance files are retained here for version control.','']
    peptide=tables['peptide'].set_index('method')
    heldout=tables['heldout_allele'].set_index('method')
    lines += [
        f"AETHER narrowly leads the peptide-disjoint comparison: within-allele AUROC {peptide.loc['aether','within_allele_auroc_common']:.3f}, against BigMHC {peptide.loc['bigmhc','within_allele_auroc_common']:.3f}. The difference is small and comes from one seed.", '',
        f"On the separately trained allele/MHC-sequence-disjoint protocol, AETHER reaches {heldout.loc['aether','within_allele_auroc_common']:.3f}, below BigMHC {heldout.loc['bigmhc','within_allele_auroc_common']:.3f}, MHCflurry {heldout.loc['mhcflurry','within_allele_auroc_common']:.3f}, and NetMHCpan {heldout.loc['netmhcpan','within_allele_auroc_common']:.3f}. These results do not support an overall pan-allele leadership claim. External tools may have seen these alleles or records during pretraining.", '',
        '![Pooled and within-allele AUROC](benchmark_comparison.png)', '']
    for protocol,table in tables.items():
        record=records[protocol]
        epoch=json.loads((result_root/protocol/'test_metrics.json').read_text())['checkpoint_epoch']
        lines += [f'## {"Peptide-disjoint" if protocol == "peptide" else "Held-out alleles and MHC sequences"}', '',
            f'Checkpoint: `{record["checkpoint"]}` (selected epoch {epoch} after 10 training epochs; SHA-256 `{record["checkpoint_sha256"]}`).', '',
            f'Common scored population: {int(table.n_common.iloc[0]):,} rows. '
            f'Within-allele AUROC: {int(table.within_allele_n_alleles_common.iloc[0])} eligible alleles, '
            f'{int(table.within_allele_n_common.iloc[0]):,} rows.', '',
            '| Method | Pooled AUROC | Within-allele AUROC | AUPRC | PPV@100 | Coverage |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for row in table.itertuples():
            lines += [f'| {row.method} | {row.auroc_common:.3f} | {row.within_allele_auroc_common:.3f} | {row.auprc_common:.3f} | {row.ppv_at_100_common:.3f} | {row.coverage:.1%} |']
        lines += ['', 'Pooled AUROC, AUPRC and PPV@100 use all common scored rows. Within-allele AUROC uses the eligible allele population counted above. Coverage is against the full test split.', '']
        for method,c in record['comparisons'].items():
            lines += [f'AETHER exceeds {method} on {c["aether_wins"]}/{c["n_alleles"]} eligible alleles '
                f'({c["ties"]} ties); median allele AUROC difference {c["median_delta"]:+.3f}, '
                f'support-weighted difference {c["weighted_delta"]:+.3f}.']
        lines += ['']
    lines += ['## Interpretation and limits','',
        '- Pooled AUROC can reflect allele separation. Within-allele AUROC is support weighted over alleles with both labels and at least 50 common scored rows; it is not a macro average.',
        '- The complete peptide test split has 35,339 rows across 146 alleles. Only 66 alleles qualify for the default within-allele aggregate; the original statement that all 35,339 rows were across 66 alleles conflates these populations.',
        '- AUPRC is average precision. PPV@100 uses min(100, scored rows), with expected credit for ties at the cutoff. Per-allele tables expose allele-prior effects on triage.',
        '- The ESM2 linear probe fits 100,000 training rows; its gap to AETHER also reflects training-set size.',
        '- The held-out-allele model is trained separately with both allele names and encoded MHC sequences disjoint across splits. Peptides may recur across alleles. External pretrained tools are not known to have held these alleles out.',
        '- The peptide partition is frozen in peptide_split_assignments.csv.gz. Use neoantigen.data.regroup --assignments_file to restore its membership and row order; peptide_split_restoration.json verifies all three canonical hashes.',
        '- The roughly 0.32 pooled-versus-within gap belongs to the allele-prior baseline, not a universal correction for all models. The two protocols have different test populations, so their score difference does not isolate the causal effect of holding alleles out.',
        '- NetMHCpan 4.2d uses standard EL scores, without neoepitope fine-tuning; BigMHC and MHCflurry use presentation models. Their training data may overlap IEDB/CEDAR. These results do not prove independent prospective performance.',
        '- These are single-seed runs. High-support alleles dominate the weighted result; per-allele wins/medians should accompany aggregate claims.',
        '- CEDAR label noise, limited independent negatives, and overlap with phase-one data remain limitations. No claim of validated immunogenicity, presentation or TCR prediction follows from this binding benchmark.',
        '- The working tree already has an optional TCR head; it was not retrained in these binding experiments. Attention export now provides existing interaction and pooling weights, with no causal interpretation asserted.', '']
    lines += ['## Reproduction and validation', '',
        'The peptide run uses 279,527 training rows; the allele run uses 274,923. Input file hashes, cleanup counts and split checks are recorded in [data_audit.json](data_audit.json).', '',
        'The original row split had 63.1% unique-peptide overlap, 35.8% unique-pair overlap, and 27.5% pooled exact duplicates. The peptide split has zero on all three. The allele split has no shared allele names or encoded MHC sequences; shared peptides across different alleles are permitted.', '',
        'Restore the exact peptide partition and row order from the AETHER root into a fresh directory:', '',
        '```bash',
        'python -m neoantigen.data.regroup --in_dir data/mhcattnnet/classI --out_dir outputs/restored_split --assignments_file results/20261027/peptide_split_assignments.csv.gz',
        '```', '',
        '[Restoration verification](peptide_split_restoration.json) matches the training run’s canonical hashes for train, validation and test. Random regrouping defaults create a different valid partition of the same cleaned corpus.', '',
        'Checkpoint, source snapshot, data hash and best-epoch checks are recorded in [peptide verification](peptide/verification.json) and [allele verification](heldout_allele/verification.json). Standalone allele evaluation also verifies against the checkpoint’s actual training data. Runtime events are retained in [runtime_events.json](runtime_events.json).', '',
        '[Attention example](peptide/attention_example.png), [token metadata](peptide/attention_example.json), and [full attention arrays](peptide/attention_example.npz) come from the clean peptide checkpoint. The example uses CPU float32 inference; benchmark scoring used the recorded GPU/AMP configuration. The weights are descriptive attention, not causal attribution.', '',
        'NetMHCpan/BigMHC source and weight hashes, MHCflurry model hashes, the MHCflurry environment, and its model-release download record are retained alongside this report. Known contradictory pairs are removed by regrouping; rows explicitly marked label_conflict by the CEDAR builder are excluded during loading. These steps do not resolve all label noise.', '']
    (result_root/'README.md').write_text('\n'.join(line.rstrip() for line in lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',type=int,default=0,help='Seconds to wait for complete runs')
    parser.add_argument('--result_dir',default='results/20261027')
    args=parser.parse_args()
    result_root=ROOT/args.result_dir
    tables,records={},{}
    for protocol in PROTOCOLS:
        tables[protocol],records[protocol]=finalize(protocol,result_root,args.wait)
    plot(tables,result_root)
    write_report(tables,records,result_root)
    print('Saved audited records to',result_root,flush=True)

if __name__=='__main__':
    main()
