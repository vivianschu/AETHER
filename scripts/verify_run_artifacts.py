#!/usr/bin/env python
"""Verify a finished checkpoint against its validation, data and source records."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from neoantigen.provenance import file_sha256
from neoantigen.train.dataloader import DataConfig, load_splits
from neoantigen.infer.evaluate import verify_unseen_alleles


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir',required=True)
    p.add_argument('--out_json',required=True)
    p.add_argument('--require_unseen_alleles',action='store_true')
    args=p.parse_args()
    run=Path(args.run_dir)
    artifacts=json.loads((run/'artifact_manifest.json').read_text())
    for name,expected in artifacts.items():
        assert file_sha256(run/name)==expected, name
    manifest=json.loads((run/'run_manifest.json').read_text())
    manifest_hash=file_sha256(run/'run_manifest.json')
    for name,expected in manifest['source_sha256'].items():
        assert file_sha256(run/'source_snapshot'/name)==expected, name
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    metrics=json.loads((run/'test_metrics.json').read_text())
    assert checkpoint['run_manifest_sha256']==manifest_hash
    assert metrics['run_manifest_sha256']==manifest_hash
    assert metrics['checkpoint_sha256']==artifacts['best.pt']
    assert metrics['checkpoint_epoch']==checkpoint['epoch']
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    validation=[r for r in rows if r['phase']=='val']
    metric=checkpoint['args']['best_metric']
    key=f'val_{metric}'
    selection=(min if metric=='loss' else max)(validation,key=lambda r:r[key])
    assert selection['epoch']==checkpoint['epoch']
    assert checkpoint['validation_metrics'][metric]==selection[key]
    saved=checkpoint['args']
    cfg=DataConfig(**{k:saved[k] for k in
        ['data_dir','csv_url','seed','split_group','max_train_rows','max_eval_rows'] if k in saved})
    splits=dict(zip(['train','valid','test'],load_splits(cfg)))
    for name,frame in splits.items():
        assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()==manifest['split_sha256'][name], name
    unseen=verify_unseen_alleles(checkpoint,run/'best.pt',splits['test']) if args.require_unseen_alleles else None
    result=dict(checkpoint_epoch=checkpoint['epoch'],checkpoint_sha256=artifacts['best.pt'],
        run_manifest_sha256=manifest_hash,selection_metric=metric,
        selection_value=selection[key], completed_validation_epochs=[r['epoch'] for r in validation],
        artifact_hashes_verified=True,source_snapshot_hashes_verified=True,
        data_hashes_verified=True,best_epoch_selection_verified=True,
        unseen_allele_and_sequence_verified=unseen)
    out=Path(args.out_json);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
