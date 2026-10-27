"""Export existing attention weights without changing a checkpoint or its output.

Weights describe model attention, not causal attribution or experimental evidence.
Each row writes an NPZ of head-averaged interaction attention and pool weights,
plus JSON with input sequences, token axes, score and checkpoint identity.
"""
import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from neoantigen.models.config import ModelConfig
from neoantigen.models.pooling import AttentionPool
from neoantigen.models.transformer import NeoantigenModel
from neoantigen.provenance import file_sha256
from neoantigen.train.dataloader import resolve_sequence_columns


class AttentionCapture:
    """Observe the weights already calculated by the interaction modules."""
    def __init__(self, model):
        self.weights = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                def capture(module, inputs, output, key=name):
                    if output[1] is None:
                        raise ValueError(f"{key} did not return attention weights")
                    self.weights[key] = output[1].detach().float().cpu().numpy()
                self.handles.append(module.register_forward_hook(capture))
            elif isinstance(module, AttentionPool):
                def capture_pool(module, inputs, output, key=name):
                    x, mask = inputs
                    with torch.no_grad(), torch.amp.autocast(x.device.type, enabled=False):
                        logits = module.w(x.float()).squeeze(-1).masked_fill(mask == 0, -float('inf'))
                        self.weights[key] = torch.softmax(logits, -1).nan_to_num(0).cpu().numpy()
                self.handles.append(module.register_forward_hook(capture_pool))

    def close(self):
        for handle in self.handles:
            handle.remove()


def plot_attention(weights, metadata, path):
    """Readable view of peptide pooling and the final peptide-to-MHC attention."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [key for key in weights if key.endswith('.pep_to_mhc.attn')]
    if not names or 'pep_pool' not in weights:
        return
    attention = weights[names[-1]][0]
    selected = np.argsort(attention.mean(0), kind='stable')[-20:]
    selected.sort()
    fig, axes = plt.subplots(2,1,figsize=(12,6),layout='constrained')
    peptide = metadata['peptide_tokens']
    axes[0].bar(np.arange(len(peptide)),weights['pep_pool'][0],color='#2257a2')
    axes[0].set_xticks(np.arange(len(peptide)),peptide)
    axes[0].set_ylabel('Pooling weight')
    axes[0].set_title('Learned peptide pooling')
    im = axes[1].imshow(attention[:,selected],aspect='auto',cmap='Blues',vmin=0)
    axes[1].set_yticks(np.arange(len(peptide)),peptide)
    axes[1].set_xticks(np.arange(len(selected)),
        [f"{i}:{metadata['mhc_tokens'][i]}" for i in selected],rotation=45,ha='right')
    axes[1].set_xlabel('Top 20 MHC token positions by mean attention (0 = CLS)')
    axes[1].set_ylabel('Peptide token')
    axes[1].set_title('Final peptide-to-MHC attention, averaged across heads')
    fig.colorbar(im,ax=axes[1],label='Attention weight')
    fig.suptitle(f"{metadata['peptide']} | score {metadata['y_pred_prob']:.3f} | descriptive attention, not causal attribution")
    fig.savefig(path,dpi=160)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--max_rows', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--plot', action='store_true', help='Also save a readable attention figure per row')
    args = parser.parse_args()
    if args.max_rows < 1:
        parser.error('--max_rows must be positive')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ModelConfig.from_dict(ckpt['model_cfg'])
    cfg.freeze_backbone = True
    model = NeoantigenModel(cfg).to(args.device).eval()
    model.load_state_dict(ckpt['model_state'])
    df = pd.read_csv(args.input_csv)
    df['input_row'] = np.arange(len(df))
    df = resolve_sequence_columns(df).head(args.max_rows)
    checkpoint_hash = file_sha256(args.ckpt)
    capture = AttentionCapture(model)
    try:
        for i, row in df.iterrows():
            capture.weights.clear()
            ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if args.amp and str(args.device).startswith('cuda') else contextlib.nullcontext()
            with ctx:
                probability = float(model([row.epitope], [row.mhc_sequence]).sigmoid().item())
            def tokens(sequence):
                encoded = model.encoder.tokenizer(sequence)['input_ids']
                return model.encoder.tokenizer.convert_ids_to_tokens(encoded)
            metadata = dict(input_row=int(row.input_row), peptide=row.epitope,
                mhc_sequence=row.mhc_sequence, mhc_allele=row.get('mhc_allele'),
                y_pred_prob=probability, checkpoint_sha256=checkpoint_hash,
                device=args.device, amp=bool(args.amp and str(args.device).startswith('cuda')),
                run_manifest_sha256=ckpt.get('run_manifest_sha256'),
                peptide_tokens=tokens(row.epitope), mhc_tokens=tokens(row.mhc_sequence),
                weights={name: list(value.shape) for name,value in capture.weights.items()},
                interpretation='Head-averaged attention and pooling weights; descriptive, not causal attribution.')
            np.savez_compressed(out_dir/f'row_{i}.npz', **capture.weights)
            (out_dir/f'row_{i}.json').write_text(json.dumps(metadata, indent=2))
            if args.plot:
                plot_attention(capture.weights, metadata, out_dir/f'row_{i}.png')
    finally:
        capture.close()


if __name__ == '__main__':
    main()
