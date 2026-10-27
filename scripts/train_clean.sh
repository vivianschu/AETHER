#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# CUDA headers are needed by PyTorch's native Triton kernels on this node.
export CPATH="/usr/local/cuda-12.8/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
AETHER_PYTHON="${AETHER_PYTHON:-.pixi/envs/default/bin/python}"
AETHER_NPROC="${AETHER_NPROC:-4}"
if (( 64 % AETHER_NPROC != 0 )); then
    echo 'AETHER_NPROC must divide 64 to preserve the total batch size.' >&2
    exit 1
fi
exec "$AETHER_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$AETHER_NPROC" \
    -m neoantigen.train.train --config "$1" --out_dir "$2" \
    --batch_size "$((64 / AETHER_NPROC))" --num_workers 2 --eval_every_steps 0
