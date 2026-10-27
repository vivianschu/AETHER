"""Immutable run identity and content hashes for publication provenance."""
import hashlib
import shutil
import json
import os
import platform
import importlib.metadata
import subprocess
from pathlib import Path
from datetime import datetime, timezone


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(out_dir, args, splits, audit):
    repo = Path(__file__).resolve().parents[2]
    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(repo), *cmd], text=True).strip()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        run_dir=str(Path(out_dir).resolve()), pid=os.getpid(), hostname=platform.node(),
        config=dict(vars(args)), split_audit=audit,
        software={p: importlib.metadata.version(p) for p in
                  ["torch", "transformers", "numpy", "pandas", "scikit-learn"]},
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        runtime_env={k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "CPATH", "HF_HUB_OFFLINE"]},
        split_sha256={k: hashlib.sha256(v.to_csv(index=False).encode()).hexdigest()
                      for k,v in splits.items()},
        source_sha256={str(p.relative_to(repo)): file_sha256(p)
                       for p in sorted((repo/"src/neoantigen").rglob("*.py"))},
        git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--short"))
    for rel in manifest["source_sha256"]:
        destination = Path(out_dir)/"source_snapshot"/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/rel, destination)
    with open(Path(out_dir)/"run_manifest.json", "x") as f:
        json.dump(manifest, f, indent=2)
    return file_sha256(Path(out_dir)/"run_manifest.json")
