# AETHER
**A**ffinity **E**valuation for **T**-cell **H**LA **E**pitope Recognition.

Project Structure
-----------------
```
AETHER/
├── data/
│   ├── CEDAR/           # Cancer Epitope Database and Analysis Resource (curated CSVs, preprocess notebooks)
│   ├── HLA/             # HLA allele protein FASTA and annotation
│   └── model/           # Processed datasets, splits, and export scripts
├── src/
│   └── neoantigen/
│       ├── infer/       # Inference scripts (predict.py)
│       ├── models/      # Model code (ESM3 backbone, cross-attention head)
│       └── train/       # Training scripts (train.py)
├── requirements.txt
```

Main scripts
------------
- Training: `src/neoantigen/train/train.py` (see `src/neoantigen/train/README.md` for example command)
- Inference: `src/neoantigen/infer/predict.py` (see `src/neoantigen/infer/README.md`)
- Model: `src/neoantigen/models/esm3.py` (ESM3 backbone + attention pooling), `src/neoantigen/models/transformer.py` (NeoantigenModel with cross-attention)

Data folders
------------
- `data/CEDAR/` — curated peptide–HLA binding and T cell datasets (see `data/CEDAR/README.md`)
- `data/HLA/` — HLA allele protein sequences from IPD-IMGT/HLA (see `data/HLA/README.md`)
- `data/model/` — processed splits (e.g., `mhc_set.csv`, `mhc_tcell_set.csv`, and train/valid/test splits)

Quick start
-----------
1. Create a Python environment (conda or venv).
	```bash
	module load python/3.12 cuda/12.2
	module load arrow/21.0.0
	module load slurm
	python -m venv --system-site-packages ~/venv/aether_sys
	source ~/venv/aether_sys/bin/activate
	python -c "import pyarrow as pa; print(pa.__version__)"
	pip install datasets

	# To update packages from requirements
	pip install -r requirements.txt --upgrade
	```
2. Install dependencies for the workflow you plan to run:
	- For training: `pip install -r requirements.txt`
3. Run the appropriate `train_*.sh` script for experiment (see `src/neoantigen/train/README.md` for example command).

Usage
-----
- Train on MHC binding data:
	```bash
	python -m neoantigen.train.train \
		--csv_url data/model/mhc_set.csv \
		--out_dir outputs/mhc_binding \
		--freeze_backbone --batch_size 8 --epochs 10 --lr 2e-4 --amp
	```
- Predict binding on a test set:
	```bash
	python -m neoantigen.infer.predict \
		--ckpt outputs/mhc_binding/best.pt \
		--input_csv data/model/mhc_set.csv \
		--output_csv outputs/mhc_binding/preds.csv \
		--amp
	```

Developer notes
---------------
- The repository was standardized to use the model name **AETHER** across docs and scripts.
- Backbone: ESM3 (`esm3_sm_open_v1`, 1536-dim hidden states) via EvolutionaryScale's `esm` package. Replaces the previous ESM2 encoder.
- Architecture: bidirectional cross-attention between peptide and MHC token sequences, followed by learned attention pooling and an MLP head.