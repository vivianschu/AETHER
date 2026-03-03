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
│       ├── models/      # Model code (ESM encoder, transformer head)
│       └── train/       # Training scripts (train.py)
├── requirements.txt
```

Main scripts
------------
- Training: `src/neoantigen/train/train.py` (see `src/neoantigen/train/README.md` for example command)
- Inference: `src/neoantigen/infer/predict.py` (see `src/neoantigen/infer/README.md`)
- Model: `src/neoantigen/models/esm.py` (ESM encoder), `src/neoantigen/models/transformer.py` (NeoantigenModel)

Data folders
------------
- `data/CEDAR/` — curated peptide–HLA binding and T cell datasets (see `data/CEDAR/README.md`)
- `data/HLA/` — HLA allele protein sequences from IPD-IMGT/HLA (see `data/HLA/README.md`)
- `data/model/` — processed splits (e.g., `mhc_set.csv`, `mhc_tcell_set.csv`, and train/valid/test splits)

Quick start
-----------
1. Create a Python environment (conda or venv).
2. Install dependencies for the workflow you plan to run:
	- For training: `pip install -r requirements.txt`
3. Run the appropriate `train_*.sh` script for experiment (see `src/neoantigen/train/README.md` for example command).

Usage
-----
		```bash
		python src/neoantigen/train/train.py \
			--split_dir data/model/splits/mhc-80-10-10 \
			--out_dir outputs/mhc_binding_esm_shared \
			--freeze_backbone --batch_size 8 --epochs 10 --lr 2e-4 --amp
		```
- Predict binding on a test set:
		```bash
			--ckpt outputs/mhc_binding_esm_shared/best.pt \
			--input_csv data/model/splits/mhc-80-10-10/test.csv \
			--output_csv outputs/mhc_binding_esm_shared/test_preds.csv \
			--amp
		```

Developer notes
---------------
- The repository was standardized to use the model name **AETHER** across docs and scripts.
