# src/neoantigen/train/plotting.py
"""
Deprecated. All plotting now lives in the notebooks under ``figures/``:

    figures/01_dataset_overview.ipynb    dataset description
    figures/02_training_progress.ipynb   training curves and model results

``train.py`` writes the artifacts those notebooks read (``metrics.jsonl``,
``test_metrics.json``, ``test_predictions.csv``) and draws nothing itself, so
figures can be regenerated and restyled without re-running training.
"""

_MESSAGE = (
    "neoantigen.train.plotting has been removed. Plot training runs with the "
    "notebooks in figures/ (02_training_progress.ipynb), which read "
    "<out_dir>/metrics.jsonl and <out_dir>/test_predictions.csv."
)


def plot_test_results(*args, **kwargs):
    raise NotImplementedError(_MESSAGE)
