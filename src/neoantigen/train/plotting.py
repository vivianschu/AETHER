# src/neoantigen/train/plotting.py
import os

import numpy as np


def plot_test_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
    threshold: float = 0.5,
) -> None:
    """
    Generate and save test-set diagnostic plots.

    Outputs (saved under ``out_dir/plots/``):
      scatter.png           – predicted probability vs. true binding score
      score_distribution.png – predicted score density split by binder/non-binder
      roc_curve.png          – ROC curve (requires sklearn)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping test plots.", flush=True)
        return

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    y_bin = (y_true >= threshold).astype(bool)

    # ── 1. Scatter: predicted probability vs. true score ─────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.25, s=6, c="steelblue", rasterized=True)
    ax.axhline(threshold, color="crimson", linestyle="--", lw=1,
               label=f"threshold = {threshold}")
    ax.axvline(threshold, color="darkorange", linestyle="--", lw=1)
    ax.set_xlabel("True binding score")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Test set: predicted vs. true")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "scatter.png"), dpi=150)
    plt.close(fig)

    # ── 2. Score distribution by class ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 51)
    if y_bin.sum() > 0:
        ax.hist(y_pred[y_bin], bins=bins, alpha=0.65, label="Binder",
                color="coral", density=True)
    if (~y_bin).sum() > 0:
        ax.hist(y_pred[~y_bin], bins=bins, alpha=0.65, label="Non-binder",
                color="steelblue", density=True)
    ax.axvline(threshold, color="black", linestyle="--", lw=1,
               label=f"threshold = {threshold}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title("Predicted score distribution by class")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "score_distribution.png"), dpi=150)
    plt.close(fig)

    # ── 3. ROC curve ─────────────────────────────────────────────────────────
    if len(np.unique(y_bin)) == 2:
        try:
            from sklearn.metrics import roc_curve, auc as sk_auc
            fpr, tpr, _ = roc_curve(y_bin, y_pred)
            roc_auc = sk_auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(fpr, tpr, lw=2, color="steelblue",
                    label=f"AUC = {roc_auc:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.12, color="steelblue")
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve (test set)")
            ax.legend(fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=150)
            plt.close(fig)
        except ImportError:
            pass  # sklearn not available

    # ── 4. Calibration / residual scatter ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, alpha=0.2, s=6, c="steelblue", rasterized=True)
    ax.axhline(0, color="crimson", linestyle="--", lw=1)
    ax.set_xlabel("True binding score")
    ax.set_ylabel("Residual (pred − true)")
    ax.set_title("Residual plot (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "residuals.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {plots_dir}/", flush=True)
