"""Figures for the PEPMHC class-I scores over the sample-specific immunopeptidome.

Reads workflow/results/*.parquet (score_peptides.py) and validation.csv
(validate_model.py); writes 600 dpi PNGs to workflow/figures/ in the style of
pmlb-neoantigens/workflow/sample_specific_search/261020_poster.ipynb.

Binder calls use the %rank against a per-allele background of random natural
peptides, not the raw score: the checkpoint's raw output is saturated above 0.5
for essentially every peptide (see fig7/fig8), so its own decision boundary
carries no information.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_style import (POSTER_RC, SOURCE_COLORS, LOCUS_COLORS, MUTED,
                           BAR_COLOR, LIGHT_BAR_COLOR, bare_x, box_strip,
                           despine, save)
from background import RANK_BINDER, RANK_STRONG

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGDIR = HERE / "figures"
SOURCES = ["FragPipe", "Casanovo"]


def fig1_rank_distribution(df):
    with mpl.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        bins = np.linspace(0, 100, 101)
        for source in SOURCES:
            sub = df[df.source == source]
            if sub.empty:
                continue
            ax.hist(sub.best_rank, bins=bins, density=True, histtype="step",
                    linewidth=2.6, color=SOURCE_COLORS[source],
                    label=f"{source} (n={len(sub):,})")
        ax.axvline(RANK_BINDER, color=MUTED, linestyle="--", linewidth=1.6)
        ax.text(RANK_BINDER + 1.5, ax.get_ylim()[1] * 0.95,
                f"binder, %rank<={RANK_BINDER:g}", color=MUTED, fontsize=14, va="top")
        ax.set_xlabel("%rank vs random peptides (best allele)")
        ax.set_ylabel("Density")
        ax.set_title("Where observed peptides fall\nagainst a random background")
        ax.legend(frameon=False, loc="upper right", fontsize=14)
        despine(ax)
        save(fig, FIGDIR, "fig1_rank_distribution")


def fig2_binder_rate(df):
    rate = (df.groupby(["source", "alias"]).binder.mean().mul(100).reset_index())
    with mpl.rc_context(POSTER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": 0.32})
        for ax, source in zip(axes, SOURCES):
            vals = rate[rate.source == source].binder
            if vals.empty:
                ax.axis("off"); continue
            box_strip(ax, vals, SOURCE_COLORS[source], "Predicted binders (%)",
                      f"{source}\n({len(vals)} samples)")
        fig.suptitle(f"Peptides called binders (%rank<={RANK_BINDER:g}) per sample",
                     x=0.125, ha="left", fontsize=20)
        save(fig, FIGDIR, "fig2_binder_rate_per_sample")


def fig3_binders_per_sample(df):
    counts = df[df.binder].groupby(["source", "alias"]).size().rename("n").reset_index()
    with mpl.rc_context(POSTER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": 0.32})
        for ax, source in zip(axes, SOURCES):
            vals = counts[counts.source == source].n
            if vals.empty:
                ax.axis("off"); continue
            box_strip(ax, vals, SOURCE_COLORS[source],
                      "Predicted binders per sample", f"{source}\n({len(vals)} samples)")
        save(fig, FIGDIR, "fig3_binders_per_sample")


def fig4_rank_by_length(df):
    with mpl.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(7.5, 5.6))
        lengths = sorted(df.peptide.str.len().unique())
        width = 0.38
        for i, source in enumerate(SOURCES):
            sub = df[df.source == source]
            if sub.empty:
                continue
            L = sub.peptide.str.len()
            vals = [100 * sub.binder[L == k].mean() if (L == k).any() else 0
                    for k in lengths]
            ax.bar(np.arange(len(lengths)) + (i - 0.5) * width, vals, width,
                   color=SOURCE_COLORS[source], label=source)
        ax.set_xticks(np.arange(len(lengths)))
        ax.set_xticklabels(lengths)
        ax.set_xlabel("Peptide length (aa)")
        ax.set_ylabel("Called binders (%)")
        ax.set_title("Binder calls by peptide length")
        ax.legend(frameon=False, loc="upper right", fontsize=14)
        despine(ax)
        save(fig, FIGDIR, "fig4_binder_rate_by_length")


def fig5_allele_usage(df):
    n = df[df.binder].best_allele.value_counts().head(20)
    if n.empty:
        return
    with mpl.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(n) + 1.8))
        colors = [LOCUS_COLORS.get(a.split("-")[1][0], BAR_COLOR) for a in n.index]
        y = np.arange(len(n))
        ax.barh(y, n.values, color=colors, height=0.7)
        for i, v in enumerate(n.values):
            ax.text(v * 1.02, i, f"{v:,}", va="center", fontweight="bold", fontsize=14)
        ax.set_yticks(y); ax.set_yticklabels(n.index); ax.invert_yaxis()
        ax.set_xlim(0, n.max() * 1.2)
        ax.set_title("Best-ranking allele for called binders")
        ax.legend(handles=[Patch(color=c, label=f"HLA-{k}")
                           for k, c in LOCUS_COLORS.items()],
                  frameon=False, loc="lower right", fontsize=14)
        bare_x(ax)
        save(fig, FIGDIR, "fig5_allele_usage")


def fig6_raw_score_by_allele(pairs):
    """The raw score is almost entirely an allele offset -- this is why the
    figures above rank within an allele instead of thresholding the score."""
    g = pairs.groupby("allele").score.agg(["mean", "std", "size"])
    g = g[g["size"] >= 500].sort_values("mean", ascending=False).head(24)
    if g.empty:
        return
    with mpl.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(g) + 2.0))
        colors = [LOCUS_COLORS.get(a.split("-")[1][0], BAR_COLOR) for a in g.index]
        y = np.arange(len(g))
        ax.barh(y, g["mean"], xerr=g["std"], color=colors, height=0.7,
                error_kw={"ecolor": "#374151", "elinewidth": 1.4})
        ax.set_yticks(y); ax.set_yticklabels(g.index); ax.invert_yaxis()
        lo = max(0.0, g["mean"].min() - 4 * g["std"].max())
        ax.set_xlim(lo, min(1.0, g["mean"].max() + 4 * g["std"].max()))
        ax.set_xlabel("Raw P(bind), mean +/- sd")
        ax.set_title("Raw score is an allele offset\n(note the x-axis range)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        save(fig, FIGDIR, "fig6_raw_score_by_allele")


def fig7_validation(val):
    from sklearn.metrics import roc_curve, roc_auc_score
    with mpl.rc_context(POSTER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                                 gridspec_kw={"wspace": 0.3})
        ax = axes[0]
        fpr, tpr, _ = roc_curve(val.bind, val.score)
        auc = roc_auc_score(val.bind, val.score)
        ax.plot(fpr, tpr, color="#203A59", linewidth=3, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.6,
                label="chance")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"Held-out ROC\n(classI/test.csv, n={len(val):,})")
        ax.legend(frameon=False, loc="lower right", fontsize=14)
        despine(ax)

        # zoomed to the data: the model's own 0.5 boundary is far off-scale to
        # the left, which is the point -- every peptide is called a binder
        ax = axes[1]
        lo, hi = val.score.min(), val.score.max()
        bins = np.linspace(lo, hi, 60)
        for lab, color, name in ((0, "#d9d9d9", "non-binder"),
                                 (1, "#203A59", "binder")):
            ax.hist(val.score[val.bind == lab], bins=bins, color=color,
                    alpha=0.85, label=f"true {name}")
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.3f}"))
        ax.annotate("model's own 0.5 cutoff\nis off-scale to the left:\nevery peptide scores above it",
                    xy=(0.02, 0.97), xycoords="axes fraction", color="#e15759",
                    fontsize=14, va="top")
        ax.set_xlabel("Raw P(bind)"); ax.set_ylabel("Peptides")
        ax.set_title(f"Entire score range spans {hi - lo:.4f}")
        ax.legend(frameon=False, loc="center left", fontsize=14)
        despine(ax)
        save(fig, FIGDIR, "fig7_validation")


def fig8_diagnostic(grid):
    """How far the raw score travels when one input varies and the other is held
    fixed, over a peptide x allele grid. Median of the per-group range, so the
    two bars are directly comparable."""
    by_allele = grid.groupby("allele").score.agg(lambda s: s.max() - s.min()).median()
    by_peptide = grid.groupby("peptide").score.agg(lambda s: s.max() - s.min()).median()
    n_pep = grid.peptide.nunique()
    n_all = grid.allele.nunique()
    with mpl.rc_context(POSTER_RC):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        labels = [f"Peptide varies\n(allele fixed, {n_pep} peptides)",
                  f"Allele varies\n(peptide fixed, {n_all} alleles)"]
        vals = [by_allele, by_peptide]
        ax.bar(labels, vals, color=["#4e79a7", "#e15759"], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:.4f}", ha="center",
                    fontweight="bold")
        ax.set_ylabel("Median score range")
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_title(f"What moves the score\nthe allele moves it {by_peptide/by_allele:.0f}x more "
                     f"than the peptide")
        despine(ax)
        save(fig, FIGDIR, "fig8_what_moves_the_score")


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    made = []
    sp = RESULTS / "peptide_scores.parquet"
    if sp.exists():
        df = pd.read_parquet(sp).dropna(subset=["best_rank"])
        pairs = pd.read_parquet(RESULTS / "peptide_allele_scores.parquet").dropna(subset=["score"])
        print(f"{len(df):,} scored sample x peptide rows; {len(pairs):,} pair rows")
        fig1_rank_distribution(df); fig2_binder_rate(df); fig3_binders_per_sample(df)
        fig4_rank_by_length(df); fig5_allele_usage(df); fig6_raw_score_by_allele(pairs)
    else:
        print("no peptide_scores.parquet yet; skipping cohort figures")

    vp = RESULTS / "validation.csv"
    if vp.exists():
        fig7_validation(pd.read_csv(vp))
    dp = RESULTS / "diagnostic_grid.csv"
    if dp.exists():
        fig8_diagnostic(pd.read_csv(dp))


if __name__ == "__main__":
    main()
