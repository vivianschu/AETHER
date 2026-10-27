"""Plot style shared by the PEPMHC figures.

Matches pmlb-neoantigens/workflow/sample_specific_search/261020_poster.ipynb:
Arial, the same categorical hexes, 600 dpi PNGs cropped tight, spines hidden by
hand rather than via seaborn, titles left-aligned.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 10

DPI = 600
BAR_COLOR = "#4b5563"
LIGHT_BAR_COLOR = "#d1d5db"
MUTED = "#6b7280"
NOT_RECORDED = "#D9D9D9"

SOURCE_COLORS = {"FragPipe": "#4e79a7", "Casanovo": "#f28e2b"}
CLASS_COLORS = {"Canonical": "#4e79a7", "Mutated": "#e15759", "Novel ORF": "#f28e2b"}
LOCUS_COLORS = {"A": "#4e79a7", "B": "#e15759", "C": "#59a14f"}
BIND_COLORS = {"Predicted binder": "#203A59", "Non-binder": "#d9d9d9"}

POSTER_RC = {
    "font.size": 18, "axes.labelsize": 18, "axes.titlesize": 20,
    "xtick.labelsize": 17, "ytick.labelsize": 17,
    "axes.linewidth": 1.4, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "axes.titlelocation": "left", "axes.titlepad": 14,
}


def save_fig(fig, path):
    """600 dpi png cropped to everything drawn, legends and outside labels included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")


def save(fig, outdir, name):
    save_fig(fig, outdir / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png")


def bare_x(ax):
    """direct-labelled horizontal bars: no x axis, no tick marks on the categories"""
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_xticks([])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.tick_params(axis="y", length=0)


def despine(ax, sides=("top", "right")):
    ax.spines[list(sides)].set_visible(False)


def pct_label(p):
    return f"{p:.0f}%" if p >= 10 else f"{p:.2g}%"


def box_strip(ax, values, color, label, title, zoom=False):
    """one box (median, IQR) with every sample as a point and the median beside it"""
    values = pd.Series(values).dropna()
    ax.boxplot(values, positions=[0], widths=0.6, showfliers=False,
               boxprops={"color": "#374151", "linewidth": 1.8},
               whiskerprops={"color": "#374151", "linewidth": 1.8},
               capprops={"color": "#374151", "linewidth": 1.8},
               medianprops={"color": "black", "linewidth": 3})
    jitter = np.random.default_rng(0).uniform(-0.22, 0.22, len(values))
    ax.scatter(jitter, values, s=40, color=color, alpha=0.7, linewidths=0, zorder=3)
    median = values.median()
    ax.text(0.4, median, f"median\n{median:,.0f}{'%' if '%' in label else ''}",
            va="center", fontweight="bold")
    ax.set_xticks([])
    ax.set_xlim(-0.5, 1.0)
    step = 5 if values.max() <= 100 else 10 ** np.floor(np.log10(values.max()))
    low = np.floor(values.min() / step) * step - step / 2 if zoom else 0
    ax.set_ylim(low, np.ceil(values.max() / step) * step + step / 2)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    if values.max() >= 1000:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
