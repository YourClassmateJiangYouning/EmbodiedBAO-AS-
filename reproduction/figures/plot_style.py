"""House plotting style, learned from `D:\\数据\\画图教程\\figure (1).ipynb`.

The tutorial (a LAM image-compression benchmark notebook) uses one consistent
recipe across its 21 figures:

* **Serif text everywhere** — Times New Roman registered through
  ``font_manager.addfont`` and selected with ``plt.rc('font', family=...,
  size=...)``; math text renders with the matching STIX set.
* **One fixed six-colour palette** (green / blue / orange / yellow / purple /
  teal) reused for every categorical axis.
* **Dashed lines with distinct markers** (``o``, ``+``, ``*``, ``x``, ``D``),
  a light dashed grid, explicit ``xlim`` / ``ylim`` and ``xticks`` / ``yticks``.
* **Heatmaps** with a single-hue truncated colormap, ``interpolation='nearest'``,
  ``aspect='auto'``, inverted y axis, and a printed value in *every* cell.
* **Bars** with black edges, ``alpha=0.7``, error bars (``capsize=5``), a
  y-only dashed grid, rotated category labels (``rotation≈35``, ``ha='right'``).
* **Violins / radar** for distribution and multi-metric model comparisons.
* **No figure titles** — the caption carries the message; the axes labels
  themselves often identify the sub-panel.
* **Export**: ``dpi=500``, ``bbox_inches='tight'``, ``transparent=True`` PDF,
  large explicit ``figsize`` chosen per layout; highlighted panels get a pale
  blue background.

This module packages that recipe so every figure in this project is drawn the
same way.  Import it, call :func:`setup`, then use :func:`save`.
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# The tutorial's six-colour categorical palette, used in this order everywhere.
PALETTE = ["#3c7daa", "#e64f04", "#f2c400", "#a757a7", "#56B1AD", "#92ca2c"]
PALETTE_EXTRA = ["#e64f04", "#56B1AD", "#a757a7", "#f2c400", "#92ca2c", "#3c7daa"]
GREY = "#7f7f7f"
LIGHT_GREY = "#bdbdbd"
INK = "#1a1a1a"
HUMAN_BAND = "#f2c400"
HIGHLIGHT_BG = (0.898, 0.937, 0.965)   # pale blue panel background from the tutorial
GRID = {"color": "#d9d9d9", "linestyle": "--", "linewidth": 0.6, "alpha": 0.8}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(ROOT, "analysis", "figures_paper")

_TIMES_CANDIDATES = [
    r"C:\Windows\Fonts\times.ttf",
    "/usr/share/fonts/truetype/TIMES.TTF",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
]


def setup(base_size: float = 9.0) -> None:
    """Apply the tutorial's rcParams (Times New Roman, serif, dashed grid)."""
    for path in _TIMES_CANDIDATES:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass
            break
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_size,
        "axes.labelsize": base_size + 0.5,
        "axes.titlesize": base_size + 0.5,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "legend.fontsize": base_size - 1.0,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "grid.color": GRID["color"],
        "grid.linestyle": GRID["linestyle"],
        "grid.linewidth": GRID["linewidth"],
        "grid.alpha": GRID["alpha"],
        "savefig.dpi": 500,
        "savefig.bbox": "tight",
        "savefig.transparent": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, name: str, outdir: str | None = None, transparent: bool = True) -> None:
    """Save one figure as PDF (vector) and 500-dpi PNG, tutorial-style."""
    outdir = outdir or FIG_DIR
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", transparent=transparent,
                    dpi=500 if ext == "png" else None)
    plt.close(fig)
    print(f"[figure] {os.path.join(outdir, name)}.pdf/.png")


def truncated_cmap(name: str = "Oranges", lo: float = 0.0, hi: float = 0.85) -> ListedColormap:
    """The tutorial's ``newcolors = cmap(np.linspace(0,1,256))[...]`` idiom."""
    import numpy as np
    cmap = plt.get_cmap(name)
    colors = cmap(np.linspace(0, 1, 256))
    return ListedColormap(colors[int(lo * 256):int(hi * 256)])


def style_axes(ax, grid_axis: str = "both", background: bool = False) -> None:
    ax.grid(True, axis=grid_axis, **GRID)
    ax.set_axisbelow(True)
    if background:
        ax.set_facecolor(HIGHLIGHT_BG)


def panel_label(ax, text: str, x: float = 0.0, y: float = 1.035, size: float | None = None) -> None:
    """The (a)/(b) tag, placed just above the axes so it never covers data."""
    if not text:
        return
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=size if size is not None else plt.rcParams["font.size"] + 1.0,
            fontweight="bold", color=INK)


def heatmap(ax, matrix, row_labels, col_labels, cmap="Oranges", vmin=0.0, vmax=1.0,
            fmt: str = "{:.2f}", annot_size: float = 5.6, rotate: float = 32):
    """Tutorial heatmap: imshow + annotated cells + inverted y + truncated cmap."""
    import numpy as np
    im = ax.imshow(np.asarray(matrix, dtype=float), cmap=truncated_cmap(cmap),
                   interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=rotate, ha="right")
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        color="black", fontsize=annot_size)
    ax.set_xticks([x - 0.5 for x in range(1, len(col_labels))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(row_labels))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.invert_yaxis()
    return im
