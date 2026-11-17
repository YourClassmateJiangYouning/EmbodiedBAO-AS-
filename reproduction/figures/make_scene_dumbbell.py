"""Per-scene pass rates at wide and narrow openings (horizontal dumbbell), for Sec. 3 validity check.
Reads data/v8/S1_episodes.csv. Wilson 95% intervals."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ["Regular", "Bold", "Italic", "BoldItalic"]:
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/LiberationSerif-{f}.ttf")
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7.5,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})
csv, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv)
d["passed"] = d.passed.astype(str) == "True"
d["wide"] = d.a_s_ratio >= 1.0 - 1e-9
names = {"stage1.1": "Laboratory", "stage1.2": "Warehouse", "stage1.3": "Library",
         "stage1.4": "Park", "stage1.5": "Supermarket"}
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * p, 100 * (c - h), 100 * (c + h)
order = list(names)[::-1]
fig, ax = plt.subplots(figsize=(3.3, 1.7))
NAR, WID, LINE = "#C0504D", "#2F5597", "#BFBFBF"
for i, sc in enumerate(order):
    x = d[d.scene == sc]
    w = wilson(x[x.wide].passed.sum(), x.wide.sum()); n = wilson(x[~x.wide].passed.sum(), (~x.wide).sum())
    ax.plot([n[0], w[0]], [i, i], color=LINE, lw=2.2, zorder=1, solid_capstyle="round")
    for v, c in [(n, NAR), (w, WID)]:
        ax.plot([v[1], v[2]], [i, i], color=c, lw=0.9, zorder=2)
        ax.scatter([v[0]], [i], s=22, color=c, edgecolor="white", lw=0.5, zorder=3)
    ax.text(n[0], i + 0.2, f"{n[0]:.1f}", ha="center", va="bottom", fontsize=6.0, color=NAR)
    ax.text(w[0], i + 0.2, f"{w[0]:.1f}", ha="center", va="bottom", fontsize=6.0, color=WID)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names[s] for s in order])
ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_ylim(-0.5, len(order) - 0.15)
ax.set_xlabel("pass rate (%)")
ax.tick_params(length=2, width=0.6)
ax.scatter([], [], s=22, color=NAR, label=r"narrow ($A/S\leq0.9$)")
ax.scatter([], [], s=22, color=WID, label=r"wide ($A/S\geq1.0$)")
ax.legend(loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=2, frameon=False, fontsize=6.3, handletextpad=0.2, columnspacing=1.2)
fig.tight_layout(pad=0.2)
fig.savefig(out)
print("written", out)
