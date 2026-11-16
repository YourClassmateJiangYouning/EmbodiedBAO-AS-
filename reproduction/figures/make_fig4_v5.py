"""Figure 4 (v5): four archetypal episodes in one 3-D view, wall plane and floor projections."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import make_v4 as v4
for f in ["Regular", "Bold", "Italic", "BoldItalic"]:
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/LiberationSerif-{f}.ttf")
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7})
NAVY, TEAL, SAND, APRI, CORAL, PLUM, INK = v4.NAVY, v4.TEAL, v4.SAND, v4.APRI, v4.CORAL, v4.PLUM, v4.INK
MAIN = "../fig/main/"

def trace_color_ramp(base, n):
    cm = LinearSegmentedColormap.from_list("r", ["#FFFFFF", base])
    return cm(np.linspace(0.45, 1.0, n))

sel = v4.pick()
style = {"early pass": (TEAL, "early pass"), "reactive failure": (CORAL, "reactive failure"),
         "insufficient rotation": (PLUM, "under-rotation"), "wide-gap pass": (NAVY, "wide-gap pass")}
fig = plt.figure(figsize=(3.5, 3.0))
ax = fig.add_subplot(111, projection="3d")
# wall plane at x = 8 m (forward position), spanning the step and rotation ranges
gx, gz = np.meshgrid(np.linspace(0, 32, 2), np.linspace(0, 100, 2))
ax.plot_surface(gx, np.full_like(gx, 8.0), gz, color=SAND, alpha=0.16, linewidth=0, shade=False)
ax.plot([0, 32, 32, 0, 0], [8, 8, 8, 8, 8], [0, 0, 100, 100, 0], color=SAND, lw=0.6, alpha=0.8)
legend = []
for name, row in sel.items():
    col, label = style[name]
    ts, xs, rs, turns = v4.episode_trace(row)
    # floor shadow
    ax.plot(ts, xs, np.zeros_like(rs), color="#9A9A9A", lw=0.7, alpha=0.6, zorder=1)
    pts = np.array([ts, xs, rs]).T
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    ramp = trace_color_ramp(col, len(segs))
    ax.add_collection3d(Line3DCollection(segs, colors=ramp, linewidths=2.1, zorder=3))
    if turns:
        ax.scatter(ts[turns], xs[turns], rs[turns], s=9, color=SAND, edgecolor=INK, lw=0.3, depthshade=False, zorder=4)
    ax.scatter([ts[-1]], [xs[-1]], [rs[-1]], s=22, color=col, edgecolor="white", lw=0.6, depthshade=False, zorder=5)
    legend.append(Line2D([], [], color=col, lw=2.2, label=f"{label} (A/S {row.A:.1f})"))
ax.set_xlim(0, 32); ax.set_ylim(0, 9); ax.set_zlim(0, 100)
ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([0, 4, 8]); ax.set_zticks([0, 45, 90])
ax.tick_params(labelsize=5.2, pad=-2)
ax.set_xlabel("step", fontsize=6, labelpad=-7); ax.set_ylabel("forward position (m)", fontsize=6, labelpad=-7)
ax.set_zlabel("body rotation (°)", fontsize=6, labelpad=-7)
ax.view_init(elev=24, azim=-62)
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_facecolor((0.985, 0.98, 0.97, 1)); axis.pane.set_edgecolor("#E3DED3")
ax.text(1.0, 8.0, 5, "wall at 8 m", fontsize=5.6, color="#8A6D1F", ha="left")
ax.legend(handles=legend, frameon=False, loc="upper left", bbox_to_anchor=(-0.08, 1.02), fontsize=5.6, handlelength=1.6)
fig.savefig(MAIN + "fancy_traj.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print("fig4 v5 written")
