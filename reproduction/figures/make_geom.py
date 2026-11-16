"""Geometry schematic: torso footprint inside the aperture at four A/S settings."""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib import font_manager
for f in ["Regular", "Bold", "Italic", "BoldItalic"]:
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/liberation/LiberationSerif-{f}.ttf")
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
NAVY, RUST, GREY, PBLUE = "#2F5597", "#C55A11", "#BFBFBF", "#9DC3E6"
S, D = 0.570, 0.220
cases = [(2.0, 0), (1.0, 0), (0.9, 54.0), (0.6, 77.1)]
fig, axs = plt.subplots(1, 4, figsize=(6.9, 1.55), gridspec_kw={"wspace": 0.08})
for ax, (asr, th) in zip(axs, cases):
    A = asr * S
    half = 0.95
    # wall panels with the opening
    ax.add_patch(Rectangle((-half, 0.0), half - A / 2, 0.03, color="#7F7F7F", lw=0))
    ax.add_patch(Rectangle((A / 2, 0.0), half - A / 2, 0.03, color="#7F7F7F", lw=0))
    # body footprint: rectangle of depth D (along y) and width S rotated by th
    t = np.deg2rad(th)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    corners = np.array([[-S / 2, -D / 2], [S / 2, -D / 2], [S / 2, D / 2], [-S / 2, D / 2]])
    # body centre sits behind the wall at the opening
    c = corners @ R.T + np.array([0, -0.22])
    ax.add_patch(Polygon(c, closed=True, fc=PBLUE, ec=NAVY, lw=0.6, alpha=0.9))
    W = S * abs(np.cos(t)) + D * abs(np.sin(t))
    # dimension arrow across the opening (A) and the needed width (W)
    ax.annotate("", xy=(A / 2, 0.085), xytext=(-A / 2, 0.085),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=RUST))
    ax.text(0, 0.11, f"A = {A:.2f} m", ha="center", fontsize=6.4, color=RUST)
    ax.set_title(f"A/S {asr:.1f}, θ = {th:.0f}°", fontsize=7, pad=3)
    ax.text(0, -0.56, f"W = {W:.2f} m", ha="center", fontsize=6.4, color=NAVY)
    ax.set_xlim(-half, half); ax.set_ylim(-0.70, 0.20); ax.axis("off")
os.makedirs("../fig", exist_ok=True)
fig.savefig("../fig/fig_geometry.pdf")
print("saved")
