"""Why a partly rotated torso needs MORE room than an unrotated one.

The body is a rectangle: shoulders w = 0.570 m across, thickness d = 0.220 m deep.  The
channel constrains the lateral projection, so a torso yawed by theta needs

    needed(theta) = w*|cos theta| + d*|sin theta|

The cosine term falls with theta while the sine term rises from zero, so the projection
peaks where -w*sin + d*cos = 0, i.e. tan(theta) = d/w = 0.386, theta = 21.1 degrees.
Past the crossing at 42.2 degrees the rotation finally pays off, and at 90 degrees the
body presents its thickness.  In between, rotating makes the body worse than not
rotating -- which is what the Level 9/10 episodes show: no pass at A/S 1.00 ever had a
passage yaw between 12 and 42 degrees.

Run from the repository root:  python lab_logs/make_geometry_figure.py
"""

import math
import os
import sys

sys.path.insert(0, ".")
import memory_metrics as mm  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

TEXT_WIDTH_CM = 12.2
FONT_PT = 8
RASTER_DPI = 600
FONT_SANS = ["Arial", "Helvetica", "DejaVu Sans"]
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
OUTDIR = "lab_logs/figures"
CM = 1.0 / 2.54
SHOULDER = mm.SHOULDER_WIDTH_M
THICKNESS = mm.TORSO_THICKNESS_M


def needed(theta_deg):
    return mm.needed_width(theta_deg)


def main() -> int:
    peak_deg = math.degrees(math.atan2(THICKNESS, SHOULDER))
    peak = needed(peak_deg)
    # Analytic, not scanned: with tan(theta*) = d/w, needed(2*theta*) works out to
    # exactly w, so the rotation only starts to pay off at twice the optimal angle.
    crossing = 2.0 * peak_deg

    print("shoulder %.3f m x thickness %.3f m" % (SHOULDER, THICKNESS))
    for theta in (0, 10, 15, 21.1, 30, 42.2, 45, 60, 75, 90):
        print("  theta %5.1f deg -> needed %.4f m   (%s than frontal)" % (
            theta, needed(theta),
            "wider" if needed(theta) > SHOULDER + 1e-9 else
            "equal" if abs(needed(theta) - SHOULDER) < 1e-9 else "narrower"))
    print("peak at %.2f deg -> %.4f m" % (peak_deg, peak))
    print("rotating is worse than not rotating for 0 < yaw < %.1f deg" % crossing)

    figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_CM * CM, 3.0),
                                gridspec_kw={"width_ratios": [1.25, 1.0]})
    left, right = axes

    angles = [t / 10.0 for t in range(0, 901)]
    widths = [needed(a) for a in angles]
    left.plot(angles, widths, "-", color=PALETTE[0])
    left.axhline(SHOULDER, color="0.35", lw=0.6, ls="--")
    left.axvspan(0, crossing, color=PALETTE[1], alpha=0.10)
    left.axvline(peak_deg, color=PALETTE[1], lw=0.7, ls=":")
    left.plot([0], [SHOULDER], "o", color=PALETTE[0], markersize=3.5)
    left.plot([peak_deg], [peak], "o", color=PALETTE[1], markersize=3.5)
    left.plot([90], [THICKNESS], "o", color=PALETTE[2], markersize=3.5)
    left.annotate("frontal %.3f m" % SHOULDER, (0.5, SHOULDER + 0.006), fontsize=FONT_PT - 2)
    left.annotate("worst %.3f m\nat %.1f deg" % (peak, peak_deg),
                  (peak_deg + 2, peak + 0.004), fontsize=FONT_PT - 2)
    left.annotate("thickness %.3f m" % THICKNESS, (62, THICKNESS + 0.012), fontsize=FONT_PT - 2)
    left.annotate("rotating here needs\nMORE room than not rotating",
                  (24, 0.50), fontsize=FONT_PT - 2, color=PALETTE[1])
    left.set_xlabel("torso yaw (deg)")
    left.set_ylabel("lateral projection needed (m)")
    left.set_xlim(0, 90)
    left.set_ylim(0.19, 0.645)
    left.set_title("partly turning needs more room")

    # Top-down view: the wall is a line with a gap, the body approaches from the left,
    # and what decides it is the body's extent across the walking direction.
    slots = [(0.570, "A/S 1.00\n0.570 m"), (0.513, "A/S 0.90\n0.513 m")]
    yaws = (0, 21.1, 90)
    for column, theta in enumerate(yaws):
        right.text(column * 1.55 + 0.70, 2.36, "%.0f deg" % theta, ha="center",
                   fontsize=FONT_PT - 1)
    for row, (slot, label) in enumerate(slots):
        centre_y = row * 1.30
        for column, theta in enumerate(yaws):
            centre_x = column * 1.55 + 0.70
            wall_x = centre_x + 0.40
            right.plot([wall_x, wall_x], [centre_y + slot / 2, centre_y + 0.60],
                       color="0.30", lw=1.6, solid_capstyle="butt")
            right.plot([wall_x, wall_x], [centre_y - 0.60, centre_y - slot / 2],
                       color="0.30", lw=1.6, solid_capstyle="butt")
            right.plot([centre_x - 0.50, wall_x], [centre_y, centre_y],
                       color="0.75", lw=0.4, ls=":", zorder=0)
            body = Rectangle((-SHOULDER / 2, -THICKNESS / 2), SHOULDER, THICKNESS,
                             facecolor=PALETTE[0], edgecolor="black", lw=0.4, alpha=0.9)
            body.set_transform(matplotlib.transforms.Affine2D().rotate_deg(theta)
                               .translate(centre_x, centre_y) + right.transData)
            right.add_patch(body)
            fits = needed(theta) <= slot + 1e-9
            right.text(centre_x, centre_y + 0.40,
                       "%.3f  %s" % (needed(theta), "fits" if fits else "blocked"),
                       ha="center", fontsize=FONT_PT - 4,
                       color="black" if fits else PALETTE[1])
        right.text(-0.16, centre_y, label, ha="center", va="center", fontsize=FONT_PT - 2)
    right.set_xlim(-0.85, 3.95)
    right.set_ylim(-0.75, 2.62)
    right.set_aspect("equal")
    right.axis("off")
    right.set_title("the same body, three yaws")

    os.makedirs(OUTDIR, exist_ok=True)
    for fmt in ("pdf", "png"):
        figure.savefig(os.path.join(OUTDIR, "fig5_rotation_geometry.%s" % fmt),
                       format=fmt, dpi=RASTER_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print("wrote %s/fig5_rotation_geometry.{pdf,png}" % OUTDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
