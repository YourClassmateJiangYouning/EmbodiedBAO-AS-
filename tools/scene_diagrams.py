"""Draw each scene's layout and what it looks like from the agent's start pose.

Two panels per scene:

  left   top view (x across, z up the page): the room, the obstacle wall and its opening, the
         marker on the far wall, and every dressing item as its footprint.  The shaded band is
         the rule that floor dressing stays off the centre line -- nothing may sit where the
         agent walks.
  right  the start-view frame, from scenes.project: the same items as the rectangles they occupy
         in the camera image, so it is visible at a glance whether anything covers the opening
         or the marker.  Solid means in front of the obstacle wall, dashed means behind it.

This is a design check, not a render: it uses only the numbers in scenes.py, needs no simulator
and no API key, and it is the picture to look at before spending GPU time on a scene.

    python tools/scene_diagrams.py                 # every scene
    python tools/scene_diagrams.py --scene stage1.3
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import environment as env  # noqa: E402
import scenes as sc  # noqa: E402

OUT_DIR = os.path.join("lab_logs", "figures", "scenes")
ROOM_HALF_Z = env.ROOM_WIDTH_Z / 2.0
WALL_X = env.WALL_X
FAR_X = env.ROOM_LENGTH_X
# The opening drawn on the layout: A/S 1.0, the ratio the literature calls the threshold.
DRAWN_RATIO = 1.0
DRAWN_WIDTH = DRAWN_RATIO * env.SHOULDER_WIDTH_M if hasattr(env, "SHOULDER_WIDTH_M") else 0.57


def footprint(item):
    x, _, z = (float(v) for v in item["at"])
    sx, _, sz = (float(v) for v in item["size"])
    return x, z, sx, sz


def draw_layout(ax, scene: str) -> None:
    spec = sc.SCENES[scene]

    # Room
    ax.add_patch(patches.Rectangle((0, -ROOM_HALF_Z), FAR_X, env.ROOM_WIDTH_Z,
                                   fill=False, edgecolor="0.35", linewidth=1.0))
    ax.axvline(FAR_X, color="0.35", linewidth=2.0)
    ax.text(FAR_X - 0.25, ROOM_HALF_Z - 0.25, "far wall", rotation=90,
            ha="right", va="top", fontsize=5, color="0.35")

    # Keep-out band: floor dressing may not sit inside it
    ax.add_patch(patches.Rectangle((0, -1.9), WALL_X, 3.8, facecolor="0.93",
                                   edgecolor="none", zorder=0))
    ax.text(0.15, -1.82, "keep-out band |z| < 1.9", fontsize=5, color="0.45", zorder=1)

    # Obstacle wall with the drawn opening.  Both panels are drawn from the edge of the opening
    # out to the room wall, computed per side: an earlier version used a signed height for the
    # lower panel, which made it extend upwards and cover the opening it was supposed to flank.
    for z_lo, z_hi in ((-ROOM_HALF_Z, -DRAWN_WIDTH / 2.0), (DRAWN_WIDTH / 2.0, ROOM_HALF_Z)):
        ax.add_patch(patches.Rectangle((WALL_X - 0.05, z_lo), 0.1, z_hi - z_lo,
                                       facecolor="#2b4bb5", edgecolor="none", zorder=2))
    ax.text(WALL_X, ROOM_HALF_Z - 0.15, f"opening A/S {DRAWN_RATIO:.1f}",
            ha="center", va="top", fontsize=5, color="#2b4bb5")

    # Marker
    ax.add_patch(patches.Rectangle((sc.MARKER_X_M - 0.1, -sc.MARKER_SIZE_M / 2.0),
                                   0.2, sc.MARKER_SIZE_M, facecolor="magenta",
                                   edgecolor="black", linewidth=0.3, zorder=4))
    ax.text(sc.MARKER_X_M - 0.2, sc.MARKER_SIZE_M / 2.0 + 0.06, "marker",
            ha="right", fontsize=5)

    # Dressing
    for item in spec["dressing"]:
        x, z, sx, sz = footprint(item)
        colour = tuple(float(v) for v in item.get("colour") or (0.5, 0.5, 0.5))
        ax.add_patch(patches.Rectangle((x - sx / 2.0, z - sz / 2.0), sx, sz,
                                       facecolor=colour, edgecolor="black",
                                       linewidth=0.4, alpha=0.9, zorder=3))
        ax.text(x - sx / 2.0, z + sz / 2.0 + 0.05, item["name"], fontsize=4.5, zorder=5)

    # Start pose and the lane the agent walks
    ax.plot([0.5], [0.0], marker="o", markersize=3, color="black", zorder=5)
    ax.text(0.6, -0.28, "start", fontsize=5)
    ax.set_xlim(-0.6, FAR_X + 0.6)
    ax.set_ylim(-ROOM_HALF_Z - 0.5, ROOM_HALF_Z + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=6)
    ax.set_ylabel("z (m)", fontsize=6)
    ax.set_title(f"{scene}  {spec['label']}  -  top view", fontsize=7)
    ax.tick_params(labelsize=5)


def draw_frame(ax, scene: str) -> None:
    spec = sc.SCENES[scene]

    # The frame itself
    ax.add_patch(patches.Rectangle((-1, -1), 2, 2, fill=False, edgecolor="0.35", linewidth=1.0))

    # Opening at three ratios, so the ladder's range is visible
    for ratio, style in ((2.0, ":"), (1.0, "-"), (0.4, "--")):
        width = ratio * 0.57
        bounds = sc.screen_bounds(*sc.opening_box(width))
        if not bounds:
            continue
        u0, v0, u1, v1 = bounds
        ax.add_patch(patches.Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False,
                                       edgecolor="#2b4bb5", linestyle=style, linewidth=0.9))
        ax.text(u1 + 0.02, v1, f"A/S {ratio:.1f}", fontsize=4.5, color="#2b4bb5")

    bounds = sc.screen_bounds(*sc.marker_box())
    if bounds:
        u0, v0, u1, v1 = bounds
        ax.add_patch(patches.Rectangle((u0, v0), max(u1 - u0, 0.01), max(v1 - v0, 0.01),
                                       facecolor="magenta", edgecolor="black", linewidth=0.3))
        ax.text(u0, v1 + 0.02, "marker", fontsize=5)

    for item in spec["dressing"]:
        bounds = sc.screen_bounds(item["at"], item["size"])
        if not bounds:
            continue
        u0, v0, u1, v1 = bounds
        colour = tuple(float(v) for v in item.get("colour") or (0.5, 0.5, 0.5))
        behind = float(item["at"][0]) > WALL_X
        ax.add_patch(patches.Rectangle((u0, v0), u1 - u0, v1 - v0,
                                       facecolor=colour, alpha=0.35 if behind else 0.8,
                                       edgecolor="black", linewidth=0.4,
                                       linestyle="--" if behind else "-"))
        ax.text(u0, v0 - 0.06, item["name"], fontsize=4.2)

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("u  (right of centre)", fontsize=6)
    ax.set_ylabel("v  (above centre)", fontsize=6)
    ax.set_title(f"{scene}  -  what the agent sees at the start (A/S 1.0 drawn solid)",
                 fontsize=7)
    ax.tick_params(labelsize=5)


def describe(scene: str) -> str:
    """A line of numbers for each scene: the constraints the design has to satisfy."""
    spec = sc.SCENES[scene]
    items = spec["dressing"]
    if not items:
        return "  no dressing, no materials: the untouched baseline"
    floor = [i for i in items if i["mount"] == "floor"]
    wall = [i for i in items if i["mount"] != "floor"]
    min_z = min(abs(float(i["at"][2])) for i in items)
    worst_height = max((float(i["size"][1]) for i in floor), default=0.0)
    worst_protrusion = max((float(i["size"][0]) / 2.0 for i in wall), default=0.0)
    overlaps = 0
    marker = sc.screen_bounds(*sc.marker_box())
    for item in items:
        bounds = sc.screen_bounds(item["at"], item["size"])
        if not bounds:
            continue
        if marker and sc.rectangles_overlap(bounds, marker):
            overlaps += 1
    return (f"  {len(items)} items ({len(floor)} on the floor, {len(wall)} on a wall); "
            f"closest to the centre line |z| = {min_z:.2f} m; tallest floor item "
            f"{worst_height:.2f} m; deepest wall item {worst_protrusion:.2f} m; "
            f"{overlaps} item(s) overlapping the marker in the start view")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw the scene layouts and start views.")
    parser.add_argument("--scene", default="")
    parser.add_argument("--outdir", default=OUT_DIR)
    args = parser.parse_args()

    scenes = [args.scene] if args.scene else list(sc.SCENE_ORDER)
    os.makedirs(args.outdir, exist_ok=True)
    for scene in scenes:
        if scene not in sc.SCENES:
            print(f"unknown scene {scene}")
            return 1
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=200)
        draw_layout(axes[0], scene)
        draw_frame(axes[1], scene)
        fig.tight_layout()
        path = os.path.join(args.outdir, f"{scene}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"{scene}: {path}")
        print(describe(scene))
    return 0


if __name__ == "__main__":
    sys.exit(main())
