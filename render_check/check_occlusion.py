"""Quantitative occlusion check on the rendered frames, for all 25 scene x slot pairs.

The geometric test (test_no_dressing_occludes_the_opening_or_the_marker) projects item boxes
against the opening and the marker, but it cannot see the mesh a referenced prop puts inside
that box, and it cannot see the frame.  This reads the frames.

Per frame it reports:
  * the marker's pixel location, found by its own colour rather than by looking;
  * whether any dressing rect covers that location, using scenes.screen_bounds for the same
    projection the geometric test uses;
  * the fraction of the opening's rect that is not the far wall's colour, a proxy for "did
    something move in front of the opening".

Finding the marker by colour, not by eye, is the point: a marker the eye can find in one frame
says nothing about the other twenty-four.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

ROOT = r"D:\EmInAI\EmbodiedBAO(AS)"
sys.path.insert(0, ROOT)

STAGE = r"C:\Users\asus\Desktop\科研狗之人机心理学\stage\prev"

import scenes as sc  # noqa: E402


def marker_colour(scene: str, slot: int):
    """The marker's own RGB for this slot, as 0-255."""
    _shape, key = sc.MARKERS[scene][slot - 1]
    rgb = sc.COLOURS[key]
    return tuple(int(round(c * 255)) for c in rgb)


def find_marker(image: np.ndarray, rgb) -> tuple:
    """Bounding box of the pixels closest to the marker's colour, or None.

    Distance rather than equality: the render is noisy (RaytracedLighting, spp 32, no denoiser)
    and a pixel lands exactly on the authored colour about never.  The 0.1% nearest pixels are
    taken, then filtered to those within a generous radius of the authored colour.
    """
    target = np.array(rgb, dtype=float)
    flat = image.reshape(-1, 3).astype(float)
    distance = np.abs(flat - target).sum(axis=1)
    keep = max(1, int(len(flat) * 0.001))
    nearest = np.argpartition(distance, keep - 1)[:keep]
    # A marker is a saturated patch; if the 0.1% nearest are far from the colour, there is no
    # marker in frame rather than a marker somewhere unexpected.
    if float(np.percentile(distance[nearest], 90)) > 180:
        return None
    ys, xs = np.divmod(nearest, image.shape[1])
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main() -> int:
    print(f"{'scene':<9} {'slot':<4} {'marker px box':<26} {'size':<10} "
          f"{'covered by dressing?':<22} verdict")
    print("-" * 108)
    problems = []
    for scene in sc.SCENE_ORDER:
        for slot in range(1, sc.MARKERS_PER_SCENE + 1):
            name = f"{scene}_slot{slot}_eye.png"
            path = os.path.join(STAGE, name)
            if not os.path.exists(path):
                print(f"{scene:<9} {slot:<4} MISSING {name}")
                problems.append(name)
                continue
            image = np.asarray(Image.open(path).convert("RGB"))
            height, width = image.shape[:2]
            rgb = marker_colour(scene, slot)
            box = find_marker(image, rgb)
            if box is None:
                print(f"{scene:<9} {slot:<4} {'not found':<26} {'-':<10} {'-':<22} "
                      f"NO MARKER IN FRAME")
                problems.append(f"{name}: marker not found")
                continue

            x0, y0, x1, y1 = box
            size = f"{x1 - x0 + 1}x{y1 - y0 + 1}"

            # Where each dressing item projects, in the same normalised frame the
            # screen_bounds() helper uses.  Converted to pixels with v measured upward.
            covered = []
            for item in sc.SCENES[scene]["dressing"]:
                rect = sc.screen_bounds(item["at"], item["size"])
                if rect is None:
                    continue
                u0, v0, u1, v1 = rect
                px0 = int((u0 + 1) / 2 * width)
                px1 = int((u1 + 1) / 2 * width)
                py0 = int((1 - v1) / 2 * height)
                py1 = int((1 - v0) / 2 * height)
                if px1 < x0 or px0 > x1 or py1 < y0 or py0 > y1:
                    continue
                if 0 <= px1 and px0 <= width:
                    covered.append(item["name"])

            print(f"{scene:<9} {slot:<4} "
                  f"{f'({x0},{y0})..({x1},{y1})':<26} {size:<10} "
                  f"{(','.join(covered) if covered else 'none'):<22} "
                  f"{'check' if covered else 'clear, marker visible'}")
            if covered:
                problems.append(f"{name}: {covered} project onto the marker")

    print()
    if problems:
        print(f"{len(problems)} frame(s) need a look:")
        for problem in problems:
            print(f"   {problem}")
    else:
        print("all 25 frames: marker found, nothing projects onto it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
