"""Find the marker in each rendered frame by its own colour, and report where it is.

The first version of this was wrong in two ways, both of which produced a confident table of
nonsense, so both are recorded here:

  * it took the 0.1% nearest pixels to the marker's colour -- 1024 pixels of a 1024x1024 frame --
    and then reported their bounding box.  A marker is 13-25 px across, so the box came out as
    the whole frame whenever the nearest 1024 pixels were spread over the scene.  It now finds
    the largest CONNECTED group of pixels within a fixed colour distance, which is a marker-
    shaped thing or nothing.
  * it compared colours with a sum of absolute differences and a 180 threshold, i.e. "within 60
    per channel".  Measured: the palette's red is (217, 38, 31) and the concrete walls are grey,
    so a 60-per-channel window admits a great deal of grey at the dark end.  Euclidean distance
    in RGB with a tight threshold is used instead.

What this CANNOT do is decide occlusion by projecting the dressing: scenes.screen_bounds()
models the camera at (0.5, 1.68, 0) looking along +x, and the preview's eye camera sits on the
robot, laterally offset by -0.5.  Those are different cameras, so a projection-based occlusion
verdict for these frames is not available from that helper.  What is available, and is what
matters, is whether the marker occupies a plausible patch of its own colour: a marker behind a
box shows up as no pixels, not as a wrong location.
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


def marker_rgb(scene: str, slot: int):
    _shape, key = sc.MARKERS[scene][slot - 1]
    return np.array([c * 255.0 for c in sc.COLOURS[key]])


def largest_blob(mask: np.ndarray) -> tuple | None:
    """Bounding box of the largest 4-connected group of True, or None.

    Depth-first flood fill over the mask.  Written out rather than pulled from scipy, which is
    not installed here, and small enough that it does not need to be clever.
    """
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = None
    best_size = 0
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys, xs):
        if seen[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        seen[start_y, start_x] = True
        min_y = max_y = start_y
        min_x = max_x = start_x
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            min_y, max_y = min(min_y, y), max(max_y, y)
            min_x, max_x = min(min_x, x), max(max_x, x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if size > best_size:
            best_size = size
            best = (min_x, min_y, max_x, max_y)
    return best, best_size


def main() -> int:
    print(f"{'scene':<9} {'slot':<4} {'marker':<14} {'blob px':<26} {'size':<10} verdict")
    print("-" * 92)
    missing = []
    for scene in sc.SCENE_ORDER:
        for slot in range(1, sc.MARKERS_PER_SCENE + 1):
            path = os.path.join(STAGE, f"{scene}_slot{slot}_eye.png")
            if not os.path.exists(path):
                print(f"{scene:<9} {slot:<4} MISSING")
                missing.append(f"{scene} slot {slot}")
                continue
            image = np.asarray(Image.open(path).convert("RGB")).astype(float)
            rgb = marker_rgb(scene, slot)
            distance = np.sqrt(((image - rgb) ** 2).sum(axis=2))
            mask = distance < 60.0
            blob, size = largest_blob(mask)
            name = sc.MARKERS[scene][slot - 1][0]
            if blob is None or size < 4:
                print(f"{scene:<9} {slot:<4} {name:<14} {'none':<26} {'-':<10} "
                      f"NOT VISIBLE ({int(mask.sum())} px within 60)")
                missing.append(f"{scene} slot {slot} ({name}): {int(mask.sum())} px")
                continue
            x0, y0, x1, y1 = blob
            w, h = x1 - x0 + 1, y1 - y0 + 1
            # A marker is roughly 13-25 px of a 1024 frame at 15.48 m.  A blob far outside that
            # is the palette colliding with scene colour, not a marker.
            plausible = 4 <= w <= 90 and 4 <= h <= 90
            print(f"{scene:<9} {slot:<4} {name:<14} "
                  f"{f'({x0},{y0})..({x1},{y1})':<26} {f'{w}x{h} ({size} px)':<10} "
                  f"{'plausible marker' if plausible else 'TOO BIG -- colour collision'}")
            if not plausible:
                missing.append(f"{scene} slot {slot} ({name}): blob {w}x{h}")
    print()
    if missing:
        print(f"{len(missing)} frame(s) without a plausible marker:")
        for item in missing:
            print(f"   {item}")
    else:
        print("all 25 frames contain a plausible marker blob")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
