"""Crop and upscale the regions where the referenced props should be.

The eye frame is what the model sees, and the two referenced props sit close to the camera, but
at 1024 px they occupy only a few dozen pixels.  This makes them readable without touching the
originals.

It also reports, for a chosen crop, the per-region colour statistics: a prop that failed to
reference leaves a flat-painted box, and a prop whose material did not arrive renders black.
Both are visible in the numbers before they are visible by eye.
"""
from __future__ import annotations

import os
import sys

from PIL import Image

STAGE = r"C:\Users\asus\Desktop\科研狗之人机心理学\stage\prev"
OUT = r"D:\EmInAI\EmbodiedBAO(AS)\render_check"

# name -> (left, top, right, bottom, scale)
CROPS = {
    "s12_near_left": (0, 520, 420, 900, 3),        # pallets side, x 2.6 lateral +2.05
    "s12_near_right": (600, 520, 900, 900, 3),     # klt_bins side, x 4.0 lateral -2.00
    "s12_mid": (150, 240, 880, 620, 2),            # both walls' opening area
    "s13_near_left": (0, 480, 460, 900, 3),        # cabinet at x 7.0 lateral +2.05
    "s13_mid": (150, 240, 880, 620, 2),
}

SOURCES = {
    "s12_near_left": "stage1.2_slot1_eye.png",
    "s12_near_right": "stage1.2_slot1_eye.png",
    "s12_mid": "stage1.2_slot1_eye.png",
    "s13_near_left": "stage1.3_slot1_eye.png",
    "s13_mid": "stage1.3_slot1_eye.png",
}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for label, (l, t, r, b, scale) in CROPS.items():
        source = os.path.join(STAGE, SOURCES[label])
        if not os.path.exists(source):
            print(f"{label}: MISSING {source}")
            continue
        image = Image.open(source).convert("RGB")
        crop = image.crop((l, t, r, b))
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.NEAREST)
        dest = os.path.join(OUT, f"{label}.png")
        crop.save(dest)

        # Colour statistics of the crop, to tell a flat-painted box from a textured prop.
        small = image.crop((l, t, r, b)).convert("RGB")
        pixels = list(small.getdata())
        n = len(pixels)
        mean = tuple(round(sum(p[i] for p in pixels) / n, 1) for i in range(3))
        # How much of the crop is near-black: a prop with no material arrives black.
        black = sum(1 for p in pixels if max(p) < 25) / n
        # Colour spread: one dominant hue with no variation means flat paint.
        spread = tuple(round(max(p[i] for p in pixels) - min(p[i] for p in pixels), 1)
                       for i in range(3))
        print(f"{label:<16} {source:<28} crop {crop.size[0]}x{crop.size[1]}  "
              f"mean {mean}  spread {spread}  near-black {black * 100:.1f}%")
    print(f"\nwrote crops to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
