"""Measure the exposure of the rendered frames against the criterion already in the code.

environment.py records the acceptance test for this room:

    eye_start mean roughly 130-160 AND std above about 30

and a measured sweep behind it (light 500 -> mean 70.2; 1500 -> 137.3; 3000 -> 179.8;
7000 -> 220.0 with std 22.7, which is the washed-out signature).  So "the scene looks dim"
is not evidence to act on: the frame either meets that criterion or it does not, and the
numbers say which.  It also fixes what "brighten" can mean -- the mean has an upper bound,
so the lever is not simply intensity.

Reported per frame: mean, std, and the fraction of pixels above 240 (clipping).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

STAGE = r"C:\Users\asus\Desktop\科研狗之人机心理学\stage\prev"
SCENES = ("stage1.1", "stage1.2", "stage1.3", "stage1.4", "stage1.5")

# The criterion, quoted from environment.py rather than invented here.
MEAN_LOW, MEAN_HIGH, STD_MIN = 130.0, 160.0, 30.0


def main() -> int:
    print(f"criterion from environment.py: mean in [{MEAN_LOW:.0f}, {MEAN_HIGH:.0f}] "
          f"and std > {STD_MIN:.0f}")
    print()
    print(f"{'scene':<9} {'slot':<5} {'mean':>7} {'std':>7} {'>240':>7} {'<25':>7} verdict")
    print("-" * 62)
    rows = []
    for scene in SCENES:
        means = []
        for slot in range(1, 6):
            path = os.path.join(STAGE, f"{scene}_slot{slot}_eye.png")
            if not os.path.exists(path):
                print(f"{scene:<9} {slot:<5} MISSING")
                continue
            image = np.asarray(Image.open(path).convert("L")).astype(float)
            mean = image.mean()
            std = image.std()
            clipped = (image > 240).mean() * 100
            dark = (image < 25).mean() * 100
            ok = MEAN_LOW <= mean <= MEAN_HIGH and std > STD_MIN
            means.append(mean)
            print(f"{scene:<9} {slot:<5} {mean:>7.1f} {std:>7.1f} {clipped:>6.1f}% "
                  f"{dark:>6.1f}% {'meets criterion' if ok else 'OUTSIDE criterion'}")
        if means:
            rows.append((scene, float(np.mean(means))))
    print()
    print("per-scene mean of the five eye frames:")
    for scene, mean in rows:
        verdict = ("in range" if MEAN_LOW <= mean <= MEAN_HIGH
                   else ("too dark" if mean < MEAN_LOW else "too bright"))
        print(f"   {scene}: {mean:6.1f}   {verdict}")
    print()
    print("The eye frames are what the model is given, so this is the frame the criterion is")
    print("about.  A scene below the range is dim by the project's own definition; one above it")
    print("is heading for the washed-out signature (high mean, low std).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
