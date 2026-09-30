"""Shrink the vendored Isaac textures so the asset set can live in the repository.

Why this exists: the scenes must work from a clone, so the materials have to travel with the
repository -- the lab machine cannot fetch them at run time (its material-library cache is not
writable, and fetching is what made the first previews flat).  But the warehouse's texture set
as downloaded is 473 MB in 163 PNGs, one of them 45 MB, which no repository should carry and
every clone would pay for.

The render input is 512 px and the diagnostic previews 1024, so a 4K or 8K diffuse map is
detail nobody can see.  Every image is resampled to fit LONG_EDGE and re-encoded; file names and
relative paths are untouched, which is what matters, because the .mdl files reference their
textures by relative name and never by size.

Run from the repository root:  python tools/shrink_textures.py [--long-edge 1024]
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.join("assets", "isaac")
DEFAULT_LONG_EDGE = 1024


def shrink(path: str, long_edge: int) -> tuple:
    """Resample one image in place.  Returns (before_bytes, after_bytes, changed)."""
    from PIL import Image

    before = os.path.getsize(path)
    with Image.open(path) as image:
        width, height = image.size
        if max(width, height) <= long_edge:
            return before, before, False
        scale = long_edge / float(max(width, height))
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized = image.resize(size, Image.LANCZOS)
        # Keep the mode sane for the maps that carry data rather than colour: normal and ORM
        # maps must not be quantised, and PNG is lossless either way, so nothing is converted
        # to JPEG.  optimise=True costs time and saves a little.
        resized.save(path, format="PNG", optimize=True)
    return before, os.path.getsize(path), True


def main() -> int:
    parser = argparse.ArgumentParser(description="Shrink vendored textures.")
    parser.add_argument("--long-edge", type=int, default=DEFAULT_LONG_EDGE)
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()
    if not os.path.isdir(args.root):
        print(f"no such directory: {args.root}")
        return 1
    total_before = total_after = 0
    changed = skipped = 0
    for folder, _, files in os.walk(args.root):
        for name in sorted(files):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(folder, name)
            before, after, did = shrink(path, args.long_edge)
            total_before += before
            total_after += after
            if did:
                changed += 1
                if before - after > 4 * 1024 * 1024:
                    print(f"  {name}: {before // 1024} KB -> {after // 1024} KB")
            else:
                skipped += 1
    print(f"resampled {changed}, already small {skipped}")
    print(f"total {total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
