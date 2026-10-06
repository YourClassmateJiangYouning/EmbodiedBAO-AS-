"""Shrink every image under a directory tree, skipping the ones Pillow cannot read.

Written after a 6.5 GB fetch left 3.9 GB of raw 4K maps on disk because one image with a broken
data stream ended the whole resize pass.  This version reports each unreadable file and carries
on, which is the behaviour that should have been there from the start: a material set fetched
from the internet is allowed to contain one bad file, and that must not cost the other 2609.

    python tools/shrink_tree.py --root assets/isaac --long-edge 512
"""

from __future__ import annotations

import argparse
import os
import sys

SKIP_DIRS = (".thumbs",)


def shrink(path: str, long_edge: int) -> int:
    """Returns 1 if the file was resized, 0 if it was left alone."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if max(width, height) <= long_edge:
                return 0
            scale = long_edge / float(max(width, height))
            size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            resized = image.resize(size, Image.LANCZOS)
            resized.save(path, format="PNG", optimize=True)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {os.path.relpath(path)}: {exc!r}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Shrink a tree of images in place.")
    parser.add_argument("--root", default=os.path.join("assets", "isaac"))
    parser.add_argument("--long-edge", type=int, default=512)
    args = parser.parse_args()
    if not os.path.isdir(args.root):
        print(f"no such directory: {args.root}")
        return 1

    before = after = resized = 0
    for folder, dirs, files in os.walk(args.root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(files):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(folder, name)
            try:
                size_before = os.path.getsize(path)
                if shrink(path, args.long_edge):
                    resized += 1
                before += size_before
                after += os.path.getsize(path)
            except OSError:
                continue
    print(f"resized {resized} file(s)")
    print(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
