"""Fetch a material set from NVIDIA's public Isaac bucket and vendor it into the repository.

Why this exists: the scenes must render from a clone.  Letting Isaac fetch a material by URL at
run time is what made the first previews come back flat and noisy -- the lab machine's material
library cache is not writable, so a material created from a URL arrives without its shaders.
The vendored copy is also what makes the render independent of the network and of that cache.

The download is large and mostly invisible detail: the warehouse set was 473 MB of 4K maps for a
512 px render.  Every image is therefore resampled through tools/shrink_textures.py as it lands,
which took that same set to 30 MB without touching a single file name -- and the file names are
what the .mdl files reference, so nothing breaks.

    python tools/fetch_assets.py --list Environments/Hospital
    python tools/fetch_assets.py --set Environments/Hospital Environments/Office
    python tools/fetch_assets.py --set Props/Beaker --long-edge 512

Run from the repository root.  Re-running is safe: existing files are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
PREFIX_ROOT = "Assets/Isaac/4.5/Isaac/"
DEST_ROOT = os.path.join("assets", "isaac")
EXTENSIONS = (".mdl", ".png", ".jpg", ".jpeg", ".exr",
              # Props are USD files, not MDLs: listing Props/Beaker showed five keys and zero
              # .mdl files, because every prop is a .usd.  Without these three extensions the
              # download step would have skipped every mesh and vendored only the thumbnails.
              ".usd", ".usda", ".usdc", ".usdz")
SKIP = (".thumbs/", "/.thumbs", "thumbnails")


def list_keys(prefix: str, limit: int = 4000) -> list:
    """Every key under a prefix, following continuation tokens."""
    keys, token = [], ""
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = BUCKET + "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=120) as response:
            body = response.read().decode("utf-8", "replace")
        root = ElementTree.fromstring(body)
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for node in root.findall(".//s3:Contents/s3:Key", namespace):
            keys.append(node.text)
        if len(keys) >= limit:
            break
        token_node = root.find(".//s3:NextContinuationToken", namespace)
        token = token_node.text if token_node is not None else ""
        if not token:
            break
    return keys


def download(key: str, dest: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(BUCKET + urllib.parse.quote(key), timeout=300) as response:
        with open(tmp, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    os.replace(tmp, dest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Vendor an Isaac material set.")
    parser.add_argument("--set", nargs="*", default=[],
                        help="paths under Assets/Isaac/4.5/Isaac/, e.g. Environments/Hospital")
    parser.add_argument("--list", default="", help="just list the keys under this path")
    parser.add_argument("--long-edge", type=int, default=512)
    parser.add_argument("--no-shrink", action="store_true")
    args = parser.parse_args()

    if args.list:
        keys = list_keys(PREFIX_ROOT + args.list.strip("/") + "/")
        material = [k for k in keys if k.endswith(".mdl")]
        meshes = [k for k in keys if k.lower().endswith((".usd", ".usda", ".usdc", ".usdz"))]
        print(f"{len(keys)} key(s): {len(material)} .mdl, {len(meshes)} USD under {args.list}")
        for key in meshes[:20]:
            print("  USD " + key[len(PREFIX_ROOT):])
        for key in material[:20]:
            print("  MDL " + key[len(PREFIX_ROOT):])
        return 0

    if not args.set:
        parser.print_help()
        return 1

    total_new = total_bytes = 0
    for name in args.set:
        prefix = PREFIX_ROOT + name.strip("/") + "/"
        try:
            keys = list_keys(prefix)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED to list {name}: {exc!r}")
            continue
        wanted = [k for k in keys
                  if k.lower().endswith(EXTENSIONS) and not any(s in k for s in SKIP)]
        print(f"{name}: {len(keys)} key(s), {len(wanted)} to fetch")
        got = 0
        for key in wanted:
            relative = key[len(PREFIX_ROOT):]
            dest = os.path.join(DEST_ROOT, *relative.split("/"))
            try:
                if download(key, dest):
                    got += 1
                    total_bytes += os.path.getsize(dest)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {relative}: {exc!r}")
        print(f"  fetched {got} new file(s)")
        total_new += got

    if not args.no_shrink:
        sys.path.insert(0, os.path.join("tools"))
        try:
            import shrink_textures

            for folder, _, files in os.walk(DEST_ROOT):
                for name in files:
                    if name.lower().endswith((".png", ".jpg", ".jpeg")):
                        shrink_textures.shrink(os.path.join(folder, name), args.long_edge)
            print(f"resampled every image under {DEST_ROOT} to fit {args.long_edge} px")
        except Exception as exc:  # noqa: BLE001
            print(f"shrink step skipped: {exc!r}")

    size = sum(os.path.getsize(os.path.join(f, n))
               for f, _, ns in os.walk(DEST_ROOT) for n in ns)
    print(f"fetched {total_new} file(s), {total_bytes / 1e6:.1f} MB new; "
          f"{DEST_ROOT} is now {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
