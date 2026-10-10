"""Vendor a set of assets from NVIDIA's public bucket into the repository.

tools/fetch_assets.py does this for one root prefix (Assets/Isaac/4.5/Isaac/), which is why the
park could not be furnished: the vegetation and the outdoor furniture live elsewhere in the same
bucket.

    python tools/fetch_bucket_set.py --list Assets/Vegetation/Trees
    python tools/fetch_bucket_set.py --set Assets/Vegetation/Trees/Chinese_Juniper
    python tools/fetch_bucket_set.py --set Assets/Isaac/4.5/Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/common_assets/props_general/bench_curved_01 --long-edge 512

Files land under assets/bucket/<key with Assets/ stripped>, and .thumbs are skipped.  Re-running
skips what is already present.  Images are resampled through tools/shrink_textures.py, because
these sets ship 2K-4K maps for a 512 px render.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
DEST = os.path.join("assets", "bucket")
EXTENSIONS = (".mdl", ".png", ".jpg", ".jpeg", ".exr", ".usd", ".usda", ".usdc", ".usdz")


def list_keys(prefix: str) -> list:
    keys, token = [], ""
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(query),
                                    timeout=90) as response:
            tree = ElementTree.fromstring(response.read())
        for node in tree.findall(NS + "Contents"):
            key = node.find(NS + "Key").text
            size = int(node.find(NS + "Size").text)
            if ".thumbs" in key or key.endswith("/"):
                continue
            if key.lower().endswith(EXTENSIONS):
                keys.append((key, size))
        if tree.findtext(NS + "IsTruncated") != "true":
            break
        token = tree.findtext(NS + "NextContinuationToken") or ""
    return keys


def local_path(key: str) -> str:
    return os.path.join(DEST, *key.split("/"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="append", default=[])
    parser.add_argument("--set", nargs="+", action="append", default=[])
    parser.add_argument("--long-edge", type=int, default=512)
    parser.add_argument("--root", default="", help="prefix to strip for the local path")
    args = parser.parse_args()

    if not args.list and not args.set:
        print(__doc__)
        return 2

    for prefix in args.list:
        keys = list_keys(prefix)
        total = sum(s for _k, s in keys)
        print(f"{prefix}: {len(keys)} file(s), {total / 1e6:.1f} MB")
        for key, size in sorted(keys, key=lambda pair: pair[1])[:40]:
            print(f"   {size / 1e6:9.2f} MB  {key[len(prefix):].lstrip('/') or key}")

    fetched = 0
    bytes_new = 0
    for group in args.set:
        for prefix in group:
            keys = list_keys(prefix)
            missing = [(k, s) for k, s in keys if not os.path.exists(local_path(k))]
            print(f"{prefix}: {len(keys)} file(s), {len(missing)} to fetch")
            for key, size in missing:
                target = local_path(key)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                url = BUCKET + urllib.parse.quote(key, safe="/")
                with urllib.request.urlopen(url, timeout=180) as response:
                    with open(target, "wb") as handle:
                        handle.write(response.read())
                fetched += 1
                bytes_new += size
            print(f"  fetched {len(missing)} new file(s)")

    if fetched:
        print(f"fetched {fetched} file(s), {bytes_new / 1e6:.1f} MB; "
              f"{DEST} is now "
              f"{sum(os.path.getsize(os.path.join(f, n)) for f, _d, fs in os.walk(DEST) for n in fs) / 1e6:.1f} MB")
        script = os.path.join("tools", "shrink_textures.py")
        if os.path.exists(script):
            print(f"running {script} --long-edge {args.long_edge} --root {DEST}")
            # --root is required: shrink_textures defaults to assets/isaac and its positional
            # argument list is empty, so calling it with a bare path exits 1 with a usage error.
            subprocess.run([sys.executable, script, "--long-edge", str(args.long_edge),
                            "--root", DEST], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
