"""Check that every vendored material can actually find its textures.

A material that binds but cannot find its textures renders as a flat colour, which looks like
"no textures anywhere" while the run log cheerfully reports mdl-local for every surface.  The
.mdl files reference their maps by relative name, so the question is mechanical: pull the texture
names out of each material we use, and check the files are there.

    python3 tools/check_material_textures.py
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scene_builder as sb  # noqa: E402
import scenes as sc  # noqa: E402

# MDL texture references appear as string literals with an image extension, sometimes wrapped in
# texture_2d() / texture_3d() and sometimes with gamma suffixes.  Pulling out anything that looks
# like an image file name is enough to check presence; missing ones are what matter.
IMAGE = re.compile(r"[\w./\\-]+\.(?:png|jpg|jpeg|exr|tif|tiff|hdr|tx)\b", re.IGNORECASE)


def references(mdl_path: str) -> list:
    with open(mdl_path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    found = []
    for match in IMAGE.findall(text):
        name = match.replace("\\", "/").split("/")[-1]
        if name and name not in found:
            found.append(name)
    return found


def search_roots(mdl_path: str) -> list:
    """Where an .mdl's relative texture references can legitimately resolve.

    The first version of this check looked only in the .mdl's own directory and reported 46
    references missing, which was wrong: the Isaac material folders keep their maps in a
    Textures/ subdirectory beside the material.  A false alarm here is expensive -- it says a
    scene has no textures when it has all of them -- so the search covers the material's folder
    and one level below it.
    """
    folder = os.path.dirname(mdl_path)
    roots = [folder, os.path.join(folder, "Textures")]
    roots += [os.path.join(folder, name) for name in os.listdir(folder)
              if os.path.isdir(os.path.join(folder, name))]
    return roots


def main() -> int:
    missing_total = 0
    checked = 0
    for scene in sc.SCENE_ORDER:
        spec = sc.SCENES[scene]
        if not spec["materials"]:
            print(f"{scene}: no materials (baseline)")
            continue
        print(f"\n=== {scene} ({spec['label']}) ===")
        for surface, url in spec["materials"].items():
            local = sb.local_asset_for(url)
            if not local:
                print(f"  {surface:<10} NOT VENDORED  {os.path.basename(url)}")
                missing_total += 1
                continue
            refs = references(local)
            roots = search_roots(local)
            absent = []
            for name in refs:
                if not any(os.path.exists(os.path.join(root, name)) for root in roots):
                    absent.append(name)
            checked += 1
            status = "ok" if not absent else f"MISSING {len(absent)}/{len(refs)}"
            print(f"  {surface:<10} {os.path.basename(local):<28} {len(refs):>3} texture ref(s)  {status}")
            for name in absent[:6]:
                print(f"      missing: {name}")
            missing_total += len(absent)
    print()
    print(f"checked {checked} material(s); {missing_total} missing texture reference(s)")
    return 1 if missing_total else 0


if __name__ == "__main__":
    sys.exit(main())
