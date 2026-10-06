"""Every material a scene names must exist in the repository.

scene_builder.local_asset_for() falls back to the network when a URL's basename is not vendored,
and on a machine whose material library cache is not writable a material fetched that way arrives
without its shaders and renders as flat paint or noise.  The failure is silent: the scene still
looks like a scene, just not the one that was designed.  This test makes it loud.

Also checks that a scene's four surfaces are distinct files, because two surfaces sharing one
material is almost always a copy-paste slip rather than a decision.

    python3 test_bao_assets.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenes as sc  # noqa: E402
import scene_builder as sb  # noqa: E402


class Failure(Exception):
    pass


def check(condition, message: str) -> None:
    if not condition:
        raise Failure(message)


def test_every_material_is_vendored() -> None:
    missing = []
    for scene in sc.SCENE_ORDER:
        for surface, url in sc.SCENES[scene]["materials"].items():
            if sb.local_asset_for(url) is None:
                missing.append(f"{scene}/{surface} -> {os.path.basename(url)}")
    check(not missing,
          "material(s) named but not vendored, so they would silently fall back to flat "
          "paint or the network:\n    " + "\n    ".join(missing))
    total = sum(len(sc.SCENES[s]["materials"]) for s in sc.SCENE_ORDER)
    print(f"[ok] all {total} surface material(s) across {len(sc.SCENE_ORDER)} scenes resolve "
          f"to files in assets/isaac")


def test_surfaces_do_not_share_a_material() -> None:
    for scene in sc.SCENE_ORDER:
        names = [os.path.basename(u) for u in sc.SCENES[scene]["materials"].values()]
        check(len(set(names)) == len(names),
              f"{scene} uses one material for two surfaces: {names}")
    print("[ok] no scene reuses one material file for two different surfaces")


def test_the_laboratory_is_still_undecorated() -> None:
    """The laboratory is the running baseline: no materials, no dressing, and it must stay so.

    A decorated laboratory would not be an error in itself -- it would silently invalidate a
    sweep that is already running, because main.py re-reads this catalogue for every model.
    """
    spec = sc.SCENES["stage1.1"]
    check(not spec["materials"], f"stage1.1 gained materials: {list(spec['materials'])}")
    check(not spec["dressing"], f"stage1.1 gained dressing: {len(spec['dressing'])} items")
    print("[ok] stage1.1 is still plain: no materials and no dressing")


def main() -> int:
    tests = [
        test_every_material_is_vendored,
        test_surfaces_do_not_share_a_material,
        test_the_laboratory_is_still_undecorated,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Failure as failure:
            failures += 1
            print(f"[FAIL] {test.__name__}: {failure}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[ERROR] {test.__name__}: {exc!r}")
    if failures:
        print(f"\n{failures}/{len(tests)} checks FAILED")
        return 1
    print(f"\nall {len(tests)} asset checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
