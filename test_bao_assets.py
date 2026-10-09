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


def test_every_named_prop_that_can_be_referenced_is_vendored() -> None:
    """A prop the catalogue names must be a vendored .usd, or be a declared box.

    scene_builder.prop_asset_for() resolves a floor item's ``asset`` to a file under
    assets/isaac, refuses anything else, and place_dressing() then references it on top of the
    item's box.  A prop that resolves to nothing is not an error in itself -- the box is the
    geometry every fixture check is written against -- but it must be VISIBLE, because "this
    item is still a cube" is exactly the kind of thing that otherwise gets discovered from a
    screenshot months later.  So this test prints what resolved and what did not, and requires
    that at least one prop resolved, which would fail if the path prefix or the .usd suffix
    handling broke.

    The two unresolved names are named here rather than tolerated silently: MI_SignB.mdl and
    M_TrafficCone.mdl are MATERIALS from the warehouse set, kept on the item as the note of
    what it is meant to look like.  They are correctly refused, and a test that simply
    demanded "every asset resolves" would push someone to point them at a mesh that is not
    what they name.
    """
    resolved, refused = [], []
    for scene in sc.SCENE_ORDER:
        for item in sc.SCENES[scene]["dressing"]:
            asset = item.get("asset")
            if not asset:
                continue
            local = sb.prop_asset_for(asset)
            (resolved if local else refused).append(
                (scene, item["name"], os.path.basename(asset)))

    check(bool(resolved),
          "no catalogue asset resolved to a vendored .usd at all; prop_asset_for() or the "
          "assets/isaac layout is broken")

    check(not [r for r in refused if not r[2].endswith(".mdl")],
          f"these non-material assets resolved to nothing and will render as plain boxes: "
          f"{[r for r in refused if not r[2].endswith('.mdl')]}")

    print(f"[ok] {len(resolved)} prop(s) reference a vendored mesh: "
          f"{[f'{n} -> {a}' for _s, n, a in resolved]}")
    if refused:
        print(f"     {len(refused)} name a material and stay boxes (by design): "
              f"{[f'{n} -> {a}' for _s, n, a in refused]}")


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
        test_every_named_prop_that_can_be_referenced_is_vendored,
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
