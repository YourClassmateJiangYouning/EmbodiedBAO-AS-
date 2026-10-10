"""Ask the material system two questions and print the answers as facts.

The complaint is "the scenes are boxes and colours, with no textures at all".  Two different things
could be true, and they need different fixes:

  1. the four surfaces bind an MDL, but the MDL never compiles into a shader graph, so the surface
     renders flat grey however many times it is bound.  The lab machine's material-library cache is
     read-only, which is a known suspect and has never actually been tested;
  2. the dressing items are painted flat colours and nothing else -- 30 of the 33 have no asset at
     all, so they are boxes by construction and no amount of material work changes them.

So this reports, from the built stage:

  * for each of the four surfaces: the bound material's prim path, its type, how many children it
    has, and every input its shader declares.  A working MDL material has a shader child with an
    ``info:mdl:sourceAsset`` and a set of inputs; a prim with no children is a placeholder.
  * whether an MDL can be compiled AT ALL on this machine, by executing the same command
    resolve_material() uses and reporting the exception verbatim if it fails.
  * for every dressing item: whether it has an asset, so the "boxes are boxes" part is a count and
    not an opinion.

    ISAAC_PY=/home/ybh/isaacsim/python.sh
    $ISAAC_PY -u tools/diagnose_materials.py stage1.5

It changes nothing and writes no images.  Its whole output is text.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", nargs="?", default="stage1.5")
    parser.add_argument("--level", type=int, default=0)
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import environment
        import scenes as sc

        env = environment.setup_scene(app, task_dict={
            "headless": True, "scene": args.scene, "level": args.level,
            "image_size": 512, "use_mdl": True, "hide_robot": True,
            "scene_parts": ["materials", "marker", "dressing"]})

        from pxr import UsdShade

        stage = env.stage
        report = env.scene_report or {}

        print("=" * 100)
        print("1. how each surface was resolved (from the scene report)")
        print("=" * 100)
        for surface, info in (report.get("materials") or {}).items():
            print(f"  {surface:<10} how={info.get('how'):<16} url={os.path.basename(str(info.get('url')))}"
                  f"  prim={info.get('prim')}")

        print()
        print("=" * 100)
        print("2. what the bound material actually IS, read off the stage")
        print("=" * 100)

        def describe(path: str, indent: str = "  ") -> None:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                print(f"{indent}{path}: NOT PRESENT")
                return
            print(f"{indent}{path}  type={prim.GetTypeName()}")
            children = prim.GetChildren()
            print(f"{indent}  children: {len(children)} "
                  f"{[c.GetName() for c in children]}")
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if name.startswith("inputs:") or name in ("info:mdl:sourceAsset",
                                                          "info:implementationSource"):
                    try:
                        value = attr.Get()
                    except Exception as error:  # noqa: BLE001
                        value = f"<read failed: {type(error).__name__}>"
                    print(f"{indent}    {name} = {value}")
            # A connection is a relationship, not a value, so Get() shows None for it.
            out = prim.GetAttribute("outputs:surface")
            if out and out.IsValid():
                try:
                    sources = out.GetConnectedSources()
                    print(f"{indent}    outputs:surface connected to: "
                          f"{[[s.source.GetPath() for s in group] for group in sources[0]]}")
                except Exception as error:  # noqa: BLE001
                    print(f"{indent}    outputs:surface connection unreadable: "
                          f"{type(error).__name__}: {error}")

        for path in ("/World/room_floor", "/World/Ground", "/World/room_far",
                     "/World/room_side_left", "/World/room_side_right"):
            describe(path)

        print()
        print("  material bindings found by traversal:")
        found = 0
        for prim in stage.Traverse():
            try:
                binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            except Exception:  # noqa: BLE001
                continue
            if binding and binding.GetPrim().IsValid():
                print(f"    {prim.GetPath()}  ->  {binding.GetPath()}")
                found += 1
                if found >= 12:
                    print("    ... stopping after 12")
                    break
        if not found:
            print("    NONE -- nothing in this stage is bound to a material")

        print()
        print("=" * 100)
        print("3. can an MDL be compiled on this machine at all?")
        print("=" * 100)
        surface_url = None
        for info in (report.get("materials") or {}).values():
            if info.get("url"):
                surface_url = info["url"]
                break
        print(f"  using {surface_url}")
        try:
            import omni.kit.commands  # type: ignore

            import scene_builder

            source = scene_builder.local_asset_for(surface_url) or surface_url
            print(f"  local_asset_for -> {source}")
            omni.kit.commands.execute(
                "CreateMdlMaterialPrimCommand", mtl_url=source,
                mtl_name="DiagnoseMdl", mtl_path="/World/DiagnoseMdl")
            prim = stage.GetPrimAtPath("/World/DiagnoseMdl")
            print(f"  command returned without raising; prim present: {prim.IsValid()}")
            if prim.IsValid():
                describe("/World/DiagnoseMdl")
        except Exception:  # noqa: BLE001
            print("  the command RAISED:")
            print("    " + traceback.format_exc().replace("\n", "\n    "))

        print()
        print("=" * 100)
        print("4. dressing items: asset or flat box")
        print("=" * 100)
        items = sc.SCENES[args.scene]["dressing"]
        with_asset = [i for i in items if i.get("asset")]
        print(f"  {len(items)} item(s); {len(with_asset)} name an asset, "
              f"{len(items) - len(with_asset)} are painted boxes and nothing else")
        for item in items:
            print(f"    {item['name']:<16} asset={os.path.basename(item['asset']) if item.get('asset') else '-'}")
        print("  textures on flat-painted items: 0, by construction -- paint() sets displayColor.")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
