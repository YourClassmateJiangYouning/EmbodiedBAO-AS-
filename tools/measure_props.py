"""Measure each referenced prop against the box the item declares.

Why this exists: place_dressing puts the prop reference on a box of the declared size but does
NOT scale the prop.  The declared size is what the occlusion and walking-band checks are
written against, so it must not move; a prop's own authored size is a fact about the asset.
The two only agree by luck, and nothing on the development machine can measure it -- there is
no pxr here.  So this runs on the lab machine, where the question "is the pallet actually
1.2 m across, or 12 cm?" can be answered with numbers instead of a squint.

It does not change anything.  Run it, read the ratios, and then decide per prop whether to
scale, to move the item, or to leave it.

    ISAAC_PY=/home/ybh/isaacsim/python.sh
    $ISAAC_PY tools/measure_props.py 2>&1 | tee prop_measurements.txt
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scene_builder as sb  # noqa: E402
import scenes as sc  # noqa: E402


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        targets = []
        for scene in sc.SCENE_ORDER:
            for item in sc.SCENES[scene]["dressing"]:
                local = sb.prop_asset_for(item.get("asset"))
                if local and item["mount"] == "floor":
                    targets.append((scene, item, local))

        print(f"{len(targets)} floor item(s) name a vendored prop")
        print()
        print(f"{'scene':<9} {'item':<15} {'prop file':<34} {'declared (x,y,z) m':<22} "
              f"{'measured (x,y,z) m':<22} {'ratio x,y,z':<20} verdict")
        print("-" * 140)

        for scene, item, local in targets:
            path = f"/World/Probe/{scene}/{item['name']}"
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            prim.GetReferences().AddReference(local)

            # Extent is the authored local bounds; the world bounds need the stage to compose
            # the reference, which is what GetAttribute on extent gives once it has.
            box = UsdGeom.BoxCache(UsdGeom.BoxCache.CreateBoxCache(prim))
            rng = box.ComputeWorldBounds().ComputeAlignedRange()
            if rng.IsEmpty():
                print(f"{scene:<9} {item['name']:<15} {os.path.basename(local):<34} "
                      f"{str(tuple(round(v, 3) for v in item['size'])):<22} "
                      f"{'EMPTY':<22} {'-':<20} no geometry -- wrapper or broken layers")
                continue

            lo, hi = rng.GetMin(), rng.GetMax()
            measured = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
            # The item is declared in the user frame (x, height, lateral); the stage stores
            # (x, lateral, height), so compare height against z and lateral against y.
            declared = sc.to_world_size(item["size"])
            ratios = tuple(
                (measured[i] / declared[i]) if declared[i] else float("nan")
                for i in range(3)
            )
            ok = all(0.8 <= r <= 1.25 for r in ratios if r == r)
            print(f"{scene:<9} {item['name']:<15} {os.path.basename(local):<34} "
                  f"{str(tuple(round(v, 3) for v in declared)):<22} "
                  f"{str(tuple(round(v, 3) for v in measured)):<22} "
                  f"{str(tuple(round(r, 2) for r in ratios)):<20} "
                  f"{'fits the box' if ok else 'DIFFERS from the box'}")

        print()
        print("A ratio near 1.0 means the prop already matches the volume the fixture checks")
        print("were written against.  A ratio far from 1 means either the prop needs scaling")
        print("or the declared size needs revisiting -- a decision that needs these numbers.")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
