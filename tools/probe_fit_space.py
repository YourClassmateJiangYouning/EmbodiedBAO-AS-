"""Print, for a real place_dressing run, WHICH space the fit's measurement is taken in.

The open question left by tools/probe_prop_fit.py: inside ``place_dressing``,
``ComputeRelativeBound(proxy, parent)`` with the item's scale forced to 1 returns (1, 1, 1) for the
rack, while the same reading on a hand-built stage returns its real extent (108.019, 199.809,
301.000).  This prints, per item, the four candidate readings side by side:

    relative(proxy, parent)        what fit_reference_to_local_unit currently divides by
    relative(proxy, root)          the dressing root, whose bound is world units * scale
    world corners from extents     extent x ComputeLocalToWorldTransform, summed over the meshes
    item / parent / proxy paths    so a wrong ancestor is visible by name

It writes nothing to the stage and nothing to the repository; the report goes to stdout.

    ISAAC_PY=/home/ybh/isaacsim/python.sh
    $ISAAC_PY -u tools/probe_fit_space.py 2>&1 | tee diag_logs/fit_space.log
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

SAMPLE = ("mug_a2", "banana", "shelf_left", "checkout", "clock", "bench_left", "pallets")


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Gf, Usd, UsdGeom

        import scenes as sc
        import scene_builder as sb

        index = {}
        for scene in sc.SCENE_ORDER:
            for item in sc.SCENES[scene]["dressing"]:
                index.setdefault(item["name"], (scene, item))

        for scene in sc.SCENE_ORDER:
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.Xform.Define(stage, "/World")
            root = UsdGeom.Xform.Define(stage, "/World/Dressing").GetPrim()
            records = sb.place_dressing(stage, scene)
            for record in records:
                name = record["name"]
                if name not in SAMPLE:
                    continue
                path = f"/World/Dressing/{name}"
                prop_path = f"{path}/prop"
                proxy = stage.GetPrimAtPath(prop_path)
                if not proxy or not proxy.IsValid():
                    print(f"{name:12s} no prop prim")
                    continue
                item_prim = stage.GetPrimAtPath(path)
                parent = proxy.GetParent()
                declared = tuple(float(v) for v in record["declared_extent"])

                # (a) the reading the fit takes today
                c1 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                r1 = c1.ComputeRelativeBound(proxy, parent).ComputeAlignedRange()
                a = tuple(round(float(r1.GetMax()[i] - r1.GetMin()[i]), 6) for i in range(3)) \
                    if not r1.IsEmpty() else None
                # (b) relative to the dressing root: units times the item's scale
                c2 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                r2 = c2.ComputeRelativeBound(proxy, root).ComputeAlignedRange()
                b = tuple(round(float(r2.GetMax()[i] - r2.GetMin()[i]), 6) for i in range(3)) \
                    if not r2.IsEmpty() else None
                # (c) the meshes' own extents, transformed to world, united by hand
                lo = [float("inf")] * 3
                hi = [float("-inf")] * 3
                meshes = 0
                for prim in Usd.PrimRange(proxy):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    extent = UsdGeom.Boundable(prim).GetExtentAttr().Get()
                    if not extent:
                        continue
                    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    for corner in (extent[0], extent[1]):
                        p = xf.Transform(Gf.Vec3d(*corner))
                        for i in range(3):
                            lo[i] = min(lo[i], float(p[i]))
                            hi[i] = max(hi[i], float(p[i]))
                    meshes += 1
                c = tuple(round(hi[i] - lo[i], 6) for i in range(3)) if meshes else None

                scale_ops = [(o.GetOpName(), tuple(round(float(v), 5) for v in o.Get()))
                             for o in UsdGeom.Xformable(item_prim).GetOrderedXformOps()]
                print(f"--- {scene}/{name}")
                print(f"    declared          = {tuple(round(v, 5) for v in declared)}")
                print(f"    item path         = {path}")
                print(f"    proxy path        = {proxy.GetPath()}")
                print(f"    parent path       = {parent.GetPath() if parent and parent.IsValid() else None}")
                print(f"    item scale ops    = {scale_ops}")
                print(f"    (a) rel(parent)   = {a}")
                print(f"    (b) rel(root)     = {b}")
                print(f"    (c) extent x l2w  = {c}   ({meshes} mesh(es))")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
