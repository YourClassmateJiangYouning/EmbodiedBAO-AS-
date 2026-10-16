"""Measure what the real dressing path draws, per axis, so the prop-size fix stays fixed.

WHY THIS EXISTS.  ``fit_reference_to_local_unit`` used to return a uniform ``min(1/native)``, so only
the axis whose bound was largest filled its declared box and every other axis was drawn at
``native_i / native_max`` of it: a 9.2 cm mug at 8.5 mm, a 14.25 cm pallet 1.7 cm thick, a 2.47 m
packing table with a 23 cm top, a clock face at 9:1 where the asset is 3:1, a safety railing 2.4 mm
thick.  SCENE_SIZE_AUDIT.md section 1 has the full table; 34 of the 61 resolved props were below half
of at least one axis.

It was fixed by making the factor per axis, ``k_i = 1/native_i``, after measuring the three candidate
formulas on the real stage.  The earlier per-axis attempt (reverted as ``bfd5be7``) had failed because
it also multiplied the offset by the scale a second time, which drew the mug 9.2 m across; the
measurement that separated the two is in the docstring of ``fit_reference_to_local_unit``.

WHAT THIS DOES.  It runs the REAL ``scene_builder.place_dressing`` for every scene on an in-memory
stage and reads each resolved prop's extent back relative to the dressing root, which has no
transform -- so the numbers are metres and directly comparable with the declared box.  Per item it
prints ``drawn / declared`` on each axis and the fit scale actually applied:

    fill 1.000 on every axis   the prop is drawn at the size it declares
    fill < 1.000               drawn smaller than its box -- the defect this tool exists for
    fill > 1.000               overflows its box, which is clipping by construction

It is kept as a tool because that regression is invisible to every other check: the declared sizes
were right, the boxes were right, and only the drawn geometry was wrong.

Read-only with respect to the repository; the stage is in memory.  Note that Isaac Sim needs the
sandbox's GPU access -- under a restricted file sandbox Kit segfaults late in startup.

    ISAAC_PY=/home/ybh/isaacsim/python.sh
    $ISAAC_PY -u tools/probe_prop_fit.py 2>&1 | tee diag_logs/prop_fit.log
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

TOLERANCE = 0.02


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd, UsdGeom

        import scenes as sc
        import scene_builder as sb

        bad, checked = [], 0
        for scene in sc.SCENE_ORDER:
            # A FRESH STAGE PER SCENE.  Reusing one stage made this tool lie: /World/Dressing was
            # re-defined for every scene, so the previous scene's items were still on the stage when
            # the next scene's props were measured, and `ComputeRelativeBound` against a root that
            # other prims had already been placed under returned the wrong space -- three mugs came
            # back at 100x their declared box while the same items measured exactly right on a clean
            # stage.  A measurement tool whose answer depends on what ran before it is worse than no
            # tool, which is the rule this project already wrote down for scratch scripts.
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.Xform.Define(stage, "/World")
            root = UsdGeom.Xform.Define(stage, "/World/Dressing").GetPrim()
            records = sb.place_dressing(stage, scene)
            print(f"--- {scene}")
            for record in records:
                if not record.get("used_asset"):
                    continue
                proxy = stage.GetPrimAtPath(f"/World/Dressing/{record['name']}/prop")
                if not proxy or not proxy.IsValid():
                    print(f"    {record['name']:18s} prop prim not on the stage")
                    continue
                fresh = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                rng = fresh.ComputeRelativeBound(proxy, root).ComputeAlignedRange()
                if rng.IsEmpty():
                    print(f"    {record['name']:18s} empty bound")
                    continue
                lo, hi = rng.GetMin(), rng.GetMax()
                drawn = tuple(float(hi[a] - lo[a]) for a in range(3))
                declared = tuple(float(v) for v in record["declared_extent"])
                fill = tuple(drawn[a] / declared[a] if declared[a] else 0.0 for a in range(3))
                checked += 1
                flag = "   " if all(abs(f - 1.0) <= TOLERANCE for f in fill) else "!! "
                if flag.strip():
                    bad.append(f"{scene}/{record['name']} fills "
                               f"{tuple(round(f, 3) for f in fill)}")
                print(f"  {flag}{record['name']:18s} declared {tuple(round(v, 4) for v in declared)!s:28s}"
                      f" drawn {tuple(round(v, 4) for v in drawn)!s:28s}"
                      f" fills {tuple(round(f, 3) for f in fill)!s:22s}"
                      f" fit {tuple(round(float(v), 5) for v in (record.get('fit_scale') or ()))}")
        print()
        print(f"{checked} resolved prop(s) measured; {len(bad)} with a fill ratio outside "
              f"1.000 +/- {TOLERANCE}")
        for line in bad:
            print(f"  {line}")
        return 1 if bad else 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
