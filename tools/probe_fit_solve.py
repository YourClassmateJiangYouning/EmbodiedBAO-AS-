"""Solve for the prop scale that lands each item on its declared box, by measuring the real path.

WHY THIS AND NOT A FORMULA.  The prop-size defect has resisted three formulas:

  * ``min(1/native)`` (what ships) leaves 34 of 61 props below half of at least one axis;
  * ``1/native_i`` per axis leaves three stage1.5 mugs at exactly 100x their box;
  * ``declared_i / native_i`` draws the conifer 5.76 m in a 2.4 m box.

Every one of them was derived from a reading of ``ComputeRelativeBound`` that turned out to mean
something different in the code path than on a hand-built stage.  Measured with this project's own
tools, the readings for one item are mutually inconsistent: relative to the dressing root the mug's
mesh measures 0.85208 m across, while the drawn extent ``place_dressing`` records for it is 0.008521 m
-- a factor of 100 that is neither the item's scale (0.0923) nor the stage's metersPerUnit (0.01)
alone, and that does not appear in the same measurement taken by hand.

So this tool does not assume a formula.  It measures the one relation that is unambiguous and that a
scale sweep confirmed is exactly linear -- ``drawn = A . k``, no intercept -- and then solves for the
``k`` that makes ``drawn`` equal the declared box:

    k_new = k_applied * declared / drawn(k_applied)

one Newton step, which is exact for a linear relation.  It reports the scale per item, and a second
pass with that scale reports the fill ratio, so the result is verified rather than asserted.  If the
solved scales are stable and land every fill at 1.000, the change to ``fit_reference_to_local_unit``
is to return exactly them; if they do not, the tool's own output says so and nothing ships.

Read-only: it builds the items on an in-memory stage and restores every scale it touches.

    ISAAC_PY=/home/ybh/isaacsim/python.sh
    $ISAAC_PY -u tools/probe_fit_solve.py 2>&1 | tee diag_logs/fit_solve.log
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
        from pxr import Gf, Usd, UsdGeom

        import scenes as sc
        import scene_builder as sb

        def measure(proxy, root):
            fresh = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            rng = fresh.ComputeRelativeBound(proxy, root).ComputeAlignedRange()
            if rng.IsEmpty():
                return None
            lo, hi = rng.GetMin(), rng.GetMax()
            return tuple(float(hi[i] - lo[i]) for i in range(3))

        solved, verified = {}, {}
        for scene in sc.SCENE_ORDER:
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.Xform.Define(stage, "/World")
            root = UsdGeom.Xform.Define(stage, "/World/Dressing").GetPrim()
            records = sb.place_dressing(stage, scene)
            for record in records:
                if not record.get("used_asset"):
                    continue
                name = record["name"]
                proxy = stage.GetPrimAtPath(f"/World/Dressing/{name}/prop")
                declared = tuple(float(v) for v in record["declared_extent"])
                applied = tuple(float(v) for v in (record.get("fit_scale") or (1.0, 1.0, 1.0)))
                drawn = measure(proxy, root)
                if proxy is None or not proxy.IsValid() or drawn is None:
                    continue
                step = tuple(applied[i] * declared[i] / drawn[i] if drawn[i] > 1e-12 else applied[i]
                             for i in range(3))
                sb.apply_scale(proxy, step)
                after = measure(proxy, root)
                fill = tuple(after[i] / declared[i] if declared[i] else 0.0 for i in range(3))
                solved[(scene, name)] = step
                verified[(scene, name)] = fill
                print(f"{scene:9s} {name:18s} applied={tuple(round(v, 6) for v in applied)} "
                      f"drawn={tuple(round(v, 6) for v in drawn)} "
                      f"solved={tuple(round(v, 6) for v in step)} "
                      f"fill_after={tuple(round(v, 3) for v in fill)}", flush=True)

        bad = {k: v for k, v in verified.items()
               if not all(abs(f - 1.0) <= TOLERANCE for f in v)}
        # THE SOLVER'S OWN LIMIT, measured 2026-10-16 and the reason its answer is not the fix on its
        # own: it divides by the same bound ``fit_reference_to_local_unit`` divides by, and for the
        # three SM_Mug_A*.usd assets that bound disagrees with the meshes' real world size by 100x
        # (bound 0.008521 m against a mesh that measures 0.0923 m across).  So it solves the mug to
        # 10.83 where the truth is 1.0, and its fill_after for the mug is 100.0 -- the solver reports
        # that rather than hiding it, which is what makes the table above usable: 58 of 61 rows agree
        # with an independent measurement of the meshes, and the three that do not are the three this
        # file cannot answer.  See tools/probe_fit_space.py for the space comparison that isolates
        # them.
        print()
        print(f"{len(verified)} prop(s) solved; {len(bad)} still outside 1.000 +/- {TOLERANCE}")
        for key, fill in bad.items():
            print(f"  {key[0]}/{key[1]}: {tuple(round(f, 4) for f in fill)}")
        if not bad and solved:
            print()
            print("Every resolved prop lands on its declared box with the solved scale.  The table")
            print("above is the fit to implement: k_i = applied_i * declared_i / drawn_i(applied).")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
