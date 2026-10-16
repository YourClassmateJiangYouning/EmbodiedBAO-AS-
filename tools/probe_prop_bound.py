"""Decide what ``ComputeRelativeBound`` returns for a referenced prop, so the fit can be fixed.

THE QUESTION.  ``scene_builder.fit_reference_to_local_unit`` keeps ``min(1/native)`` because the
per-axis alternative (``k_i = 1/native_i``) was tried on 2026-10-16 and reverted the same night
(commit ``bfd5be7``): it drew ``shelf_left`` and ``checkout`` exactly on their declared boxes and drew
``mug_a2`` 9.2 METRES across.  Two constants are therefore in play -- the item Xform carries the
declared size, and the prop carries a scale of its own -- and the function only produces a correct
answer if it knows how ``native`` was already affected by the first one:

    native_i == declared_i                 -> the bound is the DECLARED box
                                              (the reference collapsed into the parent's units);
                                              filling it is ``k_i = declared_i / native_i = 1``.
    native_i == 1                          -> the bound is ONE LOCAL UNIT
                                              (already normalised); ``k_i = 1`` as well.
    native_i == V_i, the asset's own metres -> the bound is the ASSET's extent, unaffected by the
                                              parent; filling the box is ``k_i = declared_i / V_i``,
                                              the version that drew the mugs 100x too large.
    native_i == declared_i * V_i           -> the bound is the asset SCALED BY the parent;
                                              filling it is ``k_i = 1 / V_i``.

The four cases are told apart by ONE number per axis: measure the bound twice, once with the prop's
own scale at 1 and once at 2.  If it doubles, the bound is linear in the prop's scale and the parent
scale is part of it; if it does not, the bound does not see the prop.  This script prints that, plus
``bound / declared``, for every asset a scene actually references.

WHAT TO DO WITH THE OUTPUT.  ``bound(scale=1) / declared`` is the multiplier that turns the declared
box into the measured bound.  A column of 1.000 means the bound is the declared box; a column that
tracks ``PROP_MEASUREMENTS x ASSET_SCALE`` means the bound is the asset's own extent.  Whichever it
is, the correct ``k`` follows from the table above, and the test that pins it is the per-axis
``drawn == declared`` comparison ``test_bao_scenes.py`` now applies to the declared sizes.

Read-only with respect to the repository: it writes nothing and imports nothing from it except
``scenes`` (pure data).  It builds a throwaway stage in memory.

    ISAAC_PY=/home/ybh/isaacsim/python.sh
    $ISAAC_PY -u tools/probe_prop_bound.py 2>&1 | tee diag_logs/prop_bound_probe.log
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Gf, Usd, UsdGeom

        import scenes as sc
        import scene_builder as sb

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Dressing")

        def bound_of(proxy, ancestor):
            # A FRESH cache per measurement: BBoxCache does listen for change notifications, but a
            # scale written between two reads is exactly the case where a stale cache would silently
            # answer the previous question -- and the whole point here is to compare two scales.
            fresh = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            try:
                rng = fresh.ComputeRelativeBound(proxy, ancestor).ComputeAlignedRange()
            except Exception as exc:  # noqa: BLE001
                return None, f"{type(exc).__name__}: {exc}"
            if rng.IsEmpty():
                return None, "empty bound"
            lo, hi = rng.GetMin(), rng.GetMax()
            return tuple(float(hi[a] - lo[a]) for a in range(3)), ""

        print(f"{'asset':44s} {'declared':28s} {'bound@s=1':28s} {'bound@s=2':28s} ratio")
        seen = set()
        for scene in sc.SCENE_ORDER:
            for index, item in enumerate(sc.SCENES[scene]["dressing"]):
                local = sb.prop_asset_for(item.get("asset")) if item.get("asset") else None
                if not local or local in seen:
                    continue
                seen.add(local)
                name = os.path.basename(local)
                path = "/World/Dressing/" + sb.prim_name(
                    f"{scene}_{item['name']}_{index}")
                item_xform = UsdGeom.Xform.Define(stage, path)
                declared = tuple(float(v) for v in sc.to_world_size(item["size"]))
                item_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
                item_xform.AddScaleOp().Set(Gf.Vec3f(*declared))
                proxy = UsdGeom.Xform.Define(stage, f"{path}/prop").GetPrim()
                proxy.GetReferences().AddReference(local)
                sb.apply_translate(proxy, (0.0, 0.0, 0.0))
                one, err1 = bound_of(proxy, item_xform.GetPrim())
                sb.apply_scale(proxy, (2.0, 2.0, 2.0))
                two, err2 = bound_of(proxy, item_xform.GetPrim())
                if one is None or two is None:
                    print(f"{name:44s} {err1 or err2}")
                    continue
                ratio = tuple(round(two[a] / one[a], 6) if one[a] else 0.0 for a in range(3))
                print(f"{name:44s} "
                      f"{tuple(round(v, 4) for v in declared)!s:28s} "
                      f"{tuple(round(v, 4) for v in one)!s:28s} "
                      f"{tuple(round(v, 4) for v in two)!s:28s} {ratio}")
        print()
        print("ratio == (2.0, 2.0, 2.0)  -> the bound is linear in the prop's scale")
        print("bound@s=1 == declared      -> the bound is the DECLARED box (k = 1)")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
