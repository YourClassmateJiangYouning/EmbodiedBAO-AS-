"""Measure every vendored prop mesh, so a scene's declared size can be the real one.

The dressing declares a box per item and place_dressing references a prop on top of it without
scaling the prop.  So the declared size has to be the prop's actual size, or the box and the
thing inside it disagree -- which is what tools/measure_props.py found for the one item wired
so far: a declared (0.6, 0.30, 0.5) box around a mesh that is really 0.297 x 0.198 x 0.146 m,
hence a 2x mismatch on every axis.

This prints, for each vendored .usd that is a mesh layer, its size in the stage's Z-up frame
and in the catalogue's user frame (x, height, lateral) -- the frame `scenes.py` is written in --
using scenes.to_user_size() rather than an inverse derived here.

It changes nothing.  Read the table, then set the declared sizes from it.

    ISAAC_PY=/home/ybh/isaacsim/python.sh
    $ISAAC_PY -u tools/measure_assets.py 2>&1 | tee lab_measure_assets.log
    cat asset_measurements.txt
"""
from __future__ import annotations

import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

REPORT = os.path.join(_ROOT, "asset_measurements.txt")
ASSETS = os.path.join(_ROOT, "assets", "isaac", "Props")
_lines: list[str] = []


def emit(text: str = "") -> None:
    _lines.append(text)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def save() -> None:
    try:
        with open(REPORT, "w", encoding="utf-8") as handle:
            handle.write("\n".join(_lines) + "\n")
    except OSError as exc:
        sys.stdout.write(f"could not write {REPORT}: {exc}\n")
        sys.stdout.flush()


def candidates() -> list:
    """Every vendored .usd that is a mesh layer rather than a preview or a collider.

    `.thumb.usd` is a thumbnail; `*_collision*` is convex decomposition, which is not what the
    agent sees.  Both are excluded by name, and the excluded ones are printed so the filter is
    visible rather than silent.
    """
    picked, skipped = [], []
    for root, _dirs, files in os.walk(ASSETS):
        for name in sorted(files):
            if not name.endswith(".usd"):
                continue
            path = os.path.join(root, name)
            if ".thumb." in name or "collision" in name:
                skipped.append(path)
            else:
                picked.append(path)
    return picked, skipped


def main() -> int:
    emit("MARK start")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    emit("MARK SimulationApp up")
    try:
        from pxr import Usd, UsdGeom

        import scenes as sc

        emit("MARK imports done")
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        picked, skipped = candidates()
        emit(f"MARK {len(picked)} asset(s) to measure, {len(skipped)} skipped "
             f"(thumbnails and colliders)")
        for path in skipped:
            emit(f"MARK   skipped {os.path.relpath(path, ASSETS)}")
        emit()
        emit(f"{'asset':<44} {'stage x,y,z (m)':<26} {'catalogue x,height,lateral':<28} note")
        emit("-" * 140)

        for path in picked:
            rel = os.path.relpath(path, ASSETS)
            # The prim path has to be a valid SdfPath, and every one of these filenames
            # contains a dot, which is a property separator.  scene_builder.prim_name() is the
            # sanitiser the rest of the repo uses; this reuses its rule for a file path.
            prim_path = "/World/A/" + "".join(
                c if c.isalnum() else "_" for c in rel)
            holder = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
            holder.GetReferences().AddReference(path)

            measured = None
            used = "referenced prim"
            if holder.IsValid():
                rng = box.ComputeRelativeBound(holder, holder.GetParent()).ComputeAlignedRange()
                if not rng.IsEmpty():
                    lo, hi = rng.GetMin(), rng.GetMax()
                    measured = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
            if measured is None:
                # Instanceable model roots report no children and can bound empty; the geometry
                # is on the prototype.
                prototype = holder.GetPrototype()
                if prototype and prototype.IsValid():
                    used = "prototype"
                    rng = box.ComputeRelativeBound(prototype,
                                                   prototype.GetParent()).ComputeAlignedRange()
                    if not rng.IsEmpty():
                        lo, hi = rng.GetMin(), rng.GetMax()
                        measured = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])

            if measured is None:
                emit(f"{rel:<44} {'no measurable extent':<26} {'-':<28} "
                     f"UNMEASURABLE (via {used})")
                continue

            world = tuple(round(v, 4) for v in measured)
            # to_user(), not a "to_user_size": scenes.py defines to_user for a point and
            # to_world_size for a size, and the size-scale conversion is the point swap again
            # (to_world_size swaps the same two components).  Verified by round trip:
            # catalogue (0.6, 0.3, 0.5) -> world (0.6, 0.5, 0.3) -> back (0.6, 0.3, 0.5).
            user = tuple(round(v, 4) for v in sc.to_user(measured))
            emit(f"{rel:<44} {str(world):<26} {str(user):<28} via {used}")

        emit()
        emit("stage     = the size as the stage stores it, Z-up (x, lateral, height)")
        emit("catalogue = the same size in the frame scenes.py is written in,")
        emit("            (x, height, lateral) -- copy this column into a declared size")
        emit("MARK done")
        return 0
    except BaseException:
        emit("MARK EXCEPTION")
        emit(traceback.format_exc())
        return 1
    finally:
        save()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
